"""LocalEmotionInferencer: run the emotion model in-process (on this laptop).

Wraps ``emotion_detection_action.EmotionDetector`` and exposes a small
``detect_emotion()`` -> dict interface (plus ``start()`` / ``stop()``) that the
Gemini ``detect_emotion`` tool consumes directly — emotion inference runs
locally, with no network hop.

Two uses of the same instance (per the end-state design):
  * the continuous behaviour loop calls :meth:`process` every frame and drives
    motion from the returned dict; each call updates the "latest" result;
  * the Gemini tool calls :meth:`detect_emotion`, which returns that latest
    result (falling back to a one-shot capture when no loop is running).

Audio is downmixed to mono for emotion2vec (which wants 16 kHz mono float32).
Resampling is intentionally NOT done here — the Reachy ReSpeaker default is
16 kHz; if :meth:`sensor_info` reports another rate on real hardware, add a
resample step then (kept out now to avoid speculative code).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np

from reachy_emotion.local_capture import ReachySensorCapture

logger = logging.getLogger(__name__)

_UNCLEAR = "unclear"
_DEFAULT_WARMUP_FRAMES = 16


def _to_mono(audio: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Downmix multi-channel audio to mono float32; pass 1-D audio through."""
    if audio is None:
        return None
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        # Reachy media returns (num_samples, channels); average the channels.
        a = a.mean(axis=1)
    return a


def _result_to_dict(result: Any) -> dict:
    """Map a NeuralEmotionResult to the flat dict the Gemini tool consumes."""
    metrics = getattr(result, "metrics", None) or {}
    scores = getattr(result, "emotion_scores", None) or {}
    return {
        "dominant_emotion": result.dominant_emotion,
        "confidence": round(float(result.confidence), 2),
        "confidence_scores": {k: round(float(v), 3) for k, v in scores.items()},
        "stress": round(float(metrics.get("stress", 0.0)), 2),
        "engagement": round(float(metrics.get("engagement", 0.0)), 2),
        "arousal": round(float(metrics.get("arousal", 0.0)), 2),
    }


class LocalEmotionInferencer:
    """In-process multimodal emotion inference for reachy-emotion.

    Args:
        mini: Connected ReachyMini (used to build a capture if none is given).
        model_path: Path to the fine-tuned two-tower checkpoint (.pt).
        device: Torch device for the model ("mps" | "cpu" | "cuda").
        warmup_frames: Results before this many frames carry ``"warming": True``
            (the SDK repeat-pads a cold 16-frame buffer, so early predictions are
            unreliable).
        capture: Optional pre-built capture (injected in tests).
        detector: Optional pre-built detector (injected in tests; skips model load).
    """

    def __init__(
        self,
        mini: Any = None,
        model_path: Optional[str] = None,
        device: str = "mps",
        warmup_frames: int = _DEFAULT_WARMUP_FRAMES,
        capture: Optional[ReachySensorCapture] = None,
        detector: Any = None,
    ) -> None:
        self._mini = mini
        self._model_path = model_path
        self._device = device
        self._warmup_frames = warmup_frames
        self._capture = capture
        self._detector = detector
        self._latest: Optional[dict] = None
        self._frame_count = 0
        # Serialises process() so a background warm-up loop and an on-demand
        # detect_emotion() call can't run the detector concurrently (its rolling
        # buffers + GRU state are not thread-safe).
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Build (if needed) and initialise the detector and capture."""
        if self._detector is None:
            # Lazy import: keeps this module importable (and unit-testable)
            # without loading torch / the SDK.
            from emotion_detection_action import Config, EmotionDetector

            cfg = Config(
                two_tower_device=self._device,
                two_tower_model_path=self._model_path,
                two_tower_pretrained=True,
            )
            self._detector = EmotionDetector(cfg)
        self._detector.initialize()
        if self._capture is None and self._mini is not None:
            self._capture = ReachySensorCapture(self._mini)
        logger.info("LocalEmotionInferencer ready (device=%s)", self._device)

    def stop(self) -> None:
        if self._detector is not None:
            try:
                self._detector.shutdown()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("detector.shutdown failed: %s", exc)

    def reset(self) -> None:
        """Reset temporal state + warm-up counter (call on subject change)."""
        self._frame_count = 0
        self._latest = None
        if self._detector is not None:
            self._detector.reset()

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def process(self, frame_bgr: np.ndarray, audio: Optional[np.ndarray]) -> dict:
        """Run one frame (+audio) through the model and return the result dict.

        Updates the stored "latest" result that :meth:`detect_emotion` returns.
        """
        if self._detector is None:
            raise RuntimeError("LocalEmotionInferencer.start() must be called first.")
        with self._lock:
            result = self._detector.process_frame(frame_bgr, _to_mono(audio))
            self._frame_count += 1
            out = _result_to_dict(result)
            if self._frame_count < self._warmup_frames:
                out["warming"] = True
            self._latest = out
        return out

    def detect_emotion(self) -> dict:
        """Return the latest emotion result (Gemini ``detect_emotion`` tool seam).

        Prefers the result kept fresh by the continuous loop; if none exists yet
        (e.g. text mode with no loop), does a single capture+inference.
        """
        # Snapshot the reference once: the continuous behaviour loop may replace
        # or reset() self._latest from another thread between the check and the
        # return. Reading into a local avoids handing back None mid-reset.
        latest = self._latest
        if latest is not None:
            return latest
        if self._capture is None:
            return {"dominant_emotion": _UNCLEAR, "confidence": 0.0, "note": "no capture available"}
        frame, audio = self._capture.read()
        if frame is None:
            return {"dominant_emotion": _UNCLEAR, "confidence": 0.0, "note": "no frame available"}
        return self.process(frame, audio)

    @property
    def latest(self) -> Optional[dict]:
        return self._latest
