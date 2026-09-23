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
import time
from typing import Any, Optional

import numpy as np

from reachy_emotion.local_capture import ReachySensorCapture

logger = logging.getLogger(__name__)

_UNCLEAR = "unclear"
# The SDK samples its 16 frames evenly over the last 3 s of calls
# (Config.two_tower_video_window_seconds); until the calls span that window the
# clip repeats a few frames, so results are indicative only.
_DEFAULT_WARMUP_S = 3.0
# A continuous loop refreshes the latest result every ~0.5 s; anything older means
# no loop is running (text mode) or the camera stopped, so take a fresh reading.
_FRESH_S = 2.0


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
        warmup_s: Results carry ``"warming": True`` until consecutive calls span
            this many seconds (the SDK's frame window; a gap longer than it
            empties the window, so warm-up restarts — text mode's one-shot
            readings are always warming).
        capture: Optional pre-built capture (injected in tests).
        detector: Optional pre-built detector (injected in tests; skips model load).
    """

    def __init__(
        self,
        mini: Any = None,
        model_path: Optional[str] = None,
        device: str = "mps",
        warmup_s: float = _DEFAULT_WARMUP_S,
        capture: Optional[ReachySensorCapture] = None,
        detector: Any = None,
    ) -> None:
        self._mini = mini
        self._model_path = model_path
        self._device = device
        self._warmup_s = warmup_s
        self._capture = capture
        self._detector = detector
        # (result, monotonic time) stored as one tuple so readers never pair a
        # result with another result's timestamp.
        self._latest: tuple[Optional[dict], float] = (None, 0.0)
        # Start and last time of the current unbroken run of process() calls.
        self._window_start: Optional[float] = None
        self._last_call = 0.0
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
        """Reset the frame/audio window + warm-up (call on subject change)."""
        # Under the lock: a reset between the SDK's add-frame and infer steps
        # would make process_frame() return None mid-call.
        with self._lock:
            self._window_start = None
            self._latest = (None, 0.0)
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
            now = time.monotonic()
            if self._window_start is None or now - self._last_call > self._warmup_s:
                self._window_start = now  # first call, or a gap emptied the SDK window
            self._last_call = now
            out = _result_to_dict(result)
            if now - self._window_start < self._warmup_s:
                out["warming"] = True
            self._latest = (out, now)
        return out

    def detect_emotion(self) -> dict:
        """Return the current emotion result (Gemini ``detect_emotion`` tool seam).

        Returns the result the continuous loop keeps fresh; if there is none or it
        is stale (text mode has no loop; or the camera stopped), does a fresh
        capture+inference instead of replaying an old reading.
        """
        # Snapshot once: the continuous loop may replace or reset() self._latest
        # from another thread between the check and the return.
        latest, stamp = self._latest
        if latest is not None and time.monotonic() - stamp < _FRESH_S:
            return latest
        if self._capture is None:
            return {"dominant_emotion": _UNCLEAR, "confidence": 0.0, "note": "no capture available"}
        frame, audio = self._capture.read()
        if frame is None:
            return {"dominant_emotion": _UNCLEAR, "confidence": 0.0, "note": "no frame available"}
        return self.process(frame, audio)

    @property
    def latest(self) -> Optional[dict]:
        return self._latest[0]
