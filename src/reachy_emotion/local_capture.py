"""Local sensor capture from the Reachy Mini daemon (camera + mic).

Per the sensor-path decision, emotion sensing reads the Reachy camera and mic
through the daemon (``mini.media.*``), not the laptop webcam. This replaces the
cloud client's per-call camera grab (which released/re-acquired the device on
every call) with a long-lived reader whose frames feed the local EmotionDetector.

``mini.media.get_frame()`` returns a **BGR** uint8 frame (per its docstring),
which is exactly what ``EmotionDetector.process_frame`` expects — so frames pass
through without any colour conversion.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Frame = Optional[np.ndarray]
Audio = Optional[np.ndarray]

# Small idle wait when the camera has no frame yet, to avoid a busy-spin loop.
_IDLE_SLEEP_S = 0.005


class ReachySensorCapture:
    """Thin reader over ``mini.media`` yielding ``(bgr_frame, audio_chunk)``.

    Args:
        mini: A connected ReachyMini instance (owned by the caller).
    """

    def __init__(self, mini: Any) -> None:
        self._mini = mini

    def sensor_info(self) -> dict:
        """Return the mic samplerate and channel count.

        Used by the inferencer to adapt audio for emotion2vec (which wants
        16 kHz mono). Values are ``None`` when the audio device is not up
        (``mini.media`` returns ``-1`` in that case).
        """
        media = self._mini.media

        def _safe(fn: Any) -> Optional[int]:
            try:
                v = fn()
            except Exception:
                return None
            return int(v) if v is not None and v >= 0 else None

        return {
            "audio_samplerate": _safe(media.get_input_audio_samplerate),
            "audio_channels": _safe(media.get_input_channels),
        }

    def read(self) -> Tuple[Frame, Audio]:
        """Return the latest ``(bgr_frame, audio_chunk)``; either may be ``None``."""
        media = self._mini.media
        frame: Frame = None
        try:
            frame = media.get_frame()
        except Exception as exc:
            logger.debug("get_frame failed: %s", exc)
        audio: Audio = None
        try:
            audio = media.get_audio_sample()
        except Exception as exc:
            logger.debug("get_audio_sample failed: %s", exc)
        return frame, audio

    def stream(
        self, stop: Optional[threading.Event] = None
    ) -> Iterator[Tuple[np.ndarray, Audio]]:
        """Yield ``(bgr_frame, audio_chunk)`` tuples, skipping frames that are ``None``.

        Runs until *stop* is set (or forever if *stop* is ``None``).
        """
        while stop is None or not stop.is_set():
            frame, audio = self.read()
            if frame is None:
                time.sleep(_IDLE_SLEEP_S)
                continue
            yield frame, audio
