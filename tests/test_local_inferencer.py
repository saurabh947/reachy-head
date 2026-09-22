"""Unit tests for LocalEmotionInferencer (W3-B1).

Hardware/model-free: a fake detector stands in for EmotionDetector, so no torch
or SDK load is needed.
"""

from __future__ import annotations

import numpy as np

from reachy_emotion.local_inferencer import LocalEmotionInferencer, _to_mono


class _FakeResult:
    def __init__(self, dom, scores, metrics, conf):
        self.dominant_emotion = dom
        self.emotion_scores = scores
        self.metrics = metrics
        self.confidence = conf


class _FakeDetector:
    def __init__(self):
        self.initialized = False
        self.calls = []  # (frame, audio) seen by process_frame
        self.reset_count = 0

    def initialize(self):
        self.initialized = True

    def process_frame(self, frame, audio, timestamp=None):
        self.calls.append((frame, audio))
        return _FakeResult(
            "happy",
            {"happy": 0.912, "sad": 0.088},
            {"stress": 0.2, "engagement": 0.7, "arousal": 0.5},
            0.912,
        )

    def reset(self):
        self.reset_count += 1

    def shutdown(self):
        self.initialized = False


class _FakeCapture:
    def __init__(self, frame, audio=None):
        self._frame = frame
        self._audio = audio

    def read(self):
        return self._frame, self._audio


def _frame():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_to_mono_downmixes_2d_and_passes_1d():
    assert _to_mono(None) is None
    mono = _to_mono(np.ones(10, dtype=np.float32))
    assert mono.ndim == 1 and mono.shape == (10,)
    stereo = np.stack([np.ones(10), np.zeros(10)], axis=1).astype(np.float32)  # (10, 2)
    down = _to_mono(stereo)
    assert down.ndim == 1 and down.shape == (10,)
    assert np.allclose(down, 0.5)


def test_process_returns_cloud_compatible_dict():
    det = _FakeDetector()
    inf = LocalEmotionInferencer(detector=det, warmup_frames=0)
    inf.start()
    out = inf.process(_frame(), None)
    assert set(out) >= {
        "dominant_emotion", "confidence", "confidence_scores",
        "stress", "engagement", "arousal",
    }
    assert out["dominant_emotion"] == "happy"
    assert out["confidence"] == 0.91  # rounded to 2dp
    assert out["engagement"] == 0.7


def test_process_downmixes_audio_before_detector():
    det = _FakeDetector()
    inf = LocalEmotionInferencer(detector=det, warmup_frames=0)
    inf.start()
    stereo = np.ones((8, 2), dtype=np.float32)
    inf.process(_frame(), stereo)
    _, audio_seen = det.calls[-1]
    assert audio_seen.ndim == 1 and audio_seen.shape == (8,)


def test_warmup_flag_then_clears():
    # warming is set while frame_count < warmup_frames; with warmup=3 that is
    # the first two frames, then it clears from the 3rd onward.
    det = _FakeDetector()
    inf = LocalEmotionInferencer(detector=det, warmup_frames=3)
    inf.start()
    assert inf.process(_frame(), None).get("warming") is True   # frame_count=1 < 3
    assert inf.process(_frame(), None).get("warming") is True   # frame_count=2 < 3
    assert "warming" not in inf.process(_frame(), None)          # frame_count=3, not < 3


def test_detect_emotion_returns_latest_after_process():
    det = _FakeDetector()
    inf = LocalEmotionInferencer(detector=det, warmup_frames=0)
    inf.start()
    inf.process(_frame(), None)
    assert inf.detect_emotion()["dominant_emotion"] == "happy"


def test_detect_emotion_one_shot_fallback_via_capture():
    det = _FakeDetector()
    inf = LocalEmotionInferencer(detector=det, capture=_FakeCapture(_frame()), warmup_frames=0)
    inf.start()
    # no process() yet → falls back to a one-shot capture read
    assert inf.detect_emotion()["dominant_emotion"] == "happy"
    assert len(det.calls) == 1


def test_detect_emotion_unclear_when_no_capture_and_no_latest():
    det = _FakeDetector()
    inf = LocalEmotionInferencer(detector=det, warmup_frames=0)
    inf.start()
    out = inf.detect_emotion()
    assert out["dominant_emotion"] == "unclear"
    assert out["confidence"] == 0.0


def test_reset_clears_latest_and_counter():
    det = _FakeDetector()
    inf = LocalEmotionInferencer(detector=det, warmup_frames=0)
    inf.start()
    inf.process(_frame(), None)
    inf.reset()
    assert inf.latest is None
    assert det.reset_count == 1
