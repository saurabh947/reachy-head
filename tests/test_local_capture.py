"""Unit tests for ReachySensorCapture (W3-A1).

Hardware-free: a hand-rolled fake stands in for ``mini.media``.
"""

from __future__ import annotations

import itertools

import numpy as np

from reachy_emotion.local_capture import ReachySensorCapture


class _FakeMedia:
    def __init__(self, frames, audios=None, samplerate=16000, channels=1):
        self._frames = list(frames)
        self._audios = list(audios) if audios is not None else None
        self._sr = samplerate
        self._ch = channels
        self._i = 0

    def _next(self, seq):
        if not seq:
            return None
        val = seq[min(self._i, len(seq) - 1)]
        return val

    def get_frame(self):
        val = self._next(self._frames)
        self._i += 1
        return val

    def get_audio_sample(self):
        return None if self._audios is None else self._next(self._audios)

    def get_input_audio_samplerate(self):
        return self._sr

    def get_input_channels(self):
        return self._ch


class _FakeMini:
    def __init__(self, media):
        self.media = media


def _frame():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_read_passes_frame_and_audio_through():
    f = _frame()
    a = np.ones(3, dtype=np.float32)
    cap = ReachySensorCapture(_FakeMini(_FakeMedia([f], [a])))
    frame, audio = cap.read()
    assert frame is f
    assert audio is a


def test_read_handles_none():
    cap = ReachySensorCapture(_FakeMini(_FakeMedia([None])))
    frame, audio = cap.read()
    assert frame is None
    assert audio is None


def test_sensor_info_reports_samplerate_and_channels():
    cap = ReachySensorCapture(_FakeMini(_FakeMedia([_frame()], samplerate=48000, channels=6)))
    info = cap.sensor_info()
    assert info == {"audio_samplerate": 48000, "audio_channels": 6}


def test_sensor_info_maps_negative_one_to_none():
    # mini.media returns -1 when the audio device is not initialised.
    cap = ReachySensorCapture(_FakeMini(_FakeMedia([_frame()], samplerate=-1, channels=-1)))
    info = cap.sensor_info()
    assert info == {"audio_samplerate": None, "audio_channels": None}


def test_stream_skips_none_frames():
    # First get_frame() returns None (skipped), then real frames.
    f = _frame()
    cap = ReachySensorCapture(_FakeMini(_FakeMedia([None, f, f, f])))
    got = list(itertools.islice(cap.stream(), 2))
    assert len(got) == 2
    assert all(frame is f for frame, _ in got)
