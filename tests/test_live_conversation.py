"""Tests for the Gemini Live conversation helpers (W4-C1).

Covers the pure audio/format + tool logic, and drives the async receive loop
via asyncio.run (no pytest-asyncio needed). The full streaming session needs a
real API key + audio hardware and is out of scope here.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import numpy as np

from reachy_emotion import conversation_app, live_conversation as lc


# --------------------------------------------------------------------------- #
# Pure helpers                                                                  #
# --------------------------------------------------------------------------- #

def test_to_mono_downmixes():
    stereo = np.ones((10, 2), dtype=np.float32)
    assert lc._to_mono(stereo).shape == (10,)


def test_resample_linear_changes_length_and_is_identity_when_equal():
    x = np.linspace(0, 1, 300).astype(np.float32)
    assert lc._resample_linear(x, 24000, 16000).shape[0] == 200
    same = lc._resample_linear(x, 16000, 16000)
    assert same.shape[0] == 300


def test_pcm16_roundtrip():
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    back = lc._pcm16_to_float(lc._float_to_pcm16(x))
    assert np.allclose(back, x, atol=1e-3)


def test_mic_to_mono16k_downmixes_stereo_no_resample():
    sample = np.ones((320, 2), dtype=np.float32)  # stereo @ 16k
    mono = lc._mic_to_mono16k(sample, src_rate=16000)
    assert mono.shape == (320,)
    assert np.allclose(mono, 1.0)


def test_mic_to_mono16k_resamples_when_needed():
    sample = np.zeros(480, dtype=np.float32)  # 480 @ 48k -> 160 @ 16k
    mono = lc._mic_to_mono16k(sample, src_rate=48000)
    assert mono.shape == (160,)


def test_audio_ring_accumulates_then_drains_once():
    ring = lc._AudioRing()
    assert ring.drain() is None
    ring.append(np.ones(10, dtype=np.float32))
    ring.append(np.ones(5, dtype=np.float32))
    out = ring.drain()
    assert out.shape == (15,)
    assert ring.drain() is None  # cleared after drain


def test_audio_ring_bounded_to_max_seconds():
    ring = lc._AudioRing(max_seconds=0.001, rate=16000)  # cap = 16 samples
    ring.append(np.ones(100, dtype=np.float32))
    assert ring.drain().shape == (16,)


def test_live_pcm_to_speaker_resamples_24k_to_16k_shape():
    data = np.zeros(240, dtype="<i2").tobytes()  # 240 samples @ 24k
    out = lc._live_pcm_to_speaker(data)
    assert out.shape == (160, 1)  # 240 * 16000/24000
    assert out.dtype == np.float32


def test_encode_jpeg_returns_bytes():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    jpeg = lc._encode_jpeg(frame)
    assert isinstance(jpeg, bytes) and len(jpeg) > 0


def test_emotion_tool_result_dispatch():
    class _FC:
        def __init__(self, name):
            self.name = name
            self.id = "x"

    client = MagicMock()
    client.detect_emotion.return_value = {"dominant_emotion": "sad"}
    assert lc._emotion_tool_result(_FC("detect_emotion"), client) == {"dominant_emotion": "sad"}
    assert "error" in lc._emotion_tool_result(_FC("other_tool"), client)
    assert "error" in lc._emotion_tool_result(_FC("detect_emotion"), None)


# --------------------------------------------------------------------------- #
# Async receive loop                                                            #
# --------------------------------------------------------------------------- #

class _FC:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.args = {}


class _ToolCall:
    def __init__(self, fcs):
        self.function_calls = fcs


class _Msg:
    def __init__(self, data=None, tool_call=None, server_content=None):
        self.data = data
        self.tool_call = tool_call
        self.server_content = server_content


class _FakeSession:
    """Serves one turn's worth of messages, then reports the session idle/closed
    (receive() yields nothing) — mirroring the real turn-then-end behaviour."""

    def __init__(self, msgs):
        self._msgs = msgs
        self._served = False
        self.tool_responses = []

    async def receive(self):
        if self._served:
            return
        self._served = True
        for m in self._msgs:
            yield m

    async def send_tool_response(self, function_responses):
        self.tool_responses.append(function_responses)


def test_receive_loop_plays_audio_and_answers_tool(monkeypatch):
    # _react_to_emotion would try to load RecordedMoves — stub it to a no-op.
    monkeypatch.setattr(conversation_app, "_get_recorded_moves", lambda lib: None)

    mini = MagicMock()
    emotion_client = MagicMock()
    emotion_client.detect_emotion.return_value = {"dominant_emotion": "happy", "confidence": 0.9}

    audio = np.zeros(240, dtype="<i2").tobytes()
    session = _FakeSession([
        _Msg(data=audio),
        _Msg(tool_call=_ToolCall([_FC("id1", "detect_emotion")])),
    ])

    asyncio.run(lc._receive_loop(session, mini, threading.Event(), emotion_client))

    mini.media.push_audio_sample.assert_called_once()
    assert len(session.tool_responses) == 1
    fr = session.tool_responses[0][0]
    assert fr.name == "detect_emotion"
    assert fr.response == {"dominant_emotion": "happy", "confidence": 0.9}
    emotion_client.detect_emotion.assert_called_once()


def test_receive_loop_stops_when_event_set(monkeypatch):
    monkeypatch.setattr(conversation_app, "_get_recorded_moves", lambda lib: None)
    mini = MagicMock()
    stop = threading.Event()
    stop.set()  # already stopped → first message breaks out
    session = _FakeSession([_Msg(data=np.zeros(10, dtype="<i2").tobytes())])

    asyncio.run(lc._receive_loop(session, mini, stop, MagicMock()))

    mini.media.push_audio_sample.assert_not_called()


class _CollectSession:
    def __init__(self):
        self.sent = []

    async def send_realtime_input(self, **kw):
        self.sent.append(kw)


def test_emotion_video_loop_warms_model_and_sends_video():
    # A continuous warm loop must feed the local model AND stream video to Gemini.
    stop = threading.Event()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    mini = MagicMock()
    mini.media.get_frame.return_value = frame

    emotion_client = MagicMock()
    emotion_client.process.side_effect = lambda f, a: stop.set()  # exit after 1 pass

    ring = lc._AudioRing()
    ring.append(np.ones(8, dtype=np.float32))
    session = _CollectSession()

    asyncio.run(lc._emotion_video_loop(session, mini, stop, emotion_client, ring))

    emotion_client.process.assert_called_once()          # model kept warm
    assert any("video" in kw for kw in session.sent)     # first frame past throttle → sent
