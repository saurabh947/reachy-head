"""Gemini 3.8 Live conversation loop for reachy-emotion (voice mode).

A persistent, bidirectional streaming session (``client.aio.live.connect``) with
the ``gemini-3.8-live`` model replaces the turn-based STT → chat → TTS loop:

  * Reachy mic audio (16 kHz PCM) streams into the session;
  * Reachy camera frames (JPEG, <= 1 fps per the Live spec) stream in too;
  * the model streams audio back (24 kHz PCM), played on Reachy's speaker;
  * the local emotion model runs CONTINUOUSLY in the background (kept warm so its
    16-frame + audio buffers fill), and when the model calls the ``detect_emotion``
    tool it gets that fresh multimodal read + Reachy plays the matching RecordedMove.

The single mic stream (a consume-once GStreamer queue) is read once and fanned
out to both Gemini and the emotion model via a shared ring buffer, so they don't
starve each other.

Audio contracts (verified): Live input = raw 16-bit PCM @ 16 kHz mono; Live
output = raw 16-bit PCM @ 24 kHz; Reachy ``push_audio_sample`` wants float32
@ 16 kHz (see tts_announcer). So output is resampled 24 kHz -> 16 kHz.

The pure format/tool helpers below are unit-tested; the async streaming loop
itself needs a real API key + audio hardware to exercise end to end.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Optional

import numpy as np

from reachy_emotion.conversation_app import _load_api_key, _load_env, _load_system_prompt

logger = logging.getLogger(__name__)

DEFAULT_LIVE_MODEL = "gemini-3.8-live"

# Live API audio rates (raw little-endian 16-bit PCM).
_LIVE_INPUT_RATE = 16000
_LIVE_OUTPUT_RATE = 24000
# Reachy Mini speaker rate (matches tts_announcer._SPEAKER_SAMPLE_RATE).
_SPEAKER_RATE = 16000

_VIDEO_PERIOD_S = 1.0  # Live accepts <= 1 image/s
_EMOTION_LOOP_GAP_S = 0.1  # idle between emotion inferences — leaves CPU for smooth audio


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)                                                    #
# --------------------------------------------------------------------------- #

def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        a = a.mean(axis=1)
    return a


def _resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resample (adequate for speech, no extra deps)."""
    x = np.asarray(x, dtype=np.float32)
    if src_rate == dst_rate or x.size == 0:
        return x
    n_out = int(round(x.shape[0] * dst_rate / src_rate))
    if n_out <= 0:
        return x[:0]
    idx = np.linspace(0.0, x.shape[0] - 1, n_out)
    return np.interp(idx, np.arange(x.shape[0]), x).astype(np.float32)


def _float_to_pcm16(x: np.ndarray) -> bytes:
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def _pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _mic_to_mono16k(sample: np.ndarray, src_rate: int) -> np.ndarray:
    """Reachy mic float32 (stereo) -> mono float32 @ 16 kHz.

    Used for BOTH Gemini (after ``_float_to_pcm16``) and the emotion model
    (emotion2vec also wants 16 kHz mono), so the mic is converted once.
    """
    mono = _to_mono(sample)
    if src_rate != _LIVE_INPUT_RATE:
        mono = _resample_linear(mono, src_rate, _LIVE_INPUT_RATE)
    return mono


def _live_pcm_to_speaker(data: bytes) -> np.ndarray:
    """Live output bytes (24 kHz PCM) -> Reachy push_audio_sample array (16 kHz, (N,1))."""
    floats = _pcm16_to_float(data)
    floats = _resample_linear(floats, _LIVE_OUTPUT_RATE, _SPEAKER_RATE)
    return floats.reshape(-1, 1)


def _encode_jpeg(bgr_frame: np.ndarray) -> Optional[bytes]:
    import cv2

    ok, buf = cv2.imencode(".jpg", bgr_frame)
    return buf.tobytes() if ok else None


def _emotion_tool_result(function_call: Any, emotion_client: Any) -> dict:
    """Compute the result dict for one Live function call."""
    if function_call.name != "detect_emotion":
        return {"error": f"Unknown tool: {function_call.name}"}
    if emotion_client is None:
        return {"error": "emotion client not available"}
    return emotion_client.detect_emotion()


class _AudioRing:
    """Thread-safe accumulator of mono float32 audio.

    The Gemini pump appends every mic chunk; the emotion loop drains the audio
    captured since its last inference. Bounded to *max_seconds* so a slow drain
    can't grow unbounded.
    """

    def __init__(self, max_seconds: float = 3.0, rate: int = _LIVE_INPUT_RATE) -> None:
        self._chunks: list[np.ndarray] = []
        self._max = int(max_seconds * rate)
        self._lock = threading.Lock()

    def append(self, mono: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(mono)

    def drain(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._chunks:
                return None
            out = np.concatenate(self._chunks)
            self._chunks.clear()
        return out[-self._max:] if out.shape[0] > self._max else out


def _load_live_model() -> str:
    """Return GEMINI_LIVE_MODEL from env, or the default live model."""
    import os

    _load_env()
    return os.environ.get("GEMINI_LIVE_MODEL", "").strip() or DEFAULT_LIVE_MODEL


def _build_config(system_prompt: str) -> Any:
    """Build the LiveConnectConfig (audio out + the detect_emotion tool)."""
    from google.genai import types

    from reachy_emotion.gemini_bridge import _DETECT_EMOTION_SCHEMA

    tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=_DETECT_EMOTION_SCHEMA["name"],
            description=_DETECT_EMOTION_SCHEMA["description"],
            parameters=types.Schema(type="OBJECT", properties={}),
        )
    ])
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=system_prompt,
        tools=[tool],
    )


# --------------------------------------------------------------------------- #
# Async streaming tasks                                                         #
# --------------------------------------------------------------------------- #

async def _audio_pump(
    session: Any, mini: Any, stop: threading.Event, src_rate: int, ring: _AudioRing
) -> None:
    """Read the mic once, send it to Gemini, and fan a copy to the emotion ring."""
    from google.genai import types

    while not stop.is_set():
        # get_audio_sample() = appsink.try_pull_sample(20 ms): it blocks up to
        # ~20 ms for the next buffer, then returns it (or None when idle). Draining
        # it back-to-back keeps us in real time; a fixed sleep here makes us fall
        # behind and the appsink (drop=True) discards buffers, garbling the audio.
        sample = await asyncio.to_thread(mini.media.get_audio_sample)
        if sample is None or not np.asarray(sample).size:
            continue
        mono = _mic_to_mono16k(sample, src_rate)
        ring.append(mono)  # fan out to the local emotion model
        await session.send_realtime_input(
            audio=types.Blob(data=_float_to_pcm16(mono), mime_type=f"audio/pcm;rate={_LIVE_INPUT_RATE}")
        )


async def _emotion_video_loop(
    session: Any, mini: Any, stop: threading.Event, emotion_client: Any, ring: _AudioRing
) -> None:
    """Keep the local emotion model warm and stream video to Gemini at <= 1 fps.

    A single frame reader feeds both the emotion model (every iteration, with the
    audio captured since last time) and Gemini (throttled), so they don't compete
    for the camera. Inference runs in a worker thread; on MPS one pass is ~0.3 s,
    so the model naturally warms over a few seconds.
    """
    from google.genai import types

    last_video = 0.0
    while not stop.is_set():
        frame = await asyncio.to_thread(mini.media.get_frame)
        if frame is None:
            await asyncio.sleep(0.03)
            continue

        if emotion_client is not None:
            audio = ring.drain()
            try:
                await asyncio.to_thread(emotion_client.process, frame, audio)
            except Exception as exc:
                logger.debug("emotion process failed: %s", exc)

        now = time.monotonic()
        if now - last_video >= _VIDEO_PERIOD_S:
            jpeg = await asyncio.to_thread(_encode_jpeg, frame)
            if jpeg:
                await session.send_realtime_input(
                    video=types.Blob(data=jpeg, mime_type="image/jpeg")
                )
            last_video = now

        # Leave CPU headroom for smooth audio (emotion needs only ~2 fps).
        await asyncio.sleep(_EMOTION_LOOP_GAP_S)


async def _receive_loop(session: Any, mini: Any, stop: threading.Event, emotion_client: Any) -> None:
    """Play model audio and answer tool calls, turn after turn, until stop is set.

    ``session.receive()`` yields one *complete model turn* then ends (it breaks
    on ``turn_complete``), so it is re-entered in an outer loop to keep the
    conversation going. If a ``receive()`` yields nothing the session has closed
    and the loop exits (rather than spinning).
    """
    from google.genai import types

    from reachy_emotion.conversation_app import _react_to_emotion

    while not stop.is_set():
        got = False
        async for msg in session.receive():
            got = True
            if stop.is_set():
                return

            # Audio out → Reachy speaker.
            if getattr(msg, "data", None):
                samples = _live_pcm_to_speaker(msg.data)
                await asyncio.to_thread(mini.media.push_audio_sample, samples)

            # Tool calls → local emotion model (+ physical reaction).
            tool_call = getattr(msg, "tool_call", None)
            if tool_call and tool_call.function_calls:
                responses = []
                for fc in tool_call.function_calls:
                    result = _emotion_tool_result(fc, emotion_client)
                    logger.info("detect_emotion → %s", result)
                    if result.get("dominant_emotion"):
                        _react_to_emotion(result, mini)
                    responses.append(
                        types.FunctionResponse(id=fc.id, name=fc.name, response=result)
                    )
                await session.send_tool_response(function_responses=responses)

            # Log transcripts when present (handy while debugging).
            content = getattr(msg, "server_content", None)
            if content is not None and getattr(content, "output_transcription", None):
                text = getattr(content.output_transcription, "text", None)
                if text:
                    logger.info("Reachy → %s", text)

        if not got:
            break  # receive() yielded nothing → the session has closed


async def run_live_conversation(
    mini: Any,
    stop_event: threading.Event,
    system_prompt: str | None = None,
    model: str | None = None,
    emotion_client: Any = None,
) -> None:
    """Drive a Gemini 3.8 Live voice conversation until *stop_event* is set.

    Args:
        mini: Connected ReachyMini instance (owned by caller).
        stop_event: Set to exit the session cleanly.
        system_prompt: Override the system prompt (else env / built-in default).
        model: Live model id (else GEMINI_LIVE_MODEL env / DEFAULT_LIVE_MODEL).
        emotion_client: Object exposing ``process()`` / ``detect_emotion()`` (the
            local model), or None to disable emotion.
    """
    from google import genai

    from reachy_emotion.gemini_bridge import DEFAULT_SYSTEM_PROMPT

    prompt = system_prompt or _load_system_prompt() or DEFAULT_SYSTEM_PROMPT
    model = model or _load_live_model()
    client = genai.Client(api_key=_load_api_key())
    config = _build_config(prompt)

    try:
        src_rate = mini.media.get_input_audio_samplerate()
    except Exception:
        src_rate = _LIVE_INPUT_RATE
    if not src_rate or src_rate <= 0:
        src_rate = _LIVE_INPUT_RATE

    ring = _AudioRing()
    mini.media.start_recording()
    mini.media.start_playing()
    logger.info("Live conversation started (model=%s) — speak to Reachy", model)

    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            tasks = [
                asyncio.create_task(_audio_pump(session, mini, stop_event, src_rate, ring)),
                asyncio.create_task(_emotion_video_loop(session, mini, stop_event, emotion_client, ring)),
                asyncio.create_task(_receive_loop(session, mini, stop_event, emotion_client)),
            ]
            while not stop_event.is_set() and not any(t.done() for t in tasks):
                await asyncio.sleep(0.1)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as exc:
        logger.error("Live conversation failed: %s", exc)
    finally:
        for fn in (mini.media.stop_recording, mini.media.stop_playing):
            try:
                fn()
            except Exception:
                pass
        if emotion_client is not None:
            emotion_client.stop()
