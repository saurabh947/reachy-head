"""Tests for GeminiBridge: text extraction, function-call handling, tool execution."""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs for Gemini response objects
# ---------------------------------------------------------------------------

class _Part:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class _Candidate:
    def __init__(self, parts):
        self.content = MagicMock(parts=parts)


class _Response:
    def __init__(self, *parts):
        self.candidates = [_Candidate(list(parts))]


class _EmptyResponse:
    """Simulates a safety-blocked or empty response with no candidates."""
    candidates = []


def _fn_call(name: str) -> MagicMock:
    fc = MagicMock()
    fc.name = name
    return fc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNCLEAR_RESULT = {
    "dominant_emotion": "unclear",
    "confidence": 0.0,
    "note": "No result from emotion-cloud within timeout",
}


def _make_emotion_client(result=None) -> MagicMock:
    """Return a mock emotion client with a pre-configured detect_emotion return value."""
    client = MagicMock()
    client.detect_emotion.return_value = result if result is not None else _UNCLEAR_RESULT
    return client


def _make_bridge(emotion_client=None):
    from reachy_emotion.gemini_bridge import GeminiBridge
    return GeminiBridge(
        api_key="test-key",
        emotion_client=emotion_client or _make_emotion_client(),
        model="gemini-test",
    )


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------

def test_extract_text_single_part():
    bridge = _make_bridge()
    response = _Response(_Part(text="Hello there"))
    assert bridge._extract_text(response) == "Hello there"


def test_extract_text_multiple_parts_concatenated():
    bridge = _make_bridge()
    response = _Response(_Part(text="Hello "), _Part(text="world"))
    result = bridge._extract_text(response)
    assert "Hello" in result and "world" in result


def test_extract_text_ignores_parts_without_text():
    bridge = _make_bridge()
    fc = _fn_call("detect_emotion")
    response = _Response(_Part(function_call=fc), _Part(text="Final reply"))
    assert bridge._extract_text(response) == "Final reply"


def test_extract_text_returns_empty_string_when_no_text():
    bridge = _make_bridge()
    fc = _fn_call("detect_emotion")
    response = _Response(_Part(function_call=fc))
    assert bridge._extract_text(response) == ""


def test_extract_text_returns_empty_for_empty_candidates():
    """Safety-blocked or empty response must not raise."""
    bridge = _make_bridge()
    assert bridge._extract_text(_EmptyResponse()) == ""


def test_extract_text_handles_none_content():
    """Candidate with content=None must not raise AttributeError."""
    bridge = _make_bridge()
    candidate = MagicMock()
    candidate.content = None
    response = MagicMock()
    response.candidates = [candidate]
    assert bridge._extract_text(response) == ""


# ---------------------------------------------------------------------------
# _extract_function_calls
# ---------------------------------------------------------------------------

def test_extract_function_calls_empty_when_no_fn_call():
    bridge = _make_bridge()
    response = _Response(_Part(text="Just text"))
    assert bridge._extract_function_calls(response) == []


def test_extract_function_calls_finds_single_call():
    bridge = _make_bridge()
    fc = _fn_call("detect_emotion")
    response = _Response(_Part(function_call=fc))
    calls = bridge._extract_function_calls(response)
    assert len(calls) == 1
    assert calls[0].name == "detect_emotion"


def test_extract_function_calls_finds_multiple_calls():
    bridge = _make_bridge()
    fc1 = _fn_call("detect_emotion")
    fc2 = _fn_call("detect_emotion")
    response = _Response(_Part(function_call=fc1), _Part(function_call=fc2))
    calls = bridge._extract_function_calls(response)
    assert len(calls) == 2


def test_extract_function_calls_returns_empty_for_empty_candidates():
    bridge = _make_bridge()
    assert bridge._extract_function_calls(_EmptyResponse()) == []


# ---------------------------------------------------------------------------
# _run_emotion_detection
# ---------------------------------------------------------------------------

def test_run_emotion_detection_returns_error_when_no_client():
    from reachy_emotion.gemini_bridge import GeminiBridge
    bridge = GeminiBridge(api_key="key", emotion_client=None, model="test")
    result = bridge._run_emotion_detection()
    assert "error" in result


def test_run_emotion_detection_returns_unclear_when_no_result():
    """detect_emotion() returning unclear is passed through as-is."""
    bridge = _make_bridge(emotion_client=_make_emotion_client(result=None))
    result = bridge._run_emotion_detection()
    assert result["dominant_emotion"] == "unclear"
    assert result["confidence"] == 0.0
    assert "note" in result
    # unclear result must NOT be stored as last_emotion_result
    assert bridge._last_emotion_result is None


def test_run_emotion_detection_returns_cloud_result():
    cloud_result = {
        "dominant_emotion": "happy",
        "confidence": 0.91,
        "confidence_scores": {"happy": 0.91, "neutral": 0.05},
        "stress": 0.1,
        "engagement": 0.88,
        "arousal": 0.65,
    }
    bridge = _make_bridge(emotion_client=_make_emotion_client(result=cloud_result))
    result = bridge._run_emotion_detection()
    assert result["dominant_emotion"] == "happy"
    assert result["confidence"] == pytest.approx(0.91, abs=0.01)
    assert bridge._last_emotion_result is cloud_result


def test_run_emotion_detection_stores_last_result():
    cloud_result = {"dominant_emotion": "sad", "confidence": 0.75}
    bridge = _make_bridge(emotion_client=_make_emotion_client(result=cloud_result))
    bridge._run_emotion_detection()
    assert bridge._last_emotion_result == cloud_result


# ---------------------------------------------------------------------------
# chat — top-level orchestration
# ---------------------------------------------------------------------------

def test_chat_raises_if_not_initialized():
    bridge = _make_bridge()
    with pytest.raises(RuntimeError, match="initialize"):
        bridge.chat("Hello!")


def test_chat_returns_text_when_no_tool_calls():
    bridge = _make_bridge()
    bridge._chat = MagicMock()
    bridge._chat.send_message.return_value = _Response(_Part(text="Nice to meet you!"))

    text, emotion = bridge.chat("Hello Reachy!")

    assert text == "Nice to meet you!"
    assert emotion is None
    bridge._chat.send_message.assert_called_once_with("Hello Reachy!")


def test_chat_resets_last_emotion_result_each_turn():
    bridge = _make_bridge()
    bridge._chat = MagicMock()
    bridge._last_emotion_result = {"dominant_emotion": "sad"}  # leftover from previous turn
    bridge._chat.send_message.return_value = _Response(_Part(text="Hi"))

    _, emotion = bridge.chat("Hello")

    assert emotion is None  # reset at start of turn


def test_chat_handles_detect_emotion_tool_call():
    """Full tool-call loop: Gemini calls detect_emotion, gets cloud result, returns text."""
    from google.genai import types as genai_types

    cloud_result = {
        "dominant_emotion": "happy",
        "confidence": 0.88,
        "confidence_scores": {"happy": 0.88},
        "stress": 0.1,
        "engagement": 0.9,
        "arousal": 0.6,
    }
    bridge = _make_bridge(emotion_client=_make_emotion_client(result=cloud_result))
    bridge._chat = MagicMock()

    fc = _fn_call("detect_emotion")
    tool_response = _Response(_Part(function_call=fc))
    final_response = _Response(_Part(text="You look happy!"))

    bridge._chat.send_message.side_effect = [tool_response, final_response]

    mock_part = MagicMock()
    with patch.object(genai_types.Part, "from_function_response", return_value=mock_part):
        text, emotion = bridge.chat("How am I feeling?")

    assert text == "You look happy!"
    assert emotion == cloud_result
    assert bridge._chat.send_message.call_count == 2


def test_chat_handles_unknown_tool_gracefully():
    """If Gemini calls an unknown tool, the bridge sends an error response and continues."""
    from google.genai import types as genai_types

    bridge = _make_bridge()
    bridge._chat = MagicMock()

    fc = _fn_call("nonexistent_tool")
    tool_response = _Response(_Part(function_call=fc))
    final_response = _Response(_Part(text="Okay."))

    bridge._chat.send_message.side_effect = [tool_response, final_response]

    mock_part = MagicMock()
    with patch.object(genai_types.Part, "from_function_response", return_value=mock_part):
        text, emotion = bridge.chat("Do something weird")

    assert text == "Okay."
    assert emotion is None


def test_chat_respects_max_tool_call_depth():
    """If the model keeps returning tool calls, the loop exits after _MAX_TOOL_CALL_DEPTH."""
    from reachy_emotion.gemini_bridge import _MAX_TOOL_CALL_DEPTH
    from google.genai import types as genai_types

    bridge = _make_bridge(emotion_client=_make_emotion_client(result=None))
    bridge._chat = MagicMock()

    fc = _fn_call("detect_emotion")
    tool_response = _Response(_Part(function_call=fc))
    bridge._chat.send_message.return_value = tool_response

    mock_part = MagicMock()
    with patch.object(genai_types.Part, "from_function_response", return_value=mock_part):
        text, emotion = bridge.chat("Loop forever?")

    # send_message call count: 1 (user msg) + _MAX_TOOL_CALL_DEPTH (tool results)
    assert bridge._chat.send_message.call_count == 1 + _MAX_TOOL_CALL_DEPTH


# ---------------------------------------------------------------------------
# Lifecycle: initialize() and shutdown()
# ---------------------------------------------------------------------------

def test_shutdown_is_idempotent():
    """Calling shutdown() multiple times must not raise."""
    bridge = _make_bridge()
    bridge.shutdown()
    bridge.shutdown()


def test_api_key_cleared_after_initialize():
    """The API key must be erased from the instance after the Gemini client is created."""
    from reachy_emotion.gemini_bridge import GeminiBridge

    bridge = GeminiBridge(
        api_key="super-secret-key",
        emotion_client=_make_emotion_client(),
        model="test",
    )

    mock_client = MagicMock()
    fake_genai_module = MagicMock()
    fake_genai_module.Client.return_value = mock_client
    fake_google = MagicMock()
    fake_google.genai = fake_genai_module

    with patch.dict(sys.modules, {
        "google": fake_google,
        "google.genai": fake_genai_module,
        "google.genai.types": MagicMock(),
    }):
        bridge.initialize()

    assert bridge._GeminiBridge__api_key == ""
