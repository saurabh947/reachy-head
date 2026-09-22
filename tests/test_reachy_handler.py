"""Tests for ReachyMiniActionHandler: connect/disconnect lifecycle, action mapping, throttling."""

import sys
import time
from unittest.mock import MagicMock, patch



# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_mock_recorded_moves(moves=("happy", "sad", "angry", "fearful", "surprised")):
    rm = MagicMock()
    rm.list_moves.return_value = list(moves)
    rm.get.side_effect = lambda name: MagicMock(name=f"move:{name}")
    return rm


def _patched_reachy_modules(fake_mini=None, recorded_moves=None):
    """Return a patch.dict context that stubs out reachy_mini sub-modules."""
    fake_mini = fake_mini or MagicMock()
    rm_module = MagicMock()
    rm_module.RecordedMoves.return_value = recorded_moves or _make_mock_recorded_moves()

    reachy_module = MagicMock()
    reachy_module.ReachyMini.return_value = fake_mini

    return patch.dict(sys.modules, {
        "reachy_mini": reachy_module,
        "reachy_mini.motion": MagicMock(),
        "reachy_mini.motion.recorded_move": rm_module,
    }), reachy_module, rm_module


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_handler_stores_external_mini():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    mini = MagicMock()
    handler = ReachyMiniActionHandler(mini=mini)
    assert handler._external_mini is mini
    assert handler._mini is None  # not yet connected


# ---------------------------------------------------------------------------
# connect — app framework mode (external mini)
# ---------------------------------------------------------------------------

def test_connect_with_external_mini_reuses_instance():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    mini = MagicMock()
    handler = ReachyMiniActionHandler(mini=mini)

    patcher, _, _ = _patched_reachy_modules(fake_mini=mini)
    with patcher:
        result = handler.connect()

    assert result is True
    assert handler.mini is mini
    assert handler._is_connected is True


def test_connect_with_external_mini_does_not_call_enter():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    mini = MagicMock()
    handler = ReachyMiniActionHandler(mini=mini)

    patcher, _, _ = _patched_reachy_modules(fake_mini=mini)
    with patcher:
        handler.connect()

    mini.__enter__.assert_not_called()


# ---------------------------------------------------------------------------
# connect — standalone mode (no external mini)
# ---------------------------------------------------------------------------

def test_connect_standalone_calls_enter():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    fake_mini = MagicMock()
    entered_mini = MagicMock()
    fake_mini.__enter__ = MagicMock(return_value=entered_mini)

    patcher, reachy_module, _ = _patched_reachy_modules(fake_mini=fake_mini)
    with patcher:
        handler = ReachyMiniActionHandler()  # no external mini
        result = handler.connect()

    assert result is True
    fake_mini.__enter__.assert_called_once()
    assert handler.mini is entered_mini


def test_connect_returns_false_on_exception():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    handler = ReachyMiniActionHandler()

    # Make RecordedMoves() raise so connect() hits the except branch.
    # MagicMock(side_effect=...) means *calling* the mock raises; attribute
    # access (i.e. `from module import RecordedMoves`) still succeeds because
    # Python does attribute access, not a call, when doing `from x import y`.
    rm_mock = MagicMock()
    rm_mock.RecordedMoves.side_effect = RuntimeError("library not found")

    with patch.dict(sys.modules, {
        "reachy_mini": MagicMock(),
        "reachy_mini.motion": MagicMock(),
        "reachy_mini.motion.recorded_move": rm_mock,
    }):
        result = handler.connect()

    assert result is False


# ---------------------------------------------------------------------------
# disconnect
# ---------------------------------------------------------------------------

def test_disconnect_with_external_mini_does_not_call_exit():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    mini = MagicMock()
    handler = ReachyMiniActionHandler(mini=mini)
    handler._mini = mini
    handler._is_connected = True

    handler.disconnect()

    mini.__exit__.assert_not_called()
    assert handler._mini is None
    assert handler._is_connected is False


def test_disconnect_standalone_calls_exit():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    handler = ReachyMiniActionHandler()  # standalone: no external mini
    mini = MagicMock()
    handler._mini = mini
    handler._is_connected = True

    handler.disconnect()

    mini.__exit__.assert_called_once_with(None, None, None)
    assert handler._mini is None


def test_disconnect_standalone_handles_exit_exception_gracefully():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    handler = ReachyMiniActionHandler()
    mini = MagicMock()
    mini.__exit__.side_effect = RuntimeError("cleanup error")
    handler._mini = mini
    handler._is_connected = True

    handler.disconnect()  # should not raise

    assert handler._mini is None
    assert handler._is_connected is False


# ---------------------------------------------------------------------------
# get_supported_actions
# ---------------------------------------------------------------------------

def test_get_supported_actions_includes_stub():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    handler = ReachyMiniActionHandler()
    assert "stub" in handler.get_supported_actions()


# ---------------------------------------------------------------------------
# _emotion_to_move
# ---------------------------------------------------------------------------

def test_emotion_to_move_direct_match():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    handler = ReachyMiniActionHandler()
    handler._recorded_moves = _make_mock_recorded_moves(["happy", "sad"])

    assert handler._emotion_to_move("happy") == "happy"
    assert handler._emotion_to_move("sad") == "sad"


def test_emotion_to_move_alias_matching():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    handler = ReachyMiniActionHandler()
    # Only the alias "joy" is present, not "happy" itself
    handler._recorded_moves = _make_mock_recorded_moves(["joy", "sadness"])

    assert handler._emotion_to_move("happy") == "joy"
    assert handler._emotion_to_move("sad") == "sadness"


def test_emotion_to_move_returns_first_on_no_match():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    handler = ReachyMiniActionHandler()
    handler._recorded_moves = _make_mock_recorded_moves(["wave", "dance"])

    result = handler._emotion_to_move("completely_unknown")
    assert result == "wave"  # first available as fallback


def test_emotion_to_move_returns_none_when_empty_library():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    handler = ReachyMiniActionHandler()
    handler._recorded_moves = _make_mock_recorded_moves([])

    assert handler._emotion_to_move("happy") is None


def test_emotion_to_move_case_insensitive():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    handler = ReachyMiniActionHandler()
    handler._recorded_moves = _make_mock_recorded_moves(["happy"])

    assert handler._emotion_to_move("HAPPY") == "happy"
    assert handler._emotion_to_move("Happy") == "happy"


# ---------------------------------------------------------------------------
# execute — throttling
# ---------------------------------------------------------------------------

def test_execute_throttles_rapid_calls():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler
    from emotion_detection_action.core.types import ActionCommand

    mini = MagicMock()
    handler = ReachyMiniActionHandler(mini=mini)
    handler._mini = mini
    handler._is_connected = True
    handler._recorded_moves = _make_mock_recorded_moves()
    handler._action_interval = 9999.0  # very long throttle window

    cmd = ActionCommand(action_type="idle", parameters={})

    with patch.object(handler, "_execute_motion", return_value=True) as mock_exec:
        result1 = handler.execute(cmd)
        result2 = handler.execute(cmd)  # within throttle window

    assert mock_exec.call_count == 1   # only first call executes
    assert result1 is True
    assert result2 is True             # throttled still returns True


def test_execute_allows_call_after_interval():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler
    from emotion_detection_action.core.types import ActionCommand

    mini = MagicMock()
    handler = ReachyMiniActionHandler(mini=mini)
    handler._mini = mini
    handler._is_connected = True
    handler._recorded_moves = _make_mock_recorded_moves()
    handler._action_interval = 0.0  # no throttle

    cmd = ActionCommand(action_type="idle", parameters={})

    with patch.object(handler, "_execute_motion", return_value=True) as mock_exec:
        handler.execute(cmd)
        handler.execute(cmd)

    assert mock_exec.call_count == 2


# ---------------------------------------------------------------------------
# execute — guard conditions
# ---------------------------------------------------------------------------

def test_execute_returns_false_when_not_connected():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler
    from emotion_detection_action.core.types import ActionCommand

    handler = ReachyMiniActionHandler()
    # _is_connected and _mini remain at default (False / None)
    cmd = ActionCommand(action_type="idle", parameters={})

    assert handler.execute(cmd) is False


# ---------------------------------------------------------------------------
# _do_announce — throttling
# ---------------------------------------------------------------------------

def test_do_announce_throttled_for_same_emotion():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    handler = ReachyMiniActionHandler()
    handler._mini = MagicMock()
    handler._announce_interval = 9999.0

    with patch("reachy_emotion.reachy_handler.threading.Thread") as mock_thread:
        handler._do_announce("happy")  # first: should fire
        handler._last_announce_time = time.time()  # stamp as just-announced
        handler._do_announce("happy")  # second within interval: should not fire

    assert mock_thread.call_count == 1


def test_do_announce_fires_again_for_different_emotion():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    handler = ReachyMiniActionHandler()
    handler._mini = MagicMock()
    handler._announce_interval = 9999.0

    with patch("reachy_emotion.reachy_handler.threading.Thread") as mock_thread:
        handler._do_announce("happy")
        handler._last_announce_time = time.time()
        handler._do_announce("sad")  # different emotion → fires regardless of interval

    assert mock_thread.call_count == 2


def test_do_announce_is_thread_safe():
    """Concurrent _do_announce calls for the same emotion must fire exactly once.

    A Barrier is used so all threads enter the critical section simultaneously,
    maximising the chance of a race condition if the lock is missing or broken.
    """
    import threading as _threading
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    N = 10
    handler = ReachyMiniActionHandler()
    handler._mini = MagicMock()
    handler._announce_interval = 9999.0

    fired: list[int] = []
    barrier = _threading.Barrier(N)

    def fake_thread(*args, **kwargs):
        m = MagicMock()
        m.start.side_effect = lambda: fired.append(1)
        return m

    def run():
        barrier.wait()  # all threads enter the lock simultaneously
        handler._do_announce("happy")

    # Create REAL threads before the patch is applied.
    # patch() replaces threading.Thread globally, so _threading.Thread
    # (same module object) would also be the mock inside the `with` block.
    threads = [_threading.Thread(target=run) for _ in range(N)]

    with patch("reachy_emotion.reachy_handler.threading.Thread", side_effect=fake_thread):
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(fired) == 1


# ---------------------------------------------------------------------------
# _emotion_to_move — fallback warning
# ---------------------------------------------------------------------------

def test_emotion_to_move_logs_warning_on_fallback():
    from reachy_emotion.reachy_handler import ReachyMiniActionHandler

    handler = ReachyMiniActionHandler()
    handler._recorded_moves = _make_mock_recorded_moves(["wave", "dance"])

    with patch.object(handler, "_get_available_moves", return_value=["wave"]):
        with patch("reachy_emotion.reachy_handler.logger") as mock_log:
            result = handler._emotion_to_move("completely_unknown")

    assert result == "wave"
    mock_log.warning.assert_called_once()
