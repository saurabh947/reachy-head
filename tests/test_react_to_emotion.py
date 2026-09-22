"""Tests for the live emotion->move reaction path (W4-B1).

Verifies conversation_app._react_to_emotion now resolves moves via the curated
resolver (fixing the 5/8-broken substring match) without touching hardware.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reachy_emotion import conversation_app

# A realistic slice of the emotions library (expressive names, numeric suffixes).
_MOVES = [
    "furious1", "cheerful1", "fear1", "serenity1", "sad1", "downcast1",
    "surprised1", "disgusted1", "confused1",
]


def _fake_recorded_moves(moves=_MOVES):
    rm = MagicMock()
    rm.list_moves.return_value = list(moves)
    rm.get.side_effect = lambda name: f"move:{name}"
    return rm


@pytest.mark.parametrize(
    "emotion,expected_move",
    [
        ("happy", "cheerful1"),   # previously BROKEN (no move contains 'happy')
        ("angry", "furious1"),    # previously BROKEN
        ("fearful", "fear1"),     # previously BROKEN
        ("neutral", "serenity1"), # previously BROKEN
        ("sad", "downcast1"),     # preference order
    ],
)
def test_react_plays_resolved_move(monkeypatch, emotion, expected_move):
    rm = _fake_recorded_moves()
    monkeypatch.setattr(conversation_app, "_get_recorded_moves", lambda lib: rm)
    mini = MagicMock()

    conversation_app._react_to_emotion({"dominant_emotion": emotion}, mini)

    rm.get.assert_called_once_with(expected_move)
    mini.play_move.assert_called_once()


def test_react_skips_unclear(monkeypatch):
    rm = _fake_recorded_moves()
    monkeypatch.setattr(conversation_app, "_get_recorded_moves", lambda lib: rm)
    mini = MagicMock()

    conversation_app._react_to_emotion({"dominant_emotion": "unclear"}, mini)

    mini.play_move.assert_not_called()


def test_react_skips_when_no_move_resolves(monkeypatch):
    # Library has none of the 'angry' candidates → resolve_move returns None → hold.
    rm = _fake_recorded_moves(["serenity1", "sad1"])
    monkeypatch.setattr(conversation_app, "_get_recorded_moves", lambda lib: rm)
    mini = MagicMock()

    conversation_app._react_to_emotion({"dominant_emotion": "angry"}, mini)

    mini.play_move.assert_not_called()


def test_react_no_op_when_library_unavailable(monkeypatch):
    monkeypatch.setattr(conversation_app, "_get_recorded_moves", lambda lib: None)
    mini = MagicMock()

    conversation_app._react_to_emotion({"dominant_emotion": "happy"}, mini)

    mini.play_move.assert_not_called()
