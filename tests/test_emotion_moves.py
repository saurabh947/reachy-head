"""Unit tests for emotion->move resolution (W4-A1).

Uses the real 81-move vocabulary from pollen-robotics/reachy-mini-emotions-library
(verified 2026-09-18) as the fixture.
"""

from __future__ import annotations

import pytest

from reachy_emotion.emotion_moves import (
    EMOTION_MOVE_CANDIDATES,
    emotion_move_map,
    resolve_move,
)

# The actual 81 move names in the installed emotions library.
LIBRARY = [
    "amazed1", "anxiety1", "attentive1", "attentive2", "boredom1", "boredom2",
    "calming1", "cheerful1", "come1", "confused1", "contempt1", "curious1",
    "dance1", "dance2", "dance3", "disgusted1", "displeased1", "displeased2",
    "downcast1", "dying1", "electric1", "enthusiastic1", "enthusiastic2",
    "exhausted1", "fear1", "frustrated1", "furious1", "go_away1", "grateful1",
    "helpful1", "helpful2", "impatient1", "impatient2", "incomprehensible2",
    "indifferent1", "inquiring1", "inquiring2", "inquiring3", "irritated1",
    "irritated2", "laughing1", "laughing2", "lonely1", "lost1", "loving1",
    "no1", "no_excited1", "no_sad1", "oops1", "oops2", "proud1", "proud2",
    "proud3", "rage1", "relief1", "relief2", "reprimand1", "reprimand2",
    "reprimand3", "resigned1", "sad1", "sad2", "scared1", "serenity1", "shy1",
    "sleep1", "success1", "success2", "surprised1", "surprised2", "thoughtful1",
    "thoughtful2", "tired1", "uncertain1", "uncomfortable1", "understanding1",
    "understanding2", "welcoming1", "welcoming2", "yes1", "yes_sad1",
]

# The 8 model labels (EMOTION_ORDER).
MODEL_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised", "unclear"]


def test_all_eight_labels_resolve_to_a_real_move():
    mapping = emotion_move_map(LIBRARY)
    for label in MODEL_LABELS:
        assert label in mapping
        assert mapping[label] is not None, f"{label} did not resolve"
        assert mapping[label] in LIBRARY, f"{label} -> {mapping[label]} not in library"


@pytest.mark.parametrize(
    "label,expected",
    [
        ("angry", "furious1"),      # was BROKEN under substring match
        ("happy", "cheerful1"),     # was BROKEN
        ("fearful", "fear1"),       # was BROKEN
        ("neutral", "serenity1"),   # was BROKEN
        ("sad", "downcast1"),       # preference order: downcast before sad
        ("surprised", "surprised1"),
        ("disgusted", "disgusted1"),
        ("unclear", "confused1"),
    ],
)
def test_first_preference_resolution(label, expected):
    assert resolve_move(label, LIBRARY) == expected


def test_previously_broken_labels_now_resolve():
    # Under the old `label in move_name` substring test these produced no move.
    for label in ("angry", "happy", "fearful", "neutral"):
        assert resolve_move(label, LIBRARY) is not None


def test_unknown_and_empty_return_none():
    assert resolve_move("ecstatic", LIBRARY) is None
    assert resolve_move("", LIBRARY) is None
    assert resolve_move(None, LIBRARY) is None


def test_case_insensitive():
    assert resolve_move("ANGRY", LIBRARY) == "furious1"


def test_falls_through_to_later_candidate_when_first_missing():
    # Library lacks any 'cheerful'/'enthusiastic'/'laughing'; happy should fall
    # through to the next available candidate ('proud').
    lib = ["proud1", "sad1", "serenity1"]
    assert resolve_move("happy", lib) == "proud1"


def test_returns_none_when_no_candidate_available():
    assert resolve_move("happy", ["sad1", "serenity1"]) is None


def test_candidates_cover_all_model_labels():
    assert set(EMOTION_MOVE_CANDIDATES) == set(MODEL_LABELS)
