"""Emotion label -> RecordedMove resolution (W4-A1).

The live app matched a model emotion label against the emotions-library move
names with a naive substring test (``label in move_name``) at
``conversation_app._react_to_emotion``. That silently resolved only 3 of the 8
model labels, because the library uses expressive names with numeric suffixes
(``furious1``, ``cheerful1``, ``fear1`` ...) rather than the label words
(``angry``, ``happy``, ``fearful``). This module maps each of the 8 model labels
(``EMOTION_ORDER``) to curated candidate move *stems* and resolves them
suffix-tolerantly against whatever the installed library actually offers.

Stems were verified against ``pollen-robotics/reachy-mini-emotions-library``
(81 moves) on 2026-09-18; every label below resolves to a real move.
"""

from __future__ import annotations

from typing import Iterable, Optional

# Model labels (EMOTION_ORDER): angry, disgusted, fearful, happy, neutral, sad,
# surprised, unclear. Values are candidate stems in preference order; matching
# is prefix-based, so 'furious' resolves 'furious1'.
EMOTION_MOVE_CANDIDATES: dict[str, list[str]] = {
    "happy": ["cheerful", "enthusiastic", "laughing", "proud", "success", "grateful", "loving"],
    "sad": ["downcast", "sad", "exhausted", "tired", "lonely", "resigned", "boredom"],
    "angry": ["furious", "rage", "irritated", "frustrated", "displeased", "reprimand"],
    "fearful": ["fear", "scared", "anxiety"],
    "disgusted": ["disgusted", "contempt", "uncomfortable"],
    "surprised": ["surprised", "amazed"],
    "neutral": ["serenity", "calming", "indifferent", "attentive", "thoughtful"],
    # 'unclear' = no confident read; callers may choose to hold instead of playing.
    "unclear": ["confused", "uncertain", "lost"],
}


def resolve_move(emotion: Optional[str], available_moves: Iterable[str]) -> Optional[str]:
    """Return a move name from *available_moves* for *emotion*, or ``None``.

    For each candidate stem (in preference order), return the first available
    move whose name equals the stem or starts with it (case-insensitive; sorted
    for determinism). Unknown/empty labels return ``None`` so the caller can hold
    rather than play an arbitrary move (the old handler fell back to
    ``available[0]``, which played an unrelated move for unmatched labels).
    """
    label = (emotion or "").strip().lower()
    candidates = EMOTION_MOVE_CANDIDATES.get(label)
    if not candidates:
        return None
    avail = sorted(available_moves)
    lower = {m.lower(): m for m in avail}
    for stem in candidates:
        if stem in lower:
            return lower[stem]
        for move in avail:
            if move.lower().startswith(stem):
                return move
    return None


def emotion_move_map(available_moves: Iterable[str]) -> dict[str, Optional[str]]:
    """Resolve every known label against *available_moves* (for the motion layer)."""
    avail = list(available_moves)
    return {label: resolve_move(label, avail) for label in EMOTION_MOVE_CANDIDATES}
