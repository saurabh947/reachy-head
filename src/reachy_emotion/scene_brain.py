"""SceneBrain: Gemini Robotics ER 2 as Reachy's embodied-reasoning "brain".

The Live conversation model (``gemini-3.8-live``) handles voice; ER 2 is called
as a tool when the conversation needs spatial reasoning:

  * ``locate`` — find objects in the current camera frame and return
    ``[{"label", "point": [y, x]}]`` with points normalised to 0-1000
    (the format documented for ER models);
  * ``plan`` — break a physical goal into ordered robot steps (the input the
    M3 skill router will consume once the SO-101 arm is connected).

``look_at_objects`` then turns Reachy's head toward each located object via
the SDK's ``look_at_image`` (verified: it needs the Reachy camera intrinsics,
which the daemon provides).

ER 2 streaming (``gemini-robotics-er-2-streaming-preview``) was ruled out as the
voice model: it rejects AUDIO output (1007), so it can't replace Live.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ER_MODEL = "gemini-robotics-er-2-preview"

_MAX_OBJECTS = 8
_MAX_STEPS = 8
_ER_TIMEOUT_MS = 20_000  # plan_task measured ~5 s; bound it so a stalled call can't hang
_PLAN_ACTIONS = ("locate", "grasp", "place", "move")

_LOCATE_PROMPT = (
    "Point to {target}. Answer only as JSON: "
    '[{{"point": [y, x], "label": "<short name>"}}] with points normalized to 0-1000. '
    "At most {n} items."
)

_PLAN_PROMPT = (
    "You control a robot arm that can see this scene. Break the goal \"{goal}\" into "
    "short, ordered steps using only these actions: {actions}. Only use objects you "
    "can actually see. Answer only as JSON: "
    '[{{"step": 1, "action": "<action>", "object": "<object>", "target": "<where, or null>"}}]. '
    "At most {n} steps."
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)                                                    #
# --------------------------------------------------------------------------- #

def _load_er_model() -> str:
    """Return GEMINI_ER_MODEL from env, or the default ER 2 model."""
    from reachy_emotion.conversation_app import _load_env

    _load_env()
    return os.environ.get("GEMINI_ER_MODEL", "").strip() or DEFAULT_ER_MODEL


def _parse_json(text: Optional[str]) -> Any:
    """Parse model JSON, tolerating the ```json fences ER 2 wraps it in."""
    if not text:
        return None
    try:
        return json.loads(_FENCE.sub("", text.strip()))
    except (json.JSONDecodeError, ValueError):
        return None


def _valid_objects(parsed: Any) -> list[dict]:
    """Keep only well-formed ``{"label", "point": [y, x]}`` entries."""
    out: list[dict] = []
    if not isinstance(parsed, list):
        return out
    for item in parsed:
        if not isinstance(item, dict):
            continue
        point = item.get("point")
        label = str(item.get("label", "")).strip()
        if (
            label
            and isinstance(point, (list, tuple))
            and len(point) == 2
            and all(isinstance(c, (int, float)) for c in point)
        ):
            out.append({"label": label, "point": [float(point[0]), float(point[1])]})
            if len(out) == _MAX_OBJECTS:
                break
    return out


def points_to_pixels(points: list, width: int, height: int) -> list[tuple[int, int]]:
    """Normalised ``[y, x]`` (0-1000) -> ``(u, v)`` pixels, clamped strictly inside
    the image (``look_at_image`` asserts ``0 < u < width`` and ``0 < v < height``)."""
    pixels = []
    for y, x in points:
        u = int(round(x / 1000.0 * width))
        v = int(round(y / 1000.0 * height))
        pixels.append((min(max(u, 1), width - 1), min(max(v, 1), height - 1)))
    return pixels


def horizontal_position(point: list) -> str:
    """Coarse left/center/right from a normalised ``[y, x]`` point (for narration)."""
    x = point[1]
    return "left" if x < 1000 / 3 else "right" if x > 2000 / 3 else "center"


def camera_size(mini: Any, frame: np.ndarray) -> tuple[int, int]:
    """``(width, height)`` used by ``look_at_image``'s bounds; fall back to the frame."""
    try:
        w, h = mini.media.camera.resolution
        return int(w), int(h)
    except Exception:
        return int(frame.shape[1]), int(frame.shape[0])


def look_at_objects(
    mini: Any,
    pixels: list[tuple[int, int]],
    capture_pose: Optional[np.ndarray] = None,
    duration: float = 0.8,
    hold_s: float = 0.5,
) -> None:
    """Turn the head toward each pixel in turn, then return to the starting pose.

    ``look_at_image`` builds its ray from the head's *current* pose, but every pixel
    comes from one frame taken at ``capture_pose``. So: go back to the capture pose
    (if the head has moved since), work out every target pose up front with
    ``perform_movement=False``, and only then move — otherwise each look would be
    computed from the previous one and aim at the wrong spot.

    Blocking (each ``goto_target`` waits for its move) — run it off the event loop.
    ``body_yaw=None`` keeps the body's rotation (``look_at_world`` would reset it to 0).
    """
    try:
        home = mini.get_current_head_pose()
    except Exception:
        home = None

    if capture_pose is not None and (home is None or not np.allclose(home, capture_pose, atol=1e-3)):
        try:
            mini.goto_target(head=capture_pose, duration=duration, body_yaw=None)
        except Exception as exc:
            logger.warning("could not return to the frame's capture pose: %s", exc)
            return

    targets = []
    for u, v in pixels:
        try:
            targets.append(mini.look_at_image(u, v, perform_movement=False))
        except Exception as exc:
            logger.warning("look_at_image(%s, %s) failed: %s", u, v, exc)
            break

    for pose in targets:
        try:
            mini.goto_target(head=pose, duration=duration, body_yaw=None)
        except Exception as exc:
            logger.warning("gaze move failed: %s", exc)
            break
        time.sleep(hold_s)

    if home is not None:
        try:
            mini.goto_target(head=home, duration=duration, body_yaw=None)
        except Exception as exc:
            logger.debug("returning head to start pose failed: %s", exc)


# --------------------------------------------------------------------------- #
# ER 2 client                                                                   #
# --------------------------------------------------------------------------- #

class SceneBrain:
    """Thin wrapper over Gemini Robotics ER 2 for pointing and task planning.

    Args:
        client: A ``google.genai.Client`` (shared with the Live session).
        model: ER model id (else GEMINI_ER_MODEL env / DEFAULT_ER_MODEL).
    """

    def __init__(self, client: Any, model: Optional[str] = None) -> None:
        self._client = client
        self.model = model or _load_er_model()
        self.last_plan: Optional[dict] = None  # kept for the M3 skill router

    def _ask(self, frame_bgr: np.ndarray, prompt: str) -> Any:
        import cv2
        from google.genai import types

        ok, jpg = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise RuntimeError("could not encode camera frame")
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), prompt],
            # Without an explicit timeout google-genai waits forever on a stalled request.
            config=types.GenerateContentConfig(http_options=types.HttpOptions(timeout=_ER_TIMEOUT_MS)),
        )
        return _parse_json(resp.text)

    def locate(self, frame_bgr: np.ndarray, query: Optional[str] = None) -> list[dict]:
        """Find objects (all distinct ones, or those matching *query*)."""
        target = query.strip() if query and query.strip() else "the distinct objects in view"
        parsed = self._ask(frame_bgr, _LOCATE_PROMPT.format(target=target, n=_MAX_OBJECTS))
        return _valid_objects(parsed)

    def plan(self, frame_bgr: np.ndarray, goal: str) -> list[dict]:
        """Break *goal* into ordered steps for the (future) SO-101 arm."""
        parsed = self._ask(
            frame_bgr,
            _PLAN_PROMPT.format(goal=goal, actions=", ".join(_PLAN_ACTIONS), n=_MAX_STEPS),
        )
        steps = [s for s in parsed[:_MAX_STEPS] if isinstance(s, dict)] if isinstance(parsed, list) else []
        self.last_plan = {"goal": goal, "steps": steps}
        return steps
