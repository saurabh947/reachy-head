"""Tests for SceneBrain (Gemini Robotics ER 2 scene tools) — no network, no robot."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from reachy_emotion import scene_brain as sb

_FRAME = np.zeros((48, 64, 3), np.uint8)


# --------------------------------------------------------------------------- #
# Parsing                                                                       #
# --------------------------------------------------------------------------- #

def test_parse_json_strips_code_fences():
    text = '```json\n[{"point": [500, 250], "label": "red circle"}]\n```'
    assert sb._parse_json(text) == [{"point": [500, 250], "label": "red circle"}]


def test_parse_json_plain_and_invalid():
    assert sb._parse_json("[1, 2]") == [1, 2]
    assert sb._parse_json("not json") is None
    assert sb._parse_json("") is None
    assert sb._parse_json(None) is None


def test_valid_objects_filters_malformed_before_capping():
    parsed = [
        {"point": [1, 2, 3], "label": "bad point"},
        {"point": [100, 100], "label": ""},
        {"label": "no point"},
        "not a dict",
        *[{"point": [10, 10 + i], "label": f"o{i}"} for i in range(20)],
    ]
    out = sb._valid_objects(parsed)
    # malformed entries don't consume the cap: we still get a full set of valid ones
    assert len(out) == sb._MAX_OBJECTS
    assert out[0] == {"label": "o0", "point": [10.0, 10.0]}
    assert sb._valid_objects({"not": "a list"}) == []


# --------------------------------------------------------------------------- #
# Geometry                                                                      #
# --------------------------------------------------------------------------- #

def test_points_to_pixels_maps_and_clamps_strictly_inside():
    px = sb.points_to_pixels([[500, 250], [0, 0], [1000, 1000]], width=640, height=480)
    assert px[0] == (160, 240)
    assert px[1] == (1, 1)        # look_at_image asserts 0 < u < width
    assert px[2] == (639, 479)


def test_horizontal_position():
    assert sb.horizontal_position([500, 100]) == "left"
    assert sb.horizontal_position([500, 500]) == "center"
    assert sb.horizontal_position([500, 900]) == "right"


def test_camera_size_prefers_camera_resolution_then_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    mini = MagicMock()
    mini.media.camera.resolution = (1280, 720)
    assert sb.camera_size(mini, frame) == (1280, 720)
    no_cam = MagicMock()
    no_cam.media.camera = None
    assert sb.camera_size(no_cam, frame) == (640, 480)


# --------------------------------------------------------------------------- #
# Gaze                                                                          #
# --------------------------------------------------------------------------- #

def _gaze_mini(home):
    mini = MagicMock()
    mini.get_current_head_pose.return_value = home
    targets = [np.full((4, 4), 1.0), np.full((4, 4), 2.0)]
    mini.look_at_image.side_effect = targets
    return mini, targets


def test_look_at_objects_computes_all_targets_before_moving(monkeypatch):
    # look_at_image rays come from the head's *current* pose, so every target must be
    # computed (perform_movement=False) before the head moves — else they stack.
    monkeypatch.setattr(sb.time, "sleep", lambda s: None)
    home = np.eye(4)
    mini, targets = _gaze_mini(home)
    sb.look_at_objects(mini, [(100, 200), (300, 400)], capture_pose=home)

    names = [c[0] for c in mini.mock_calls if c[0] in ("look_at_image", "goto_target")]
    assert names == ["look_at_image", "look_at_image", "goto_target", "goto_target", "goto_target"]
    for c in mini.look_at_image.call_args_list:
        assert c.kwargs["perform_movement"] is False
    heads = [c.kwargs["head"] for c in mini.goto_target.call_args_list]
    assert heads[0] is targets[0] and heads[1] is targets[1] and heads[2] is home
    assert all(c.kwargs["body_yaw"] is None for c in mini.goto_target.call_args_list)


def test_look_at_objects_returns_to_capture_pose_first(monkeypatch):
    monkeypatch.setattr(sb.time, "sleep", lambda s: None)
    home, capture = np.eye(4), np.eye(4) * 3.0  # head moved since the frame was taken
    mini, _ = _gaze_mini(home)
    sb.look_at_objects(mini, [(100, 200)], capture_pose=capture)

    first = [c for c in mini.mock_calls if c[0] in ("look_at_image", "goto_target")][0]
    assert first[0] == "goto_target" and first.kwargs["head"] is capture


def test_look_at_objects_stops_on_failure_but_still_returns_home(monkeypatch):
    monkeypatch.setattr(sb.time, "sleep", lambda s: None)
    home = np.eye(4)
    mini = MagicMock()
    mini.get_current_head_pose.return_value = home
    mini.look_at_image.side_effect = RuntimeError("Camera specs not set.")
    sb.look_at_objects(mini, [(100, 200), (300, 400)])
    assert mini.look_at_image.call_count == 1
    mini.goto_target.assert_called_once()  # just the return home
    assert mini.goto_target.call_args.kwargs["head"] is home


# --------------------------------------------------------------------------- #
# SceneBrain (fake ER client)                                                   #
# --------------------------------------------------------------------------- #

class _FakeModels:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append((model, contents))
        self.config = config
        return MagicMock(text=self.text)


def _brain(text):
    client = MagicMock()
    client.models = _FakeModels(text)
    return sb.SceneBrain(client, model="er-test"), client.models


def test_locate_parses_er_points_and_passes_query():
    brain, models = _brain('```json\n[{"point": [500, 250], "label": "cup"}]\n```')
    assert brain.locate(_FRAME, "the cup") == [{"label": "cup", "point": [500.0, 250.0]}]
    model, contents = models.calls[0]
    assert model == "er-test"
    assert "the cup" in contents[1]
    # a stalled ER request must not hang forever (google-genai's default is no timeout)
    assert models.config.http_options.timeout == sb._ER_TIMEOUT_MS


def test_locate_defaults_to_all_objects():
    brain, models = _brain("[]")
    assert brain.locate(_FRAME) == []
    assert "distinct objects" in models.calls[0][1][1]


def test_plan_returns_steps_and_keeps_last_plan():
    brain, _ = _brain('[{"step": 1, "action": "grasp", "object": "cup", "target": null}]')
    steps = brain.plan(_FRAME, "clear the table")
    assert steps == [{"step": 1, "action": "grasp", "object": "cup", "target": None}]
    assert brain.last_plan == {"goal": "clear the table", "steps": steps}


def test_plan_non_list_gives_no_steps():
    brain, _ = _brain('{"oops": true}')
    assert brain.plan(_FRAME, "x") == []
