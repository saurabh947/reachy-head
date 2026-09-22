"""Tests for the emotion-source resolution in conversation_app.

Verifies the local emotion inferencer is built from EMOTION_MODEL_PATH, with a
graceful None when unset or on load failure — hardware-free (the class and env
loaders are patched).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from reachy_emotion import conversation_app


def _patch_env(monkeypatch, model_path=None, device="cpu"):
    monkeypatch.setattr(conversation_app, "_load_local_model_path", lambda: model_path)
    monkeypatch.setattr(conversation_app, "_load_emotion_device", lambda: device)


def test_builds_local_inferencer_when_model_path_set(monkeypatch):
    _patch_env(monkeypatch, model_path="/tmp/m.pt")
    local = MagicMock()
    local_cls = MagicMock(return_value=local)
    monkeypatch.setattr("reachy_emotion.local_inferencer.LocalEmotionInferencer", local_cls)

    out = conversation_app._resolve_emotion_client(MagicMock())

    assert out is local
    local.start.assert_called_once()
    _, kwargs = local_cls.call_args
    assert kwargs["model_path"] == "/tmp/m.pt"
    assert kwargs["device"] == "cpu"


def test_none_when_model_path_unset(monkeypatch):
    _patch_env(monkeypatch, model_path=None)
    assert conversation_app._resolve_emotion_client(MagicMock()) is None


def test_none_when_local_load_fails(monkeypatch):
    _patch_env(monkeypatch, model_path="/tmp/m.pt")
    local = MagicMock()
    local.start.side_effect = RuntimeError("bad checkpoint")
    monkeypatch.setattr(
        "reachy_emotion.local_inferencer.LocalEmotionInferencer", MagicMock(return_value=local)
    )

    assert conversation_app._resolve_emotion_client(MagicMock()) is None
