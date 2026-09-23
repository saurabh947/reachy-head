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


def _dotenv_at(monkeypatch, tmp_path, env_value, exported):
    env_file = tmp_path / ".env"
    env_file.write_text(f"EMOTION_MODEL_PATH={env_value}\n")
    monkeypatch.setattr(conversation_app, "_load_env", lambda: None)
    monkeypatch.setattr("dotenv.find_dotenv", lambda: str(env_file))
    monkeypatch.setenv("EMOTION_MODEL_PATH", exported)


def test_relative_model_path_from_dotenv_resolves_against_dotenv_dir(monkeypatch, tmp_path):
    # Launched from another cwd (e.g. the dashboard), a relative .env path must
    # still point next to the .env, not into the process cwd.
    _dotenv_at(monkeypatch, tmp_path, "models/m.pt", exported="models/m.pt")
    monkeypatch.chdir("/")
    assert conversation_app._load_local_model_path() == str(tmp_path / "models/m.pt")


def test_relative_model_path_exported_in_shell_is_left_alone(monkeypatch, tmp_path):
    _dotenv_at(monkeypatch, tmp_path, "models/m.pt", exported="other/m.pt")
    assert conversation_app._load_local_model_path() == "other/m.pt"


def test_none_when_local_load_fails(monkeypatch):
    _patch_env(monkeypatch, model_path="/tmp/m.pt")
    local = MagicMock()
    local.start.side_effect = RuntimeError("bad checkpoint")
    monkeypatch.setattr(
        "reachy_emotion.local_inferencer.LocalEmotionInferencer", MagicMock(return_value=local)
    )

    assert conversation_app._resolve_emotion_client(MagicMock()) is None
