import json
from types import SimpleNamespace

from lora_trainer import app


def test_settings_keep_simple_and_hires_sampling_independent():
    settings = app.AppSettings()
    assert settings.sampling.sampler == "euler"
    settings.simple.steps = 12
    settings.simple.denoise = 0.45
    settings.hires.first_steps = 28
    settings.hires.first_denoise = 1.0
    settings.hires.second_steps = 18
    settings.hires.second_denoise = 0.7

    settings.mode = "hires"
    settings.mode = "simple"
    settings.mode = "hires"

    assert settings.simple.steps == 12
    assert settings.simple.denoise == 0.45
    assert settings.hires.first_steps == 28
    assert settings.hires.first_denoise == 1.0
    assert settings.hires.second_steps == 18
    assert settings.hires.second_denoise == 0.7


def test_settings_round_trip_uses_only_config_object(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(app, "_SETTINGS_PATH", path)
    settings = app.AppSettings()
    settings.mode = "hires"
    settings.models.checkpoint = "/models/base.safetensors"
    settings.simple.steps = 9
    settings.hires.first_steps = 31
    settings.hires.second_denoise = 0.65

    app._save_settings(settings)
    loaded = app._load_settings()

    assert json.loads(path.read_text())["mode"] == "hires"
    assert loaded.mode == "hires"
    assert loaded.models.checkpoint == "/models/base.safetensors"
    assert loaded.simple.steps == 9
    assert loaded.hires.first_steps == 31
    assert loaded.hires.second_denoise == 0.65


def test_reset_defaults_clears_private_widget_keys(monkeypatch, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{}")
    state = {
        "settings": app.AppSettings(mode="hires"),
        "ui.simple.steps": 1,
        "ui.hires.first_steps": 3,
        "_sock_path": "/tmp/app.sock",
    }
    monkeypatch.setattr(app, "_SETTINGS_PATH", path)
    monkeypatch.setattr(app, "st", SimpleNamespace(session_state=state))

    app._reset_defaults()

    assert state["settings"] == app.AppSettings()
    assert "ui.simple.steps" not in state
    assert "ui.hires.first_steps" not in state
    assert state["_sock_path"] == "/tmp/app.sock"
    assert not path.exists()
