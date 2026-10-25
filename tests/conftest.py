"""Deployment runs unit tests in the checkout: never touch its live bot state."""

import pytest

from stockfish_bot import api


@pytest.fixture(autouse=True)
def isolated_runtime_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DEFAULT_STATE_PATH", tmp_path / "default-state.json")
    monkeypatch.setattr(api, "DEFAULT_ENV_PATH", tmp_path / "default.env")
    api.configure_runtime_paths()
    yield
    api._CHECKPOINT_STORE = None
