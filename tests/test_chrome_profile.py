from __future__ import annotations

from pathlib import Path

from agent_control.policy import Policy
from agent_control.skills.browser.backend import (
    _chrome_profile_name,
    _chrome_user_data_dir,
    _profile_dir,
)
from agent_control import os_tools


def test_chrome_defaults_to_real_user_data_and_default_profile(monkeypatch) -> None:
    monkeypatch.delenv("DEIMOS_CHROME_USER_DATA_DIR", raising=False)
    monkeypatch.delenv("DEIMOS_CHROME_PROFILE", raising=False)
    monkeypatch.delenv("DEIMOS_CHROME_PROFILE_NAME", raising=False)

    user_data = _chrome_user_data_dir()
    profile = _profile_dir()

    assert profile == user_data / "Default"
    assert _chrome_profile_name() == "Default"


def test_chrome_profile_name_can_select_an_existing_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEIMOS_CHROME_USER_DATA_DIR", str(tmp_path / "Chrome" / "User Data"))
    monkeypatch.setenv("DEIMOS_CHROME_PROFILE_NAME", "Profile 2")
    monkeypatch.delenv("DEIMOS_CHROME_PROFILE", raising=False)

    assert _chrome_user_data_dir() == (tmp_path / "Chrome" / "User Data").resolve()
    assert _profile_dir() == (tmp_path / "Chrome" / "User Data" / "Profile 2").resolve()


def test_legacy_complete_profile_override_wins(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "custom-profile"
    monkeypatch.setenv("DEIMOS_CHROME_PROFILE", str(configured))
    monkeypatch.setenv("DEIMOS_CHROME_USER_DATA_DIR", str(tmp_path / "other-root"))
    monkeypatch.setenv("DEIMOS_CHROME_PROFILE_NAME", "Profile 9")

    assert _profile_dir() == configured.resolve()


def test_os_launch_chrome_uses_same_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEIMOS_CHROME_USER_DATA_DIR", str(tmp_path / "Chrome" / "User Data"))
    monkeypatch.setenv("DEIMOS_CHROME_PROFILE_NAME", "Default")
    monkeypatch.delenv("DEIMOS_CHROME_PROFILE", raising=False)

    assert os_tools._chrome_agent_profile(
        Policy(workspace=tmp_path / "workspace", refuse_if_elevated=False)
    ) == _profile_dir()
