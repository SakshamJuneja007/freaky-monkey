from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_control import api, os_tools, verifiers
from agent_control.policy import Policy
from agent_control.types import Action, PolicyDenied, Verdict
from agent_control.planner.base import ALLOWED_ACTION_KINDS
from agent_control.planner.openai_compat import ACTION_SCHEMA, SYSTEM_PROMPT


def test_planner_has_distinct_app_file_url_command_actions():
    assert {"launch_app", "open_file", "open_url", "run_command"} <= set(ALLOWED_ACTION_KINDS)
    assert 'open_url' in ACTION_SCHEMA
    assert 'open_url' in SYSTEM_PROMPT
    assert 'launch_app accepts only an app name' not in SYSTEM_PROMPT


def test_launch_app_policy_blocks_shells(policy: Policy):
    for name in ("cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "bash"):
        decision, reason = policy.check(Action(kind="launch_app", params={"app": name}))
        assert decision.value == "DENY"
        assert "blocked" in reason.lower()


def test_launch_app_rejects_file_or_url_target(policy: Policy):
    action = Action(kind="launch_app", params={"app": "notepad", "open_path": "x.txt"})
    decision, reason = policy.check(action)
    assert decision.value == "DENY"
    assert "open_file" in reason


def test_open_file_can_target_directory(policy: Policy, tmp_path: Path, monkeypatch):
    folder = tmp_path / "folder"
    folder.mkdir()
    policy.readable_roots = (tmp_path,)
    monkeypatch.setattr(os_tools.sys, "platform", "linux")
    monkeypatch.setattr(os_tools.subprocess, "Popen", lambda *a, **k: object())
    result = os_tools.open_file(policy, Action(kind="open_file", params={"path": str(folder), "settle_s": 0}))
    assert result.ok is True


def test_read_policy_allows_downloads_and_d_drive_but_not_writes(tmp_path: Path):
    downloads = tmp_path / "Downloads"
    ddrive = tmp_path / "D"
    downloads.mkdir()
    ddrive.mkdir()
    policy = Policy(workspace=tmp_path / "ws", readable_roots=(downloads, ddrive), refuse_if_elevated=False)
    assert policy.resolve_read_path(downloads / "a.txt")
    assert policy.resolve_read_path(ddrive / "b.txt")
    with pytest.raises(PolicyDenied):
        policy.resolve_write_path(downloads / "a.txt")
    with pytest.raises(PolicyDenied):
        policy.resolve_write_path(ddrive / "b.txt")


def test_dynamic_app_resolver_accepts_a_real_executable(monkeypatch, tmp_path: Path):
    exe = tmp_path / "ExampleApp.exe"
    exe.write_bytes(b"MZ")
    monkeypatch.setattr(os_tools, "sys", type("S", (), {"platform": "win32"})())
    monkeypatch.setattr(os_tools, "_common_windows_candidates", lambda app: [exe])
    monkeypatch.setattr(os_tools, "_registry_app_executable", lambda app: None)
    assert os_tools.resolve_app_executable("Example App") == exe


def test_dynamic_app_resolver_never_returns_blocked_executable(monkeypatch, tmp_path: Path):
    exe = tmp_path / "powershell.exe"
    exe.write_bytes(b"MZ")
    monkeypatch.setattr(os_tools, "sys", type("S", (), {"platform": "win32"})())
    monkeypatch.setattr(os_tools, "_common_windows_candidates", lambda app: [exe])
    assert os_tools.resolve_app_executable("powershell") is None
