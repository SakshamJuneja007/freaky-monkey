from pathlib import Path

from agent_control import api


def test_general_write_root_is_persistent_and_outside_sandbox(monkeypatch, tmp_path):
    monkeypatch.delenv(api.GENERAL_WRITE_ROOT_ENV, raising=False)

    root = api.general_write_root()

    assert root == (api.projects_root() / "deimos").resolve()
    assert ".sandbox" not in root.parts


def test_general_write_root_honors_explicit_override(monkeypatch, tmp_path):
    target = tmp_path / "user-output"
    monkeypatch.setenv(api.GENERAL_WRITE_ROOT_ENV, str(target))

    assert api.general_write_root() == target.resolve()


def test_general_write_root_does_not_change_benchmark_workspace(monkeypatch):
    monkeypatch.delenv(api.GENERAL_WRITE_ROOT_ENV, raising=False)

    sandbox = api.default_workspace("some-task")
    persistent = api.general_write_root()

    assert ".sandbox" in sandbox.parts
    assert persistent != sandbox
