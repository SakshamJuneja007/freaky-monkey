"""Shared fixtures.

``refuse_if_elevated=False`` in the ``policy`` fixture on purpose: whether the
developer's shell happens to be elevated is not what any of these tests are
about, and plan S10's refusal has its own test that fakes the answer instead of
depending on how pytest was launched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_control.policy import Policy
from agent_control.trace import Trace


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "ws"


@pytest.fixture
def policy(workspace: Path) -> Policy:
    return Policy(workspace=workspace, refuse_if_elevated=False)


@pytest.fixture
def trace(tmp_path: Path):
    tracer = Trace(task_id="test", condition="test", trace_dir=tmp_path / "traces")
    yield tracer
    tracer.close()


@pytest.fixture
def events(trace):
    """Read the JSONL back as parsed records. Closes the trace first."""
    import json

    def read() -> list[dict]:
        path = trace.path
        trace.close()
        if path is None:
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    return read
