from __future__ import annotations

import importlib.util

import pytest

HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None
HAS_SQLITE_CHECKPOINTER = HAS_LANGGRAPH and importlib.util.find_spec("langgraph.checkpoint.sqlite") is not None

pytestmark = pytest.mark.skipif(
    not (HAS_LANGGRAPH and HAS_SQLITE_CHECKPOINTER),
    reason="LangGraph spike dependencies are intentionally isolated; install phase2_spike/requirements.txt",
)


def _run(tmp_path):
    from phase2_spike.langgraph_spike import run_spike
    return run_spike(tmp_path / "spike.db")


def test_19_framework_state_can_be_created(tmp_path):
    # Creation is implicit in the first graph invocation.
    result = _run(tmp_path)
    assert "interrupted" in result


def test_20_framework_workflow_can_pause(tmp_path):
    assert _run(tmp_path)["interrupted"] is True


def test_21_framework_workflow_can_resume(tmp_path):
    assert _run(tmp_path)["resumed"] is True


def test_22_framework_checkpoint_persists_state(tmp_path):
    result = _run(tmp_path)
    assert result["interrupted"] and result["resumed"]


def test_23_framework_failure_is_contained_to_spike(tmp_path):
    # The spike imports only inside its own module; production Session has no
    # LangGraph import and no framework-owned runtime database.
    import agent_control.session as session_module
    assert "langgraph" not in session_module.__dict__
    assert _run(tmp_path)["resumed"] is True


def test_24_deimos_verification_remains_authoritative(tmp_path):
    # The spike intentionally reports this as false until a DEIMOS verifier
    # supplies evidence. Graph completion is not verification success.
    assert _run(tmp_path)["deimos_verified"] is False
