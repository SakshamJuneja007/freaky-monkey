"""Durable LangGraph workflow orchestration for DEIMOS."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint import WorkflowCheckpointStore
from .nodes import (
    WorkflowRuntimeAdapter,
    approval,
    decompose,
    execute,
    execute_ready_batch,
    load_workflow,
    next_step,
    observe,
    policy,
    recover,
    route_after_policy,
    route_after_select,
    route_after_batch,
    route_after_verify,
    select_next_step,
    verify,
    workflow_input,
)
from .state import WorkflowGraphState
from .models import Workflow, WorkflowStatus


def _require_langgraph() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "LangGraph is required for DEIMOS workflow orchestration; "
            "install agent_control/langgraph_requirements.txt"
        ) from exc
    return StateGraph, START, END, Command, None


# Failure is intentionally a node rather than an implicit END edge: it gives
# /tasks and RuntimeManager an explicit workflow result.
def failed(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    workflow = Workflow.from_json(state["workflow"])
    workflow.status = WorkflowStatus.FAILED
    runtime.emit_workflow_event(workflow, "WORKFLOW_FAILED", reason=state.get("decision_reason", "step failed"))
    return {"workflow": workflow.to_json(), "result": {"status": "FAILED", "reason": state.get("decision_reason", "step failed")}}


def build_graph(runtime: WorkflowRuntimeAdapter, checkpointer: Any):
    StateGraph, START, END, _Command, _ = _require_langgraph()
    builder = StateGraph(WorkflowGraphState)
    builder.add_node("load_workflow", lambda s: load_workflow(s, runtime))
    builder.add_node("decompose", lambda s: decompose(s, runtime))
    builder.add_node("select_next_step", lambda s: select_next_step(s, runtime))
    builder.add_node("observe", lambda s: observe(s, runtime))
    builder.add_node("policy", lambda s: policy(s, runtime))
    builder.add_node("approval", lambda s: approval(s, runtime))
    builder.add_node("workflow_input", lambda s: workflow_input(s, runtime))
    builder.add_node("execute", lambda s: execute(s, runtime))
    builder.add_node("execute_batch", lambda s: execute_ready_batch(s, runtime))
    builder.add_node("verify", lambda s: verify(s, runtime))
    builder.add_node("recover", lambda s: recover(s, runtime))
    builder.add_node("next_step", lambda s: next_step(s, runtime))
    builder.add_node("failed", lambda s: failed(s, runtime))

    builder.add_edge(START, "load_workflow")
    builder.add_edge("load_workflow", "decompose")
    builder.add_edge("decompose", "select_next_step")
    builder.add_conditional_edges("select_next_step", route_after_select, {"observe": "observe", "batch": "execute_batch", "done": END})
    builder.add_edge("observe", "policy")
    builder.add_conditional_edges("execute_batch", route_after_batch, {"approval": "approval", "next": "next_step", "wait": "workflow_input", "done": END})
    builder.add_conditional_edges("policy", route_after_policy, {
        "execute": "execute", "approval": "approval", "workflow_input": "workflow_input", "failed": "failed",
    })
    builder.add_edge("approval", "execute")
    builder.add_edge("workflow_input", "observe")
    builder.add_edge("execute", "verify")
    builder.add_conditional_edges("verify", route_after_verify, {
        "next_step": "next_step", "recover": "recover", "failed": "failed",
    })
    builder.add_conditional_edges("next_step", lambda s: "done" if s.get("decision") == "DONE" else "observe", {"done": END, "observe": "observe"})
    builder.add_edge("recover", "failed")
    builder.add_edge("failed", END)
    return builder.compile(checkpointer=checkpointer)


class WorkflowOrchestrator:
    """Own one compiled graph and its durable checkpointer."""

    def __init__(self, runtime: WorkflowRuntimeAdapter, checkpoint_path: str | Path) -> None:
        self.runtime = runtime
        self.checkpoints = WorkflowCheckpointStore(checkpoint_path)
        self.graph = build_graph(runtime, self.checkpoints.saver)

    def start(self, workflow_id: str, goal: str) -> dict[str, Any]:
        config = {"configurable": {"thread_id": workflow_id}}
        return self.graph.invoke({"workflow_id": workflow_id, "goal": goal}, config=config)

    def resume(self, workflow_id: str, value: Any) -> dict[str, Any]:
        from langgraph.types import Command
        config = {"configurable": {"thread_id": workflow_id}}
        return self.graph.invoke(Command(resume=value), config=config)

    def close(self) -> None:
        self.checkpoints.close()
