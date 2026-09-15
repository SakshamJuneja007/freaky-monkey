"""Minimal LangGraph spike; never imported by production DEIMOS execution.

The graph owns only orchestration state. A DEIMOS-style verification flag is
kept outside the graph and is supplied by the caller, demonstrating that the
framework cannot declare an external action successful on its own.
"""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
except ImportError:  # pragma: no cover - exercised by environment guard
    SqliteSaver = None

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore
    from langgraph.types import Command, interrupt  # type: ignore
except ImportError:  # pragma: no cover
    StateGraph = None
    START = END = Command = interrupt = None


class SpikeState(TypedDict, total=False):
    task_id: str
    goal: str
    step: str
    approved: bool
    completed: bool
    deimos_verified: bool


def build_graph(checkpointer):
    if StateGraph is None or interrupt is None:
        raise RuntimeError("LangGraph is not installed; install agent_control/langgraph_requirements.txt")

    graph = StateGraph(SpikeState)

    def execute_step(state: SpikeState):
        return {"step": "EXECUTED", "completed": False}

    def approval(state: SpikeState):
        answer = interrupt({"task_id": state["task_id"], "question": "approve spike step"})
        return {"approved": bool(answer), "step": "APPROVED" if answer else "REJECTED"}

    def complete(state: SpikeState):
        # This node is orchestration only. It does not assert external verification.
        return {"completed": True, "step": "ORCHESTRATED"}

    graph.add_node("execute_step", execute_step)
    graph.add_node("approval", approval)
    graph.add_node("complete", complete)
    graph.add_edge(START, "execute_step")
    graph.add_edge("execute_step", "approval")
    graph.add_edge("approval", "complete")
    graph.add_edge("complete", END)
    return graph.compile(checkpointer=checkpointer)


def run_spike(db_path: str | Path) -> dict[str, object]:
    """Run create -> step -> interrupt -> resume -> complete using SQLite."""
    if SqliteSaver is None or StateGraph is None:
        raise RuntimeError("LangGraph + langgraph-checkpoint-sqlite are required for the spike")
    db_path = str(db_path)
    with SqliteSaver.from_conn_string(db_path) as saver:
        saver.setup()
        graph = build_graph(saver)
        config = {"configurable": {"thread_id": "phase2-spike-1"}}
        first = graph.invoke({"task_id": "spike-task-1", "goal": "prove pause/resume"}, config=config)
        interrupted = bool(first.get("__interrupt__"))
        resumed = graph.invoke(Command(resume=True), config=config)
        return {
            "interrupted": interrupted,
            "resumed": resumed.get("completed") is True,
            "deimos_verified": resumed.get("deimos_verified", False) is True,
        }
