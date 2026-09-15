"""LangGraph SQLite checkpointer factory."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class WorkflowCheckpointStore:
    """Own one durable SQLite connection for a Session's LangGraph threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._saver: Any | None = None
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:  # pragma: no cover - exercised without optional dependency
            self.close()
            raise RuntimeError(
                "LangGraph is required for durable workflow orchestration; "
                "install agent_control/langgraph_requirements.txt"
            ) from exc
        self._saver = SqliteSaver(self._connection)
        setup = getattr(self._saver, "setup", None)
        if callable(setup):
            setup()

    @property
    def saver(self) -> Any:
        if self._saver is None:
            raise RuntimeError("workflow checkpoint saver is closed")
        return self._saver

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None  # type: ignore[assignment]
            self._saver = None
