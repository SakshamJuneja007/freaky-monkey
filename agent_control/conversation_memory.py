from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class ConversationMemory:
    """Small persistent transcript store for conversational continuity.

    This is deliberately separate from machine-state memory. It stores what the
    user and DEIMOS said, not permissions or filesystem facts. Retrieval is a
    lightweight lexical ranking so ordinary conversation does not require a
    second model call just to find yesterday's turns.
    """

    def __init__(self, path: Path, max_records: int = 5000) -> None:
        self.path = path.expanduser().resolve()
        self.max_records = max(100, int(max_records))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "ConversationMemory":
        configured = os.getenv("DEIMOS_CONVERSATION_MEMORY", "").strip()
        if configured:
            path = Path(configured)
        else:
            root = Path(os.getenv("DEIMOS_MEMORY_ROOT", ".agent_memory"))
            path = root / "conversations.jsonl"
        return cls(path)

    def append_turn(self, turn: Any) -> None:
        task = getattr(turn, "task", None)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": str(getattr(task, "text", "") or ""),
            "assistant": str(getattr(turn, "reply", "") or ""),
            "type": "action" if getattr(turn, "result", None) is not None else "conversation",
            "status": (
                getattr(getattr(turn, "result", None), "status", None).value
                if getattr(getattr(turn, "result", None), "status", None) is not None
                else "conversation"
            ),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._trim()

    def _records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    records.append(item)
        return records

    def _trim(self) -> None:
        records = self._records()
        if len(records) <= self.max_records:
            return
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for item in records[-self.max_records:]:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        records = self._records()
        if not records:
            return []

        q = set(self._tokens(query))
        now = datetime.now(timezone.utc)
        yesterday = (now - timedelta(days=1)).date()
        query_l = (query or "").lower()

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in records:
            user = str(item.get("user", ""))
            assistant = str(item.get("assistant", ""))
            text = f"{user} {assistant}"
            tokens = set(self._tokens(text))
            score = float(len(q & tokens) * 5)

            timestamp = str(item.get("timestamp", ""))
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
                score += max(0.0, 2.0 - age_days * 0.15)
                if "yesterday" in query_l and dt.date() == yesterday:
                    score += 30
            except ValueError:
                pass

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[: max(1, limit)]]

    def recent(self, limit: int = 8) -> list[dict[str, Any]]:
        return self._records()[-max(1, limit):]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(token) > 2
        ]
