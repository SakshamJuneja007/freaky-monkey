"""Structured JSONL tracing (plan S24).

One file per run under ``benchmark/results/traces/``. Every observation records
time/source/result so stale-state failures can be measured after the fact
(plan S5), and every retry is bounded *and logged* (plan S30).

Tracing must never break a run: all writes are best-effort.

``on_event`` is an optional callback on the *existing* path -- one function, no
event bus, no subscriber registry. It exists so an interface can show real
progress while a task runs. The alternative was to have the chat layer invent
progress lines, which is precisely the fake narration this project refuses; the
callback means every status line a user sees corresponds to an event that was
actually emitted. Like the file writes, it is best-effort: a callback that raises
is dropped, because a display must not be able to fail a run.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .types import (
    Action,
    ActionResult,
    FailureClass,
    Observation,
    VerificationResult,
)

DEFAULT_TRACE_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "results" / "traces"


@dataclass
class Trace:
    """Append-only event log plus the counters the metrics table needs."""

    task_id: str
    condition: str
    trial: int = 0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    trace_dir: Path = DEFAULT_TRACE_DIR
    enabled: bool = True
    #: Called with each event record as it is emitted. Read-only by contract: the
    #: dict handed over is the one being written, so a callback that mutates it
    #: corrupts the trace.
    on_event: Callable[[dict[str, Any]], None] | None = None

    started_at: float = field(default_factory=time.time)
    seq: int = 0
    counters: Counter = field(default_factory=Counter)
    failure_categories: Counter = field(default_factory=Counter)
    _fh: Any = None

    # -- lifecycle ---------------------------------------------------------
    def __post_init__(self) -> None:
        if not self.enabled:
            return
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            name = f"{self.task_id}__{self.condition}__t{self.trial}__{self.run_id}.jsonl"
            self._fh = open(self.trace_dir / name, "a", encoding="utf-8")
        except OSError:
            self._fh = None

    @property
    def path(self) -> Path | None:
        return Path(self._fh.name) if self._fh else None

    def close(self) -> None:
        if self._fh:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- core emit ---------------------------------------------------------
    def emit(self, event: str, **fields: Any) -> None:
        self.seq += 1
        record = {
            "seq": self.seq,
            "ts": time.time(),
            "elapsed_s": round(time.time() - self.started_at, 4),
            "run_id": self.run_id,
            "task_id": self.task_id,
            "condition": self.condition,
            "trial": self.trial,
            "event": event,
            **fields,
        }
        # Before the file write, and outside its early return: a run whose trace
        # file could not be opened should still be watchable.
        if self.on_event is not None:
            try:
                self.on_event(record)
            except Exception:  # noqa: BLE001 - a display cannot fail a run
                self.on_event = None
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()
        except (OSError, ValueError):
            self._fh = None

    # -- typed helpers -----------------------------------------------------
    def observation(self, obs: Observation, *, purpose: str = "") -> Observation:
        """Log an observation and return it unchanged, so call sites can wrap."""
        self.counters["observations"] += 1
        # Observation values are acquired by the live observation functions;
        # the cache is never used as verification evidence. Count these as fresh
        # observations at the tracing boundary.
        self.counters["fresh_observations"] += 1
        self.counters[f"observations.{obs.source.value}"] += 1
        if obs.source.value == "vision":
            # Screen captures are counted separately from model invocations.
            # Plan S19's "vision calls" column is about how often a VLM was asked
            # to look at pixels, which is ``planner_call(vision=True)`` -- counting
            # the capture here as well would double every vision turn.
            self.counters["screenshots"] += 1
        self.emit("observation", purpose=purpose, observation=obs.to_json())
        return obs

    def text_observation(self, observation: Any, *, purpose: str = "") -> Any:
        """Record a UniversalTextReader observation with the same counters."""
        self.counters["observations"] += 1
        self.counters["observations.text"] += 1
        if bool(getattr(observation, "fresh", False)):
            self.counters["fresh_observations"] += 1
        self.emit("text_observation", purpose=purpose, observation=observation.to_json())
        return observation

    def policy(self, action: Action, decision: str, reason: str) -> None:
        self.counters[f"policy.{decision.lower()}"] += 1
        self.emit(
            "policy_decision",
            action=action.to_json(),
            decision=decision,
            reason=reason,
        )

    def action(self, result: ActionResult, *, precondition_age_s: float | None = None) -> None:
        self.counters["actions"] += 1
        self.counters[f"actions.{result.action.kind}"] += 1
        if not result.ok:
            self.counters["actions.failed"] += 1
        self.emit(
            "action",
            precondition_age_s=precondition_age_s,
            result=result.to_json(),
        )

    def verification(self, vr: VerificationResult, *, checkpoint: bool) -> None:
        self.counters["verifications"] += 1
        self.counters[f"verifications.{vr.verdict.value}"] += 1
        self.emit("verification", checkpoint=checkpoint, verification=vr.to_json())

    def recovery(self, *, failure_class: FailureClass, decision: str, attempt: int,
                 budget_left: dict[str, int]) -> None:
        """One recovery *attempt* (plan S9).

        The outcome is logged separately by :meth:`recovery_resolved`. Counting
        both here would make plan S19's recovery success rate
        ``successes / (attempts + successes)`` -- 50% for a failure that was
        recovered on the first try.
        """
        self.counters["recovery_attempts"] += 1
        self.failure_categories[failure_class.value] += 1
        self.emit(
            "recovery",
            failure_class=failure_class.value,
            decision=decision,
            attempt=attempt,
            budget_left=budget_left,
        )

    def recovery_resolved(self, *, failure_class: FailureClass, attempt: int,
                          budget_left: dict[str, int]) -> None:
        """The action that had failed went through. Numerator of the success rate."""
        self.counters["recovery_successes"] += 1
        self.emit(
            "recovery_resolved",
            failure_class=failure_class.value,
            attempt=attempt,
            budget_left=budget_left,
        )

    def failure(self, failure_class: FailureClass, **fields: Any) -> None:
        """A failure that ended a trial without any recovery attempt.

        The structured path populates ``failure_categories`` through
        :meth:`recovery`, which every classified failure passes through. The
        vision-only baseline has no recovery channel at all (plan S18), so
        without this its failure-category column would read empty -- an artefact
        of the missing channel rather than a property of the condition.
        """
        self.failure_categories[failure_class.value] += 1
        self.emit("failure", failure_class=failure_class.value, **fields)

    def planner_call(self, *, planner: str, kind: str, prompt_tokens: int = 0,
                     completion_tokens: int = 0, vision: bool = False,
                     latency_s: float = 0.0, error: str | None = None) -> None:
        self.counters["planner_calls"] += 1
        # ``planner_call`` is also used by MockPlanner for deterministic tests.
        # Keep planner invocations and actual model calls distinct so P2.7
        # measurements do not call deterministic planning an LLM request.
        if not str(planner).casefold().startswith(("mock", "fast-router")):
            self.counters["llm_calls"] += 1
        if vision:
            self.counters["vision_calls"] += 1
        self.counters["prompt_tokens"] += prompt_tokens
        self.counters["completion_tokens"] += completion_tokens
        self.emit(
            "planner_call",
            planner=planner,
            kind=kind,
            vision=vision,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=round(latency_s, 4),
            error=error,
        )

    def phase(self, name: str, elapsed_s: float, **fields: Any) -> None:
        """Accumulate lightweight phase timing without a telemetry subsystem."""
        key = f"{name}_ms"
        self.counters[key] += float(elapsed_s) * 1000.0
        self.emit("phase_timing", phase=name, elapsed_ms=round(float(elapsed_s) * 1000.0, 3), **fields)

    def note(self, message: str, **fields: Any) -> None:
        self.emit("note", message=message, **fields)

    def injection(self, kind: str, detail: dict) -> None:
        """State mutated by the external adversarial controller (plan S20)."""
        self.counters["injections"] += 1
        self.emit("injection", kind=kind, detail=detail)

    # -- summary -----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "condition": self.condition,
            "trial": self.trial,
            "wall_clock_s": round(time.time() - self.started_at, 4),
            "counters": dict(self.counters),
            "failure_categories": dict(self.failure_categories),
            "trace_file": os.fspath(self.path) if self.path else None,
        }
