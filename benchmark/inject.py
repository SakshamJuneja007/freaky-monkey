"""The adversarial state-change controller (plan S20).

The experiment: the agent observes state A, begins reasoning, *something else*
changes the machine to state B, and then the agent attempts the action it chose
for A. What we want to know is whether the fresh-precondition check catches the
change (hypothesis H2) or whether the agent acts on a belief the world no longer
supports.

Two firing modes, because they answer slightly different questions:

* ``arm()`` -- a real background thread on a timer. A genuine race, closest to the
  plan's wording, and nondeterministic: with a fast planner it may land after the
  action instead of before it. Every firing is timestamped in the trace so which
  happened is recoverable.
* :class:`InjectingPlanner` -- deterministic. Wraps a planner and mutates state
  after the plan is chosen and before the runner acts on it, which is exactly the
  window under test. It needs no hook inside the runner, so the code path being
  measured is unmodified.

Two properties keep this honest:

* The injector resolves its targets through the same :class:`Policy`, so the
  adversary is confined to the same disposable workspace the agent is.
* Every mutation is written to the trace via ``trace.injection(...)``. This is
  what disambiguates the caveat in ``task.state_fingerprint``: a fingerprint that
  moved with no injection logged near it was the world settling, not interference.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

from agent_control.planner.base import PlannerStep
from agent_control.policy import Policy
from agent_control.trace import Trace

#: Mutations the controller knows how to perform. Deliberately few: each one
#: corresponds to a failure the plan actually names.
INJECTION_KINDS = ("delete_path", "truncate_file", "rename_path", "kill_process")


@dataclass(frozen=True)
class Injection:
    """One scheduled mutation.

    ``target`` is workspace-relative for the path kinds and an exact process
    basename for ``kill_process``.
    """

    kind: str
    target: str
    #: Seconds after ``arm()`` before firing. Ignored by :class:`InjectingPlanner`,
    #: which fires at a fixed point in the loop instead of on a clock.
    after_s: float = 0.4
    #: For ``truncate_file``: what to leave behind. Empty bytes is the classic
    #: "the download landed but is useless" case.
    replacement: bytes = b""
    #: For ``kill_process``: PIDs that must never be touched (anything that was
    #: already running before the trial began).
    spare_pids: frozenset[int] = frozenset()

    def to_json(self) -> dict:
        return {"kind": self.kind, "target": self.target, "after_s": self.after_s,
                "replacement_bytes": len(self.replacement),
                "spare_pids": sorted(self.spare_pids)}


def _delete_path(policy: Policy, target: str) -> dict[str, Any]:
    resolved = policy.resolve_write_path(target)
    existed = resolved.exists()
    if resolved.is_dir():
        shutil.rmtree(resolved, ignore_errors=True)
    elif existed:
        resolved.unlink(missing_ok=True)
    return {"path": str(resolved), "existed": existed, "now_exists": resolved.exists()}


def _truncate_file(policy: Policy, target: str, replacement: bytes) -> dict[str, Any]:
    resolved = policy.resolve_write_path(target)
    existed = resolved.exists()
    if existed:
        resolved.write_bytes(replacement)
    return {"path": str(resolved), "existed": existed, "bytes": len(replacement)}


def _rename_path(policy: Policy, target: str) -> dict[str, Any]:
    resolved = policy.resolve_write_path(target)
    moved = resolved.with_name(resolved.name + ".moved-by-injector")
    existed = resolved.exists()
    if existed:
        resolved.replace(moved)
    return {"path": str(resolved), "moved_to": str(moved), "existed": existed}


def _kill_process(target: str, spare_pids: frozenset[int]) -> dict[str, Any]:
    """Terminate processes by exact basename, sparing anything pre-existing.

    Harness authority, not agent capability: ``kill_process`` is not in
    ``os_tools.DISPATCH`` and is DENY in policy. The adversary in plan S20 is
    explicitly something other than the agent.
    """
    killed, skipped = [], []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if (proc.info.get("name") or "").lower() != target.lower():
                continue
            if proc.info["pid"] in spare_pids:
                skipped.append(proc.info["pid"])
                continue
            proc.terminate()
            killed.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"process": target, "killed": killed, "spared": skipped}


def perform(injection: Injection, policy: Policy, trace: Trace | None = None) -> dict[str, Any]:
    """Apply one mutation and log it. Unknown kinds are refused, not guessed."""
    if injection.kind == "delete_path":
        detail = _delete_path(policy, injection.target)
    elif injection.kind == "truncate_file":
        detail = _truncate_file(policy, injection.target, injection.replacement)
    elif injection.kind == "rename_path":
        detail = _rename_path(policy, injection.target)
    elif injection.kind == "kill_process":
        detail = _kill_process(injection.target, injection.spare_pids)
    else:
        detail = {"error": f"unknown injection kind {injection.kind!r}",
                  "known": list(INJECTION_KINDS)}
    record = {"injection": injection.to_json(), "detail": detail,
              "fired_at": time.time()}
    if trace:
        trace.injection(injection.kind, record)
    return record


@dataclass
class Injector:
    """Timer-driven external controller. One thread per injection, fires once."""

    policy: Policy
    trace: Trace | None = None
    injections: list[Injection] = field(default_factory=list)
    fired: list[dict] = field(default_factory=list)
    _timers: list[threading.Timer] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def arm(self) -> None:
        for injection in self.injections:
            timer = threading.Timer(injection.after_s, self._fire, args=(injection,))
            timer.daemon = True
            self._timers.append(timer)
            timer.start()
        if self.trace and self.injections:
            self.trace.note("injector_armed",
                            injections=[i.to_json() for i in self.injections])

    def _fire(self, injection: Injection) -> None:
        try:
            record = perform(injection, self.policy, self.trace)
        except Exception as exc:  # noqa: BLE001 - an adversary that crashes is a datum
            record = {"injection": injection.to_json(),
                      "error": f"{type(exc).__name__}: {exc}"}
            if self.trace:
                self.trace.note("injection_failed", **record)
        with self._lock:
            self.fired.append(record)

    def disarm(self) -> None:
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()

    def __enter__(self) -> "Injector":
        self.arm()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disarm()


@dataclass
class InjectingPlanner:
    """Deterministic adversary: mutates state between plan and act.

    Wraps any planner. The inner planner sees state A, chooses actions for it, and
    then -- before the runner touches anything -- the injection lands. That is the
    exact window plan S20 describes, without a timing race deciding whether the
    trial tested anything.

    ``fire_on_step`` selects which planner turn to interfere with, so a multi-step
    task can be disrupted mid-sequence rather than only at the start.
    """

    inner: Any
    policy: Policy
    injections: list[Injection] = field(default_factory=list)
    fire_on_step: int = 0
    trace: Trace | None = None
    fired: list[dict] = field(default_factory=list)
    name: str = ""
    _step: int = 0

    def __post_init__(self) -> None:
        inner_name = getattr(self.inner, "name", type(self.inner).__name__)
        self.name = self.name or f"{inner_name}+injected"

    @property
    def usage(self):
        return getattr(self.inner, "usage", None)

    def plan(self, goal: str, state: dict, history: list[dict]) -> PlannerStep:
        step = self.inner.plan(goal, state, history)
        if self._step == self.fire_on_step and not step.done:
            for injection in self.injections:
                self.fired.append(perform(injection, self.policy, self.trace))
        self._step += 1
        return step


def attach(planner: Any, policy: Policy, injections: list[Injection], *,
           trace: Trace | None = None, fire_on_step: int = 0) -> Any:
    """Return ``planner`` unchanged when there is nothing to inject.

    Keeps the harness free of ``if adversarial:`` branches around the run call, so
    the control condition and the adversarial condition go through identical code.
    """
    if not injections:
        return planner
    return InjectingPlanner(inner=planner, policy=policy, injections=injections,
                            trace=trace, fire_on_step=fire_on_step)

