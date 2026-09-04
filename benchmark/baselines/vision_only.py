"""The vision-only baseline condition (plan S18, first bullet).

Same model family, same tasks, same verifiers, same metrics -- one difference:
this loop's only input is a screenshot and its only outputs are mouse and
keyboard events. That single difference is what hypothesis H1 is about.

Three asymmetries between this condition and the structured one are deliberate,
and none of them is a bug to be fixed:

* **No structured state.** The actor never sees the process table, the
  filesystem, or a checkpoint verdict. Handing it any of those would leak the
  thing under test.
* **No mid-run checkpoint verification.** The structured condition verifies each
  sub-goal and feeds the verdict into recovery. This condition has no channel for
  that, so it gets none. Final verification is identical, and that is the number
  the two conditions are compared on.
* **No policy path confinement on its actions.** Synthetic keystrokes land
  wherever focus is; an allowlist cannot constrain them. This is stated in
  ``vision_fallback.apply`` and is a real property of the mechanism, not an
  oversight in the harness. It is the reason this condition is the riskier one to
  run and why it is confined to a disposable workspace.

Read the resulting numbers with the grounding caveat from
``agent_control.vision_fallback``: a weak result here may reflect the VLM's pixel
grounding rather than screenshot-driven control in general. The trace records the
model identity so that stays checkable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agent_control.policy import Policy
from agent_control.runner import (
    RunConfig,
    RunOutcome,
    _cancelled,
    _harness_error,
    _planner_name,
)
from agent_control.task import Task
from agent_control.trace import Trace
from agent_control.types import FailureClass
from agent_control.vision_fallback import (
    VisionActor,
    VisionStep,
    apply_vision_action,
    screenshot,
)

#: Failure classes that end the trial rather than earning another screenshot.
_FATAL = (FailureClass.ENVIRONMENT, FailureClass.PERMISSION_DENIED)


@dataclass
class ScriptedVisionActor:
    """A fixed script of vision actions. Plumbing checks only, never a result.

    Exists so the baseline loop, its tracing, and its scoring can be tested
    without an API key and without depending on any model's grounding. Outcomes
    produced with it are marked ``synthetic``.
    """

    script: list[dict[str, Any]] = field(default_factory=list)
    claim_done_at_end: bool = True
    name: str = "vision:scripted"
    usage: Any = None
    _cursor: int = 0

    def __post_init__(self) -> None:
        from agent_control.planner.base import Usage

        self.usage = self.usage or Usage()

    def propose(self, goal: str, shot, history: list[dict]) -> VisionStep:
        self.usage.add(prompt=0, completion=0, vision=True)
        if self._cursor >= len(self.script):
            return VisionStep(action=None, done=self.claim_done_at_end,
                              reasoning="scripted actor exhausted")
        action = self.script[self._cursor]
        self._cursor += 1
        return VisionStep(action=action, done=False, reasoning="scripted step")

    def apply(self, step: VisionStep):
        return apply_vision_action(step)


def _shot_history(step: VisionStep, result) -> dict:
    """What the actor is told about its own last event. Pixels only, plus outcome."""
    return {
        "action": step.action,
        "ok": bool(result and result.ok),
        "error": result.error if result else None,
    }




def _drive_vision(task: Task, actor: VisionActor, config: RunConfig,
                  trace: Trace, progress: dict[str, int]) -> tuple[bool, int, str | None]:
    """Screenshot -> model -> input -> screenshot. Returns (claimed, steps, abort).

    ``progress`` is written on every turn so a caller whose ``KeyboardInterrupt``
    unwinds this function -- and so never receives its return value -- still knows
    how many turns had begun. The structured runner keeps the same count on its
    ``_Loop``, for the same reason.
    """
    history: list[dict] = []

    for index in range(config.max_steps):
        progress["steps"] = index + 1
        shot = trace.observation(screenshot(), purpose="vision_only step input")
        if not shot.ok:
            trace.failure(FailureClass.ENVIRONMENT, detail=f"screenshot failed: {shot.error}")
            return False, index + 1, f"screenshot failed: {shot.error}"

        began = time.time()
        step = actor.propose(task.goal, shot, history)
        trace.planner_call(
            planner=_planner_name(actor), kind="vision_step", vision=True,
            prompt_tokens=0, completion_tokens=0,
            latency_s=time.time() - began, error=step.error,
        )
        trace.emit("vision_step", step=step.to_json(),
                   screenshot={k: v for k, v in (shot.value or {}).items() if k != "png"})

        if step.error:
            trace.failure(FailureClass.UNKNOWN, detail=f"vision actor error: {step.error}")
            return False, index + 1, f"vision actor error: {step.error}"
        if step.done:
            trace.note("actor_reported_done", step=index)
            return True, index + 1, None
        if not step.action:
            trace.failure(FailureClass.UNKNOWN,
                          detail="no action and no completion claim")
            return False, index + 1, "actor produced no action and did not claim completion"

        result = actor.apply(step)
        trace.action(result)
        history.append(_shot_history(step, result))
        if not result.ok and result.failure_class in _FATAL:
            trace.failure(result.failure_class, detail=result.error)
            return False, index + 1, f"{result.failure_class.value}: {result.error}"

    trace.failure(FailureClass.UNKNOWN, detail=f"step ceiling {config.max_steps}")
    return False, config.max_steps, f"step ceiling ({config.max_steps}) reached"


def run_vision_task(task: Task, actor: VisionActor, policy: Policy,
                    config: RunConfig | None = None, *, trial: int = 0,
                    trace: Trace | None = None, synthetic: bool = False,
                    teardown: bool = True) -> RunOutcome:
    """One vision-only trial, scored by the task's own independent verifiers.

    ``config.fresh_precondition`` and ``config.recovery_enabled`` are ignored
    here: this condition has no structured precondition to re-read and no
    classified failure to recover from. They are still recorded in the trace so a
    reader can see they were inert rather than assume they applied.
    """
    config = config or RunConfig(condition="vision_only")
    close_trace = trace is None
    trace = trace or Trace(task_id=task.task_id, condition=config.condition, trial=trial)

    started = time.time()
    claimed = False
    steps_used = 0
    cancelled = False
    abort_reason: str | None = None
    #: Turns begun, readable even if the loop is unwound by an interrupt.
    progress: dict[str, int] = {"steps": 0}

    trace.emit(
        "run_start", goal=task.goal, bucket=getattr(task, "bucket", ""),
        planner=_planner_name(actor), config=config.to_json(),
        workspace=str(policy.workspace),
        note="vision_only: fresh_precondition and recovery_enabled do not apply",
    )
    try:
        task.setup(policy)
        claimed, steps_used, abort_reason = _drive_vision(task, actor, config, trace,
                                                         progress)
        final = task.verify_final(policy, trace)
        trace.verification(final, checkpoint=False)
    except KeyboardInterrupt:
        # Same rule as the structured path, for the same reason: the baseline has to
        # be scored identically (plan S18), and that includes how a run nobody
        # finished is recorded. This arm drives real mouse and keyboard input, so
        # Ctrl+C is the stop button a person actually reaches for.
        cancelled = True
        steps_used = progress["steps"]
        trace.note("run_cancelled", reason="KeyboardInterrupt", steps_used=steps_used,
                   reported_success=claimed)
        abort_reason = "cancelled by user before verification finished"
        final = _cancelled(task.task_id)
    except Exception as exc:  # noqa: BLE001 - a crashed trial is a datum
        steps_used = steps_used or progress["steps"]
        trace.note("run_exception", error=f"{type(exc).__name__}: {exc}")
        abort_reason = abort_reason or f"harness error: {type(exc).__name__}: {exc}"
        final = _harness_error(task.task_id, exc)
    finally:
        if teardown and not cancelled:
            try:
                task.teardown(policy)
            except Exception as exc:  # noqa: BLE001
                trace.note("teardown_failed", error=f"{type(exc).__name__}: {exc}")

    outcome = RunOutcome(
        task_id=task.task_id, condition=config.condition, trial=trial,
        planner_name=_planner_name(actor),
        reported_success=claimed,
        verified=final.verdict, final=final,
        checkpoints=[],  # see the module docstring: no channel for these here
        failure_categories=sorted(trace.failure_categories.elements()),
        steps_used=steps_used,
        wall_clock_s=time.time() - started,
        usage=actor.usage.to_json() if getattr(actor, "usage", None) else {},
        trace=trace.summary(),
        aborted_reason=abort_reason,
        synthetic=synthetic or _planner_name(actor).endswith("scripted"),
        cancelled=cancelled,
    )
    trace.emit("run_end", outcome={k: v for k, v in outcome.to_json().items()
                                  if k not in ("trace", "checkpoints")})
    if close_trace:
        trace.close()
    return outcome

