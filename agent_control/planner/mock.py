"""Deterministic planner (no API calls).

Two jobs:

1. Let the harness, verifiers, recovery, and tracing be exercised and tested
   without spending tokens or inheriting LLM nondeterminism.
2. Let the *measurement instruments* be validated. ``false_claim=True`` produces
   an agent that declares success having done nothing -- so the false-success
   metric can be shown to actually fire, rather than merely reading 0.00 and
   being assumed correct. Plan S22 lists shallow verification as a kill
   condition; this is how that is checked.

Results produced with this planner are plumbing checks, not experimental
results, and the harness marks them ``synthetic: true``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Action
from .base import PlannerStep, Usage


@dataclass
class MockPlanner:
    """Replays a task's reference plan."""

    reference_plan: list[Action] = field(default_factory=list)
    #: Claim completion without emitting any action -- a synthetic false success.
    false_claim: bool = False
    #: Indices of the reference plan to withhold, simulating an incomplete plan.
    skip_indices: frozenset[int] = frozenset()
    #: Emit one action per turn instead of the whole plan at once.
    incremental: bool = False

    name: str = "mock"
    usage: Usage = field(default_factory=Usage)
    _cursor: int = 0

    def plan(self, goal: str, state: dict, history: list[dict]) -> PlannerStep:
        # Counted as a call so latency/call-count columns stay comparable in shape
        # to the LLM conditions, with zero tokens.
        self.usage.add(prompt=0, completion=0)

        if self.false_claim:
            return PlannerStep(
                actions=[], done=True,
                reasoning="synthetic false success: claims completion without acting",
            )

        pending = [
            (index, action) for index, action in enumerate(self.reference_plan)
            if index >= self._cursor and index not in self.skip_indices
        ]
        if not pending:
            broken = [h for h in history
                      if not h.get("ok") or h.get("checkpoint") == "FAIL"]
            if broken:
                # A replay device has no alternative plan to offer, and claiming
                # completion on top of a failed action would *manufacture* a false
                # success rather than measure one -- contaminating exactly the
                # column plan S19 cares about. ``false_claim=True`` is the switch
                # for when a synthetic false success is what's wanted.
                return PlannerStep(
                    actions=[], done=False,
                    reasoning=f"reference plan exhausted; {len(broken)} action(s) "
                              "failed and a replay planner has no re-plan",
                )
            return PlannerStep(actions=[], done=True, reasoning="reference plan exhausted")

        if self.incremental:
            index, action = pending[0]
            self._cursor = index + 1
            return PlannerStep(actions=[action], done=False, reasoning="next reference step")

        self._cursor = len(self.reference_plan)
        return PlannerStep(
            actions=[action for _, action in pending], done=False,
            reasoning="full reference plan",
        )
