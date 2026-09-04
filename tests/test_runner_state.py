"""Tests for the runner's caller-supplied planner context (``extra_state``).

Narrow on purpose: this file covers only the seam Feature 2 added to the control
loop, not the loop as a whole. Three properties, and the third is the one worth
having:

1. Caller context reaches the planner alongside the observations.
2. A real reading wins a name collision, so a cache can never overwrite something
   this run actually measured.
3. ``extra_state`` takes no part in the state fingerprint, so remembered paths can
   neither satisfy nor trip the freshness precondition that guards consequential
   actions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_control import runner
from agent_control.recovery import Recovery, RecoveryBudget
from agent_control.runner import RunConfig, _Loop
from agent_control.task import state_fingerprint
from agent_control.types import Observation, Source


class Recording:
    """A planner that records the state it was shown and then stops."""

    name = "recording"

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def plan(self, goal: str, state: dict, history: list[dict]):
        from agent_control.planner.base import PlannerStep

        self.seen.append(state)
        return PlannerStep(done=True, reasoning="noted")


class Goal:
    """The smallest thing satisfying the ``Task`` protocol.

    It observes one value and verifies nothing, so a test of the state seam does
    not also depend on a verifier's behaviour.
    """

    task_id = "extra_state"
    goal = "do the thing"
    bucket = "test"

    def setup(self, policy) -> None:
        return None

    def observe(self, policy, trace=None) -> dict:
        return {"pdf_state": observation("present")}

    def reference_plan(self, policy) -> list:
        return []

    def verify_final(self, policy, trace=None):
        from agent_control.types import VerificationResult

        return VerificationResult(label=self.task_id, checks=[])

    def verify_checkpoint(self, policy, action, trace=None):
        return None

    def teardown(self, policy) -> None:
        return None


def observation(value: str) -> Observation:
    return Observation(source=Source.FILESYSTEM, query="q", value=value)


def loop_with(trace, *, observations: dict, extra: dict) -> _Loop:
    from agent_control.policy import Policy

    return _Loop(
        task=Goal(), policy=Policy(workspace=Path("ws"), refuse_if_elevated=False),
        config=RunConfig(), trace=trace,
        recovery=Recovery(budget=RecoveryBudget(), enabled=True),
        observations=observations, extra_state=extra,
    )


def test_caller_context_reaches_the_planner(trace):
    planner = Recording()
    loop = loop_with(trace, observations={"pdf_state": observation("present")},
                     extra={"remembered_locations": {"value": {"query": "x"}}})

    runner._plan(loop, planner, [])

    state = planner.seen[0]
    assert "pdf_state" in state
    assert "remembered_locations" in state


def test_an_observation_wins_a_name_collision(trace):
    """A cached path must not be able to shadow a live reading. If the two ever
    share a key, the one taken by this run is the one the planner sees."""
    planner = Recording()
    loop = loop_with(
        trace,
        observations={"pdf_state": observation("measured now")},
        extra={"pdf_state": {"value": "remembered last week"}},
    )

    runner._plan(loop, planner, [])

    assert planner.seen[0]["pdf_state"]["value"] == "measured now"


def test_caller_context_is_absent_from_the_state_fingerprint(trace):
    """The fingerprint is how the loop notices the world moved under it. Adding
    remembered locations to it would make a cache look like a state change --
    and, worse, could let one satisfy a freshness check it never observed."""
    observations = {"pdf_state": observation("present")}
    bare = loop_with(trace, observations=observations, extra={})
    with_memory = loop_with(trace, observations=observations,
                            extra={"remembered_locations": {"value": {}}})

    assert (state_fingerprint(bare.observations)
            == state_fingerprint(with_memory.observations))
    assert "remembered_locations" not in with_memory.observations


def test_no_caller_context_leaves_the_state_untouched(trace):
    """The default path must be byte-for-byte what it was before the feature."""
    from agent_control.observe import summarize

    planner = Recording()
    observations = {"pdf_state": observation("present")}
    loop = loop_with(trace, observations=observations, extra={})

    runner._plan(loop, planner, [])

    assert list(planner.seen[0]) == list(summarize(observations))


def test_run_task_defaults_to_no_caller_context(trace, tmp_path):
    """``extra_state`` is optional; omitting it must not change behaviour."""
    from agent_control.policy import Policy

    outcome = runner.run_task(
        Goal(), Recording(),
        Policy(workspace=tmp_path / "ws", refuse_if_elevated=False),
        RunConfig(max_steps=1), trace=trace, teardown=False,
    )

    # The planner claimed done immediately and the stub task verifies nothing, so
    # the only assertion that matters here is that the call shape still works.
    assert outcome.reported_success is True
    assert outcome.verified_success is False
