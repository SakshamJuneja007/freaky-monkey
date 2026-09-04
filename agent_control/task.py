"""Task contract shared by the runner, the baseline, and the harness.

A task owns four things and nothing else: the goal in natural language, what
state is relevant, a reference plan (for the deterministic planner and for
ablations), and how to verify itself independently.

Keeping verification inside the task -- next to the goal, not next to the actions
-- is what stops a verifier from quietly re-asserting whatever the executor just
claimed (plan S8).
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol, runtime_checkable

from .policy import Policy
from .trace import Trace
from .types import Action, Observation, VerificationResult


@runtime_checkable
class Task(Protocol):
    task_id: str
    goal: str
    #: Task-bucket label for plan S18's per-bucket analysis.
    bucket: str

    def setup(self, policy: Policy) -> None:
        """Put the disposable workspace into the task's starting state."""
        ...

    def observe(self, policy: Policy, trace: Trace | None = None) -> dict[str, Observation]:
        """Read the state this task's decisions depend on."""
        ...

    def reference_plan(self, policy: Policy) -> list[Action]:
        """The known-good action sequence, replayed by the deterministic planner."""
        ...

    def verify_final(self, policy: Policy, trace: Trace | None = None) -> VerificationResult:
        """Independently check the goal. Must not read any ActionResult."""
        ...

    def verify_checkpoint(self, policy: Policy, action: Action,
                          trace: Trace | None = None) -> VerificationResult | None:
        """Check the sub-goal an action was meant to reach, or None if untracked."""
        ...

    def teardown(self, policy: Policy) -> None:
        ...


def state_fingerprint(observations: dict[str, Observation]) -> str:
    """Content digest of a state snapshot, ignoring *when* it was read.

    Used to detect that state changed between the planner seeing it and the
    runtime acting on it -- the adversarial-timing experiment in plan S20.

    Caveat worth remembering when reading results: an application still settling
    (VS Code opening a second window a beat later) also changes this digest, so a
    detected change is "the world moved", not proven external interference. The
    trace records which, since the injector logs its own mutations.
    """
    payload = {
        name: {"ok": obs.ok, "value": obs.value, "error": obs.error}
        for name, obs in sorted(observations.items())
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def oldest_age_s(observations: dict[str, Observation]) -> float:
    """Age of the stalest reading in a snapshot."""
    return max((obs.age() for obs in observations.values()), default=0.0)
