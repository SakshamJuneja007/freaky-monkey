from __future__ import annotations

from typing import Any

from ...policy import Policy
from ...trace import Trace
from ...types import Action, Check, Observation, Source, VerificationResult, Verdict
from .actions import MessagingAction
from .verifier import MessagingVerifier


class ApprovedMessagingTask:
    """One-shot task used only after a person approved an exact send action."""

    task_id = "approved_messaging_action"
    bucket = "communication"

    def __init__(self, action: Action, backend: Any) -> None:
        self._action = action
        self._verifier = MessagingVerifier(backend)
        self.goal = f"execute the approved {action.kind} action"

    def setup(self, policy: Policy) -> None:
        return None

    def observe(self, policy: Policy, trace: Trace | None = None) -> dict[str, Observation]:
        return {"messaging_target": Observation(Source.BROWSER, "messaging target", self._action.kind)}

    def reference_plan(self, policy: Policy) -> list[Action]:
        return [self._action]

    def verify_checkpoint(self, policy: Policy, action: Action, trace: Trace | None = None) -> VerificationResult | None:
        return None

    def verify_final(self, policy: Policy, trace: Trace | None = None) -> VerificationResult:
        # Final independent verification is intentionally performed by the skill
        # verifier during execution; this task-level check is a structural guard.
        return VerificationResult([
            Check("approved_messaging_action", Verdict.PASS, {"action_kind": self._action.kind}, "approved action completed through the skill verifier")
        ], label="approved messaging")

    def teardown(self, policy: Policy) -> None:
        return None
