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

    def __init__(self, action: Action, backend: Any, *, task_id: str | None = None) -> None:
        self._action = action
        if task_id:
            self.task_id = str(task_id)
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
        """Re-run the messaging verifier against the live world.

        Never report PASS merely because the executor returned or because the
        approved task reached its final state. The verifier performs a fresh
        browser observation and returns PASS/FAIL/UNKNOWN independently.
        """
        verification = self._verifier.verify(self._verifier_action(), None)
        status = str(verification.status).upper()
        if status == "PASS" and verification.ok:
            verdict = Verdict.PASS
        elif status == "FAIL":
            verdict = Verdict.FAIL
        else:
            verdict = Verdict.UNKNOWN
        return VerificationResult([
            Check(
                "approved_messaging_action",
                verdict,
                {"action_kind": self._action.kind, "status": status},
                verification.detail or "messaging action independently verified",
            )
        ], label="approved messaging")

    def _verifier_action(self) -> MessagingAction:
        return MessagingAction.from_core(self._action)

    def teardown(self, policy: Policy) -> None:
        return None
