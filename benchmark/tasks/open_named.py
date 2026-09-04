"""Open one named file, wherever it turned out to be.

The difference from ``group_a.OpenLastDayPDF`` is a single field. That task knows
its path at import time; this one is handed a path chosen at request time, after
:func:`agent_control.api.resolve_open_request` has looked the name up in the
location index and -- when the name was ambiguous -- after the user has said which
one they meant.

Everything downstream is deliberately identical: the same ``open_file`` semantic
action, the same policy check on the path, the same independent re-observation
afterwards. Being parameterized changes what the task is pointed at, not what it
is allowed to do or how its success is judged.

Two verifiers run, and they answer different questions:

* ``verify_file`` -- the thing named is a real, non-empty file. A FAIL here means
  the index was stale or the path was wrong.
* ``verify_opened`` -- a visible window is titled like it. This is the check that
  makes the word "opened" mean anything; it returns UNKNOWN rather than PASS when
  it cannot tell, so a launch into a handler that does not name its file reports
  PARTIAL instead of a success nobody confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_control import observe, verifiers
from agent_control.policy import Policy
from agent_control.trace import Trace
from agent_control.types import Action, Observation, VerificationResult


def _record(
    trace: Trace | None,
    obs: Observation,
    purpose: str,
) -> Observation:
    return trace.observation(obs, purpose=purpose) if trace else obs


@dataclass
class OpenNamedFile:
    """Open an already-located file through the operating system."""

    task_id: str = "open_named_file"

    bucket: str = "filesystem_application"

    #: Absolute path, resolved before construction. Never a bare filename: the
    #: lookup that turns "main1.mp4" into a path is request resolution, and doing
    #: it here would put a second, unpolicied search inside the task.
    path: str = ""

    goal: str = ""

    def __post_init__(self) -> None:
        """Derive the goal from the path.

        An empty path is constructible on purpose -- ``TASK_IDS`` and
        ``main.py tasks`` build one specimen of every registered class to read its
        id and goal. It is not *runnable*: ``api.run_agent_task`` refuses the task
        without a path rather than letting it reach a planner with nothing to open.
        """
        self.goal = (
            f"Open the existing file at {self.path} using the operating system's "
            "default application for its type."
            if self.path
            else "Open a file by name, using its remembered location."
        )

    def target(self) -> Path:
        return Path(self.path)

    def setup(self, policy: Policy) -> None:
        """Nothing to reset. The file is the user's and predates the run."""
        return None

    def observe(
        self,
        policy: Policy,
        trace: Trace | None = None,
    ) -> dict[str, Observation]:
        """Read the target's own state.

        Only the file itself, not a listing of its folder: the folder is outside
        this run's single readable root (see ``api.readable_roots_for_task``), and
        enumerating a directory the policy would refuse to read would put paths in
        the planner's context that the runtime could not act on.
        """
        target = self.target()

        return {
            "target_file": _record(
                trace,
                observe.file_state(target),
                f"the file to open ({target.name})",
            ),
        }

    def reference_plan(
        self,
        policy: Policy,
    ) -> list[Action]:
        """One action. The path is known, so there is nothing to search for."""

        return [
            Action(
                kind="open_file",
                params={
                    "path": self.path,
                    "settle_s": 5.0,
                },
                rationale=(
                    "The file's location is already known, so the narrowest "
                    "semantic action is to hand the path to the operating "
                    "system's default handler."
                ),
            )
        ]

    def _verify(
        self,
        policy: Policy,
        trace: Trace | None,
    ) -> VerificationResult:
        """Existence and content, then whether it is actually on screen.

        The two results are merged directly rather than through
        ``verifiers.combine``. ``combine`` namespaces every check with its part's
        label so that repeated names stay distinguishable across the steps of a
        multi-step task; here the three checks already have distinct names, and
        namespacing them would put an absolute path inside every check name and
        therefore inside every line of the user-facing report.
        """
        parts = [
            verifiers.verify_file(
                policy,
                self.path,
                trace=trace,
                min_bytes=1,
            ),
            verifiers.verify_opened(
                policy,
                self.path,
                trace=trace,
            ),
        ]

        return VerificationResult(
            checks=[check for part in parts for check in part.checks],
            label=f"open:{self.target().name}",
        )

    def verify_checkpoint(
        self,
        policy: Policy,
        action: Action,
        trace: Trace | None = None,
    ) -> VerificationResult | None:
        if action.kind != "open_file":
            return None

        return self._verify(policy, trace)

    def verify_final(
        self,
        policy: Policy,
        trace: Trace | None = None,
    ) -> VerificationResult:
        return self._verify(policy, trace)

    def teardown(
        self,
        policy: Policy,
    ) -> None:
        """Opening a file leaves it open. Nothing is closed or killed."""

        return None
