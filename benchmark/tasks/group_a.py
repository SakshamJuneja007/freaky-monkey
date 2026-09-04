"""
Current benchmark task.

The task is intentionally simple:

Open the PDF named "Last Day" located in:

Downloads/
Gyansetu Internship Assignment/
Certificates/
Last Day.pdf

The agent should discover the file and open it through the operating system.
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
class OpenLastDayPDF:
    """
    Open the PDF named "Last Day" from the user's Downloads directory.
    """

    task_id: str = "open_last_day_pdf"

    goal: str = (
        "In the Downloads folder, open the folder named "
        "'Gyansetu Internship Assignment'. Inside it open the folder named "
        "'Certificates'. Find the PDF named 'Last Day' and open it."
    )

    bucket: str = "filesystem_application"

    assignment_folder: str = "Gyansetu Internship Assignment"
    certificates_folder: str = "Certificates"
    pdf_name: str = "Last Day.pdf"

    resolved_pdf: Path | None = None

    def downloads_dir(self) -> Path:
        """Return the current user's Downloads directory."""
        return Path.home() / "Downloads"

    def pdf_path(self) -> Path:
        """Return the expected absolute location of the PDF."""
        return (
            self.downloads_dir()
            / self.assignment_folder
            / self.certificates_folder
            / self.pdf_name
        )

    def setup(self, policy: Policy) -> None:
        """Reset task state."""
        self.resolved_pdf = None

    def observe(
        self,
        policy: Policy,
        trace: Trace | None = None,
    ) -> dict[str, Observation]:
        """
        Observe the filesystem hierarchy independently.

        The agent can use these observations to understand whether the
        requested file exists.
        """

        downloads = self.downloads_dir()

        assignment = (
            downloads
            / self.assignment_folder
        )

        certificates = (
            assignment
            / self.certificates_folder
        )

        pdf = self.pdf_path()

        return {
            "downloads": _record(
                trace,
                observe.dir_state(downloads),
                "Downloads directory",
            ),
            "assignment_folder": _record(
                trace,
                observe.dir_state(assignment),
                "Gyansetu Internship Assignment directory",
            ),
            "certificates_folder": _record(
                trace,
                observe.dir_state(certificates),
                "Certificates directory",
            ),
            "last_day_pdf": _record(
                trace,
                observe.file_state(pdf),
                "Last Day PDF",
            ),
        }

    def reference_plan(
        self,
        policy: Policy,
    ) -> list[Action]:
        """
        Reference semantic plan.

        The file is opened through the OS rather than using mouse coordinates.
        """

        pdf = self.pdf_path()
        self.resolved_pdf = pdf

        return [
            Action(
                kind="open_file",
                params={
                    "path": str(pdf),
                    "settle_s": 5.0,
                },
                rationale=(
                    "The requested PDF has a known location; opening the file "
                    "through the operating system is the narrowest semantic action."
                ),
            )
        ]

    def verify_checkpoint(
        self,
        policy: Policy,
        action: Action,
        trace: Trace | None = None,
    ) -> VerificationResult | None:
        """
        Verify the target file independently after the open_file action.

        This confirms that the intended PDF exists and is non-empty.
        """

        if action.kind != "open_file":
            return None

        return verifiers.verify_file(
            policy,
            str(self.pdf_path()),
            trace=trace,
            min_bytes=1,
        )

    def verify_final(
        self,
        policy: Policy,
        trace: Trace | None = None,
    ) -> VerificationResult:
        """
        Final independent verification.
        """

        return verifiers.verify_file(
            policy,
            str(self.pdf_path()),
            trace=trace,
            min_bytes=1,
        )

    def teardown(
        self,
        policy: Policy,
    ) -> None:
        """
        Opening a document should leave it open.

        No application processes are killed.
        """

        return None


TASK_CLASSES = (
    OpenLastDayPDF,
)


TASK_IDS: tuple[str, ...] = tuple(
    cls().task_id
    for cls in TASK_CLASSES
)


def build_task(task_id: str):
    """Build one fresh task instance."""

    for cls in TASK_CLASSES:
        instance = cls()

        if instance.task_id == task_id:
            return instance

    raise KeyError(
        f"unknown task {task_id!r}; "
        f"known: {list(TASK_IDS)}"
    )


def build_tasks(
    task_ids: list[str] | tuple[str, ...] | None = None,
):
    """Build the requested tasks, or the full current benchmark."""

    return [
        build_task(task_id)
        for task_id in (task_ids or TASK_IDS)
    ]