"""Open one already-located project directory in VS Code.

The sibling of ``benchmark.tasks.open_named.OpenNamedFile``, and deliberately
built from the same parts. That task hands a *file* to whatever the shell has
associated with its extension; this one hands a *directory* to one named,
registered application. Both are parameterized at request time, both are granted
exactly one readable root, and both are judged by re-observing the machine rather
than by believing the action layer.

Why a separate task rather than a parameter on the file one: the two differ in
what "it worked" means. A file is opened by an unknown handler that may not name
it in any window title, so ``open_named_file`` has to treat a missing title as
UNKNOWN. A project is opened by an application this project has a registry entry
for, so there are three independent things to check -- the directory is really
there, the editor process is really running, and a visible window is really
titled after the project -- and a failure in each of them means something
different.

Three verifiers run, and the distinctions between them are the point:

* ``verify_dir`` -- the thing named is a real directory. A FAIL here means the
  location index was stale and nothing should have been launched.
* ``verify_app_running`` -- an editor process exists and some window of it is
  titled like VS Code. A process with no window is a crash back to nothing.
* ``verify_opened`` -- a visible window is titled after *this* project. This is
  what separates "VS Code is running" (it may have been running all along) from
  "VS Code is showing what was asked for".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_control import observe, verifiers
from agent_control.os_tools import APP_REGISTRY
from agent_control.policy import Policy
from agent_control.trace import Trace
from agent_control.types import Action, Observation, VerificationResult


def _record(trace: Trace | None, obs: Observation, purpose: str) -> Observation:
    return trace.observation(obs, purpose=purpose) if trace else obs


@dataclass
class OpenProjectInEditor:
    """Open an already-located project directory in VS Code."""

    task_id: str = "open_project_in_vscode"

    bucket: str = "filesystem_application"

    #: The registry key, not a path and not user input. Fixed rather than a
    #: parameter because the task id names the application: a task called
    #: ``open_project_in_vscode`` that could be pointed at something else would
    #: make every trace line and every spoken sentence about it wrong.
    app: str = "vscode"

    #: Absolute path to the project directory, resolved before construction.
    #: Never a bare name: turning "chess-ai" into a path is request resolution
    #: (``api.resolve_project_request``), and repeating it here would put a
    #: second, unpolicied search inside the task.
    path: str = ""

    goal: str = ""

    def __post_init__(self) -> None:
        """Derive the goal from the path.

        An empty path is constructible on purpose -- ``TASK_IDS`` and
        ``main.py tasks`` build one specimen of every registered class to read
        its id and goal. It is not *runnable*: ``api.run_agent_task`` refuses the
        task without a path rather than letting it reach a planner with nothing
        to open.
        """
        self.goal = (
            f"Open the existing project directory at {self.path} in VS Code, "
            "using the launch_app action."
            if self.path
            else "Open a project directory in VS Code, using its remembered location."
        )

    def target(self) -> Path:
        return Path(self.path)

    def setup(self, policy: Policy) -> None:
        """Nothing to reset. The project is the user's and predates the run."""
        return None

    def observe(
        self,
        policy: Policy,
        trace: Trace | None = None,
    ) -> dict[str, Observation]:
        """The project's own contents, and whether the editor is already up.

        The directory listing is in scope here where it was not for
        ``OpenNamedFile``: the project folder *is* this run's single readable
        root, so its entries are inside what the runtime may act on. Knowing the
        editor's prior state matters because ``app_process`` cannot distinguish a
        process this run started from one that was already running, and the
        planner should be able to see that too.
        """
        target = self.target()
        names = APP_REGISTRY.get(self.app, {}).get("process_names") or [self.app]

        return {
            "project_dir": _record(
                trace,
                observe.dir_state(target),
                f"the project to open ({target.name})",
            ),
            "editor_before": _record(
                trace,
                observe.process_state(name_in=names),
                f"whether {self.app} was already running before this run",
            ),
        }

    def reference_plan(
        self,
        policy: Policy,
    ) -> list[Action]:
        """One action. The path is known, so there is nothing to search for."""

        return [
            Action(
                kind="launch_app",
                params={
                    "app": self.app,
                    "open_path": self.path,
                    # Longer than launch_app's own default: verification reads a
                    # window *title*, and a cold VS Code start draws its window
                    # before it finishes naming the folder in it. Settling too
                    # early turns a success into an UNKNOWN.
                    "settle_s": 8.0,
                },
                rationale=(
                    "The project's location is already known and VS Code is a "
                    "registered application, so the narrowest semantic action is "
                    "to launch that one application against that one directory."
                ),
            )
        ]

    def _verify(
        self,
        policy: Policy,
        trace: Trace | None,
    ) -> VerificationResult:
        """Directory, then process, then this project's own window.

        Merged directly rather than through ``verifiers.combine`` for the same
        reason ``OpenNamedFile`` does it: the four check names are already
        distinct, and namespacing them would put an absolute path into every
        check name and therefore into every line of the user-facing report.
        """
        parts = [
            verifiers.verify_dir(policy, self.path, trace=trace),
            verifiers.verify_app_running(policy, self.app, trace=trace),
            # ``needle`` overrides the default stem: a directory called
            # ``my.project`` has the stem ``my``, which would match almost any
            # window. The whole folder name is what an editor puts in its title.
            verifiers.verify_opened(
                policy,
                self.path,
                needle=self.target().name,
                trace=trace,
            ),
        ]

        return VerificationResult(
            checks=[check for part in parts for check in part.checks],
            label=f"project:{self.target().name}",
        )

    def verify_checkpoint(
        self,
        policy: Policy,
        action: Action,
        trace: Trace | None = None,
    ) -> VerificationResult | None:
        if action.kind != "launch_app":
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
        """Opening a project leaves it open. The editor is not closed or killed."""

        return None
