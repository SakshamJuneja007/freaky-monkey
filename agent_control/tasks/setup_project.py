"""Create a new Python project on disk, then open it in VS Code.

The first assistant task that *writes*. Everything else the assistant does reads
the machine and hands something to an application; this one produces the thing it
is then asked to open, which changes two things about how it has to be built.

**The write root is the projects directory, never the project itself.**
``Policy.__post_init__`` creates ``policy.workspace`` before the first action
runs, so a workspace pointed at the new project would make ``dir_exists`` pass
before the agent had done anything -- a verified success with nothing behind it,
which is the exact failure this project exists to rule out. The workspace is
therefore the *parent*: ``api.write_root_for_task`` returns it, the project
directory is created by an action inside it, and the check that it exists is
evidence. Nothing outside that one directory becomes writable, and nothing
outside it is read either -- ``api.readable_roots_for_task`` grants this task no
extra read roots at all, because every path it touches is inside the workspace.

**Teardown deletes nothing.** The created project is the deliverable, not scratch
space, so :meth:`teardown` is a no-op regardless of ``keep_workspace``. The
runner only ever calls the task's own teardown, so this is enough to keep the
project on disk.

Verification is per-file rather than one aggregate check, and the check names are
namespaced by project-relative path (``main.py/file_exists``). Two reasons: a
report that says "4 of 11 checks failed" is useless without knowing *which*, and
``api._derive`` keys checks by name, so three files verified under the name
``file_exists`` would collapse into one and hide two results. Checkpoints use the
same names for the same reason -- they land in the same dictionary, so a
checkpoint that named its checks differently would either duplicate a file's row
in the report or bury five of the six ``write_file`` checkpoints behind the last
one.

**The observation reports every file the goal names.** A planner is told to claim
``done`` only when the state it was handed already shows the goal met. A
directory listing cannot show that: it says nothing about ``tests/test_main.py``
one level down, and nothing about size, while verification requires every file to
be non-empty. So :meth:`observe` reads each starter file individually. Without
that, the only honest ways to end a run are to exhaust ``max_steps`` or to guess,
and a real run spent two of three planning rounds rewriting files that were
already on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_control import observe, verifiers
from agent_control.os_tools import APP_REGISTRY
from agent_control.policy import Policy
from agent_control.trace import Trace
from agent_control.types import (
    Action,
    Check,
    Observation,
    PolicyDenied,
    VerificationResult,
)

#: Placeholder substituted into the starter files. ``str.replace`` rather than
#: ``str.format`` on purpose: the templates are Python source and contain braces
#: of their own, and escaping them would make the templates unreadable for the
#: sake of a formatting call that buys nothing.
_NAME = "__NAME__"

_README = """# __NAME__

A minimal Python project, laid out flat so that the starter test can import the
module it tests without a packaging step.

    __NAME__/
      main.py             the module
      tests/test_main.py  its test
      conftest.py         puts the project root on sys.path for pytest
      .venv/              an isolated interpreter
      .gitignore

Run the test with the project's own interpreter:

    .venv/Scripts/python -m pytest        # Windows
    .venv/bin/python -m pytest            # macOS, Linux
"""

_GITIGNORE = """.venv/
__pycache__/
*.py[cod]
.pytest_cache/
"""

_MAIN_PY = '''"""Entry point for __NAME__."""


def greet(who: str) -> str:
    """Return a greeting. Written to be worth testing, not to be clever."""
    return "Hello, " + who + "!"


def main() -> None:
    print(greet("__NAME__"))


if __name__ == "__main__":
    main()
'''

_CONFTEST = '''"""Put the project root on sys.path so tests can import the module.

pytest's default import mode adds the *test* directory to sys.path, not the
project root, so a flat layout needs this one line to be importable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
'''

_TEST_MAIN = '''"""One real test, so a fresh project can prove itself immediately."""

from main import greet


def test_greet_names_the_argument() -> None:
    assert greet("world") == "Hello, world!"
'''

#: Name of the isolated interpreter directory, relative to the project.
VENV_DIR = ".venv"

#: Where explicitly requested dependencies are recorded before installation.
REQUIREMENTS = "requirements.txt"


def _record(trace: Trace | None, obs: Observation, purpose: str) -> Observation:
    return trace.observation(obs, purpose=purpose) if trace else obs


def _prefixed(prefix: str, part: VerificationResult) -> list[Check]:
    """Re-label a sub-verification's checks by what they were checking.

    ``file_exists`` on its own is ambiguous once more than one file is verified,
    and ``api._derive`` keys by check name, so the duplicates would overwrite
    each other. The prefix is the project-relative path, which is short enough to
    read aloud in a report and unique by construction.

    An empty prefix means "this *is* the project root", which already has an
    unambiguous name, so the checks pass through untouched rather than picking up
    a leading slash.
    """
    if not prefix:
        return list(part.checks)
    return [
        Check(
            name=f"{prefix}/{check.name}",
            verdict=check.verdict,
            evidence=check.evidence,
            reason=check.reason,
        )
        for check in part.checks
    ]


@dataclass
class SetupPythonProject:
    """Create a Python project at ``path`` and open it in VS Code."""

    task_id: str = "setup_python_project"

    bucket: str = "filesystem_environment"

    #: The registry key, fixed for the same reason ``OpenProjectInEditor`` fixes
    #: it: the task's own name says which editor, so a parameter that could point
    #: somewhere else would make every trace line and spoken sentence about it
    #: wrong.
    app: str = "vscode"

    #: Absolute path of the project directory *to create*. Its parent is the
    #: write root for the run, so this is never a bare name -- choosing the
    #: directory is request resolution (``api.resolve_setup_request``), which is
    #: also where an existing project is refused rather than overwritten.
    path: str = ""

    #: Dependencies the user *named out loud*, and nothing else. Empty is the
    #: normal case and means no ``requirements.txt`` is written and pip is never
    #: run: a starter project that quietly installed what a model guessed at
    #: would be a supply-chain decision made on the user's behalf.
    packages: tuple[str, ...] = ()

    goal: str = ""

    def __post_init__(self) -> None:
        # Params can arrive from JSON, where a tuple is a list. Normalising here
        # keeps ``packages`` hashable and keeps the goal sentence stable.
        self.packages = tuple(
            str(name) for name in (self.packages or ()) if str(name).strip()
        )
        self.goal = self._goal()

    # -- geometry ----------------------------------------------------------
    @property
    def name(self) -> str:
        return Path(self.path).name

    def target(self) -> Path:
        return Path(self.path)

    def venv_dir(self) -> Path:
        return self.target() / VENV_DIR

    def requirements_file(self) -> Path:
        return self.target() / REQUIREMENTS

    def starter_files(self) -> list[tuple[str, str]]:
        """``(project-relative path, content)`` for every file to be written.

        One list, used by the plan, by verification, and by the goal sentence, so
        a file cannot be created without being checked or checked without being
        created.
        """
        files = [
            ("README.md", _README),
            (".gitignore", _GITIGNORE),
            ("main.py", _MAIN_PY),
            ("conftest.py", _CONFTEST),
            ("tests/test_main.py", _TEST_MAIN),
        ]
        if self.packages:
            files.append(
                (REQUIREMENTS, "\n".join(self.packages) + "\n"),
            )
        return [
            (relative, content.replace(_NAME, self.name or "project"))
            for relative, content in files
        ]

    def _goal(self) -> str:
        """The instruction a planner receives, naming actions and paths exactly.

        Explicit to the point of being a script, and deliberately so: this task
        writes to disk, and a planner improvising the layout would produce a
        project that verification then reports as incomplete. The same list drives
        :meth:`reference_plan`, so the goal and the oracle cannot drift.
        """
        if not self.path:
            return (
                "Create a new Python project in the allowed projects directory "
                "and open it in VS Code."
            )

        steps = [f"create_dir {self.path}", f"create_dir {self.target() / 'tests'}"]
        steps += [
            f"write_file {self.target() / relative}"
            for relative, _content in self.starter_files()
        ]
        steps.append(f"create_venv {self.venv_dir()}")
        if self.packages:
            steps.append(
                f"install_requirements venv={self.venv_dir()} "
                f"requirements={self.requirements_file()}"
            )
        steps.append(f"launch_app vscode with open_path {self.path}")

        listed = (
            f"Install only these packages, which the user named: "
            f"{', '.join(self.packages)}. "
            if self.packages
            else "Do not install any packages: none were requested. "
        )

        return (
            f"Set up a new Python project at {self.path} and open it in VS Code. "
            f"Perform these actions in order: {'; '.join(steps)}. "
            + listed
            + "Write real starter content into every file; an empty file fails "
            "verification."
        )

    # -- lifecycle ---------------------------------------------------------
    def setup(self, policy: Policy) -> None:
        """Nothing to reset, and deliberately nothing to clear.

        A pre-existing project is refused during request resolution rather than
        deleted here. Deleting a directory the user already has, in order to
        satisfy a request to create one, is not a step this task is willing to
        take on its own.
        """
        return None

    def observe(
        self,
        policy: Policy,
        trace: Trace | None = None,
    ) -> dict[str, Observation]:
        """The projects root, the project, every starter file, and the editor."""
        target = self.target()
        names = APP_REGISTRY.get(self.app, {}).get("process_names") or [self.app]

        state = {
            "projects_root": _record(
                trace,
                observe.dir_state(target.parent),
                "the directory new projects are created in (the write root)",
            ),
            "project_dir": _record(
                trace,
                observe.dir_state(target),
                f"the project to create ({target.name}); absent before the run",
            ),
            "venv_dir": _record(
                trace,
                observe.dir_state(self.venv_dir()),
                "the isolated interpreter directory, once it exists",
            ),
            "editor_before": _record(
                trace,
                observe.process_state(name_in=names),
                f"whether {self.app} was already running before this run",
            ),
        }

        # One reading per starter file, keyed by the same project-relative path
        # verification uses. ``dir_state`` above lists the project's top level,
        # which is not enough on two counts: it says nothing about
        # ``tests/test_main.py`` a level down, and nothing about size, while
        # ``verify_final`` requires every file to be non-empty. The planner is
        # told to claim ``done`` only when the state it was handed already shows
        # the goal met, so the state has to carry that evidence -- otherwise the
        # only way to finish is to run out of steps or guess, and a real run
        # spends extra rounds rewriting files that were already written.
        #
        # ``want_hash=False``: the contents are ours to begin with, so a digest
        # would be re-read on every round of every run to answer a question
        # nothing asks.
        for relative, _content in self.starter_files():
            state[f"file:{relative}"] = _record(
                trace,
                observe.file_state(target / relative, want_hash=False),
                f"{relative}: whether it exists yet, and how many bytes it holds",
            )

        return state

    def reference_plan(self, policy: Policy) -> list[Action]:
        """Directories, then files, then the environment, then the editor.

        The order is a dependency order rather than a preference: ``tests/`` has
        to exist before a file is written into it, the interpreter has to exist
        before anything is installed into it, and the editor is opened last so
        that what it shows is the finished project.
        """
        target = self.target()

        actions = [
            Action(
                kind="create_dir",
                params={"path": str(target)},
                rationale=(
                    "The project directory is the one thing every later action "
                    "writes into, so it is created first."
                ),
            ),
            Action(
                kind="create_dir",
                params={"path": str(target / "tests")},
                rationale="The test file needs its directory to exist first.",
            ),
        ]

        actions += [
            Action(
                kind="write_file",
                params={"path": str(target / relative), "content": content},
                rationale=f"Starter content for {relative}.",
            )
            for relative, content in self.starter_files()
        ]

        actions.append(
            Action(
                kind="create_venv",
                params={"venv": str(self.venv_dir())},
                rationale=(
                    "A project-local interpreter, so installing anything later "
                    "cannot touch the system environment."
                ),
            )
        )

        if self.packages:
            actions.append(
                Action(
                    kind="install_requirements",
                    params={
                        "venv": str(self.venv_dir()),
                        "requirements": str(self.requirements_file()),
                    },
                    rationale=(
                        "Install exactly the packages the user named, from the "
                        "requirements file just written."
                    ),
                )
            )

        actions.append(
            Action(
                kind="launch_app",
                params={
                    "app": self.app,
                    "open_path": self.path,
                    # Same reasoning as OpenProjectInEditor: verification reads a
                    # window title, and a cold VS Code start draws its window
                    # before it finishes naming the folder in it.
                    "settle_s": 8.0,
                },
                rationale=(
                    "The project now exists, so the last step is to show it in "
                    "the editor the user asked for."
                ),
            )
        )

        return actions

    # -- verification ------------------------------------------------------
    def _inside(self, raw: object) -> Path | None:
        """``raw`` as a path inside this project, or ``None``.

        A checkpoint is only ever an opinion about the action that just ran, so a
        path the planner invented somewhere else gets no opinion rather than a
        FAIL: the action's own ``ok`` already carries the policy refusal, and a
        verifier asked to read outside its roots would raise instead of report.
        """
        if not raw or not self.path:
            return None
        try:
            candidate = Path(str(raw)).resolve()
        except OSError:
            return None
        target = self.target().resolve()
        return candidate if candidate.is_relative_to(target) else None

    def _relative(self, inside: Path) -> str:
        """``inside`` as a project-relative label, ``""`` for the project itself.

        The label has to be the *same* one ``verify_final`` uses, because
        ``api._derive`` keys checks by name across checkpoints and final
        verification alike. Left unprefixed, a checkpoint check called
        ``file_exists`` becomes a second entry in the user-facing report for a
        file that also has ``main.py/file_exists`` -- and the six ``write_file``
        checkpoints of one run all share that one name, so five of them are
        overwritten and a failed write can end up hidden behind a later passing
        one. The run's verdict is safe either way (``verify_final`` re-reads
        everything from disk and decides), but the report a person is read is
        not, and a report that hides a failure is the thing this project exists
        to rule out.
        """
        try:
            relative = inside.resolve().relative_to(self.target().resolve())
        except (OSError, ValueError):  # pragma: no cover - _inside guarantees it
            return ""
        return "" if relative == Path(".") else relative.as_posix()

    def _labelled(
        self,
        inside: Path,
        part: VerificationResult,
    ) -> VerificationResult:
        """``part`` with its checks named the way ``verify_final`` names them."""
        return VerificationResult(
            checks=_prefixed(self._relative(inside), part),
            label=part.label,
        )

    def _files(self, policy: Policy, trace: Trace | None) -> list[Check]:
        """Every starter file, each check named after the file it read."""
        checks: list[Check] = []
        for relative, _content in self.starter_files():
            part = verifiers.verify_file(
                policy, str(self.target() / relative), trace=trace, min_bytes=1,
            )
            checks += _prefixed(relative, part)
        return checks

    def _environment(self, policy: Policy, trace: Trace | None) -> list[Check]:
        """An isolated interpreter, plus every package that was asked for.

        ``verify_packages`` imports each one *inside the new environment*, so a
        pip exit code of 0 with nothing installed is caught here.
        """
        part = verifiers.verify_packages(
            policy, str(self.venv_dir()), self.packages, trace=trace,
        )
        return _prefixed(VENV_DIR, part)

    def _editor(self, policy: Policy, trace: Trace | None) -> list[Check]:
        """The editor is running, and a visible window names this project."""
        return [
            check
            for part in (
                verifiers.verify_app_running(policy, self.app, trace=trace),
                verifiers.verify_opened(
                    policy, self.path, needle=self.name, trace=trace,
                ),
            )
            for check in part.checks
        ]

    def verify_final(
        self,
        policy: Policy,
        trace: Trace | None = None,
    ) -> VerificationResult:
        """Directory, files, environment, editor -- all of it, re-read from disk.

        Nothing here consults an ``ActionResult``; the plan could have reported
        every step as fine and this would still fail if the project is not there.
        """
        checks = [
            *verifiers.verify_dir(policy, self.path, trace=trace).checks,
            *_prefixed(
                "tests",
                verifiers.verify_dir(policy, str(self.target() / "tests"), trace=trace),
            ),
            *self._files(policy, trace),
            *self._environment(policy, trace),
            *self._editor(policy, trace),
        ]
        return VerificationResult(checks=checks, label=f"project:{self.name}")

    def verify_checkpoint(
        self,
        policy: Policy,
        action: Action,
        trace: Trace | None = None,
    ) -> VerificationResult | None:
        """Only what the action just did -- never the whole goal.

        A checkpoint gates progress: ``runner._run_action`` treats a non-PASS as a
        failure and recovers or replans. Verifying the finished project after the
        first ``create_dir`` would therefore fail every run at step one, so each
        action is judged on its own effect and ``None`` means "no opinion".
        """
        try:
            return self._checkpoint(policy, action, trace)
        except PolicyDenied:
            # The action was denied for the same reason, and its own result says
            # so. A verifier that cannot legally read the path has no evidence to
            # offer, which is not the same as evidence of failure.
            return None

    def _checkpoint(
        self,
        policy: Policy,
        action: Action,
        trace: Trace | None,
    ) -> VerificationResult | None:
        if action.kind == "create_dir":
            inside = self._inside(action.params.get("path"))
            return (
                self._labelled(
                    inside, verifiers.verify_dir(policy, str(inside), trace=trace)
                )
                if inside
                else None
            )

        if action.kind == "write_file":
            inside = self._inside(action.params.get("path"))
            return (
                self._labelled(
                    inside,
                    verifiers.verify_file(
                        policy, str(inside), trace=trace, min_bytes=1,
                    ),
                )
                if inside
                else None
            )

        if action.kind == "create_venv":
            inside = self._inside(action.params.get("venv"))
            # Packages are deliberately not checked here: nothing has been
            # installed yet, and a checkpoint that fails on an empty environment
            # would send a correct plan into recovery.
            return (
                self._labelled(
                    inside, verifiers.verify_packages(policy, str(inside), (), trace=trace)
                )
                if inside
                else None
            )

        if action.kind == "install_requirements":
            inside = self._inside(action.params.get("venv"))
            return (
                self._labelled(
                    inside,
                    verifiers.verify_packages(
                        policy, str(inside), self.packages, trace=trace,
                    ),
                )
                if inside
                else None
            )

        if action.kind == "launch_app":
            return VerificationResult(
                checks=self._editor(policy, trace), label=f"editor:{self.name}",
            )

        return None

    def teardown(self, policy: Policy) -> None:
        """Deletes nothing. The project is what was asked for.

        The runner never removes a workspace itself -- it calls this method -- so
        a no-op here is what keeps the new project on disk whatever
        ``keep_workspace`` was set to.
        """
        return None
