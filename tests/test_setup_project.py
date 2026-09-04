"""Tests for creating a Python project and opening it in VS Code.

This is the first assistant task that *writes*, so most of what is worth testing
is not "does it work" but "can it lie, and can it destroy something". Six claims:

1. **The parse is lexical and narrow.** A creation verb, a project word, and an
   explicit name marker. "create a file called notes.txt" is not a project
   request, and neither is anything whose name is not a plain filesystem-safe
   token -- which is what makes ``projects_root() / name`` provably inside
   ``projects_root()``.
2. **Dependencies are only ever the ones spoken.** The package list is read from
   a literal ``with`` span and stops at the next instruction, so
   "with pandas then open it in vs code" installs pandas and not "then".
3. **An existing project is refused, not overwritten.** Resolution fails with a
   sentence a person can act on, before a planner exists.
4. **The write root is the projects directory, never the project.**
   ``Policy.__post_init__`` mkdirs the workspace, so the other choice would make
   ``dir_exists`` pass before the agent acted -- a verified success with nothing
   behind it.
5. **A checkpoint has no opinion about paths outside the project**, and never
   raises out of the runner.
6. **Nothing deletes the deliverable**, and every check has a distinct name, so
   ``api._derive`` cannot collapse eleven results into four -- including the
   checkpoints, which land in the same dictionary and so must name a file the way
   final verification names it.
7. **The observation carries the evidence a ``done`` claim needs.** The planner is
   told to claim the goal met only when the state it was handed shows it, so every
   file the goal names is read individually -- a directory listing cannot speak
   for ``tests/test_main.py`` or for a file's size.

The verification tests fabricate observations. Building a real venv per test
would cost seconds each and prove nothing about the property under test, which is
how a *reading* becomes a verdict.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_control import api, verifiers
from agent_control.policy import Policy
from agent_control.tasks.setup_project import REQUIREMENTS, VENV_DIR, SetupPythonProject
from agent_control.types import Action, Observation, PolicyDenied, Source, Verdict
from benchmark.tasks import BENCHMARK_IDS, INTERACTIVE_IDS, TASK_IDS, build_task

TASK = "setup_python_project"


@pytest.fixture
def projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A projects root of our own, so no test writes into the real one."""
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv(api.PROJECTS_ROOT_ENV, str(root))
    return root.resolve()


# ----------------------------------------------------------------------
# 1. The parse is lexical and narrow
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Set up a Python project called test_project and open it in VS Code.",
     "test_project"),
    ("set up a python project named demo_app", "demo_app"),
    ("create a python project called foo", "foo"),
    ("make a new python project called foo-bar", "foo-bar"),
    ("scaffold a project called tiny", "tiny"),
    ("bootstrap a python project titled things", "things"),
    ("setup a project called my_thing then open it", "my_thing"),
    ("start a python project called note taker", "note-taker"),   # joined
    ('create a python project called "Space Name"', "Space-Name"),
    ("initialise a python project called alpha", "alpha"),
])
def test_a_creation_request_yields_its_new_name(text: str, expected: str):
    assert api.parse_setup_request(text) == expected


@pytest.mark.parametrize("text", [
    "create a file called notes.txt",        # a file, not a project
    "make a new directory called logs",      # a directory, not a project
    "open my chess-ai project in vs code",   # opening, not creating
    "set up a python project",               # names nothing
    "create a project",                      # ditto
    "delete the project called foo",         # verb outside the set
    "a python project called foo",           # no verb at all
    "",
])
def test_anything_else_is_not_a_creation_request(text: str):
    assert api.parse_setup_request(text) is None
    assert api.resolve_setup_request(text) is None


@pytest.mark.parametrize("name", [
    "../etc", "..", ".", "/tmp/x", "C:\\Windows", "a/b", "a\\b",
    "-rf", "$env", "*", "~", "%APPDATA%",
])
def test_a_name_that_is_not_a_plain_token_is_refused(name: str):
    """The anchor of the path guarantee. ``_SAFE_NEW_NAME`` starts at an
    alphanumeric and admits only ``[A-Za-z0-9._-]``, so a name can carry no
    separator, no parent reference and no drive letter -- and therefore
    ``projects_root() / name`` cannot leave ``projects_root()``."""
    assert api.parse_setup_request(f"create a python project called {name}") is None


@pytest.mark.parametrize("spoken", [
    "../etc", "..", "/tmp/x", "C:\\Windows", "a/b", "-rf", "$env", "*",
    ".hidden", "my.thing", "UPPER_case-9", "x" * 200, "note taker",
])
def test_whatever_survives_the_parse_stays_inside_the_projects_root(
        spoken: str, projects: Path):
    """The property that actually matters, stated over the parser's whole output
    rather than one rejection list: a name either does not parse, or it is a
    single safe segment whose resolved path is strictly inside the root.

    ``.hidden`` parses as ``hidden`` -- the sentence tokeniser strips the
    punctuation that ends "called foo." along with a leading dot. That is a
    normalisation, not a hole: the name is still one plain segment, and the
    response layer says which name was used, so nothing is claimed that is not
    on disk under that name.
    """
    name = api.parse_setup_request(f"create a python project called {spoken}")
    if name is None:
        return

    assert "/" not in name and "\\" not in name
    resolved = (projects / name).resolve()

    assert resolved.is_relative_to(projects)
    assert resolved != projects


def test_a_creation_request_is_not_read_as_an_open_request():
    """The three parsers are ordered strictest-first in ``resolve_request``.
    Without that, ``parse_project_candidates`` reads "start a python project
    called foo" as a request to open a folder named "python"."""
    text = "start a python project called foo"

    assert api.parse_open_request(text) is None
    assert api.parse_setup_request(text) == "foo"
    assert api.parse_request(text) == "foo"


def test_an_open_request_is_not_stolen_by_the_setup_parser():
    text = "start my chess-ai project in vs code"

    assert api.parse_setup_request(text) is None
    assert api.parse_project_request(text) == "chess-ai project"


def test_the_new_name_is_a_name_and_never_a_path(projects: Path):
    """Resolution turns it into a path; parsing must not, or the guarantee in
    ``_SAFE_NEW_NAME`` would be checked against the wrong string."""
    assert api.parse_setup_request("create a python project called x1") == "x1"


# ----------------------------------------------------------------------
# 2. Dependencies are only ever the ones spoken
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("set up a python project called a with pandas", ("pandas",)),
    ("set up a python project called a with pandas and requests",
     ("pandas", "requests")),
    ("set up a python project called a with pandas then open it in vs code",
     ("pandas",)),
    ("set up a python project called a with pandas and open it in vs code",
     ("pandas",)),
    ("set up a python project called a using numpy", ("numpy",)),
    ("set up a python project called a including the requests library",
     ("requests",)),
    ("set up a python project called a with pytest, black and ruff",
     ("pytest", "black", "ruff")),
    ("set up a python project called a", ()),                # the normal case
    ("set up a python project called a and open it in vs code", ()),
])
def test_packages_come_only_from_what_was_said(text: str, expected: tuple):
    assert api.parse_setup_packages(text) == expected


def test_a_package_list_stops_at_the_next_instruction():
    """"then open it in vs code" is an instruction, not three dependencies."""
    assert api.parse_setup_packages(
        "create a python project called a with flask then run the tests"
    ) == ("flask",)


def test_a_package_name_cannot_be_a_pip_option():
    """``install_requirements`` passes the file to pip, which reads
    ``--index-url`` and ``-r`` out of a requirements line. A spoken word that
    became a pip option would be a remote-code-execution path with a friendly
    voice, so the name pattern refuses a leading dash."""
    assert api.parse_setup_packages(
        "create a python project called a with --index-url and pandas"
    ) == ()


def test_the_spoken_package_list_is_capped():
    many = " and ".join(f"pkg{i}" for i in range(20))
    packages = api.parse_setup_packages(
        f"create a python project called a with {many}")

    assert len(packages) == api.MAX_SPOKEN_PACKAGES
    assert packages[0] == "pkg0"


def test_no_packages_means_no_requirements_file_and_no_pip(tmp_path: Path):
    task = SetupPythonProject(path=str(tmp_path / "p"))

    assert [relative for relative, _ in task.starter_files()] == [
        "README.md", ".gitignore", "main.py", "conftest.py", "tests/test_main.py"]
    assert REQUIREMENTS not in [relative for relative, _ in task.starter_files()]
    assert "install_requirements" not in [
        a.kind for a in task.reference_plan(Policy(workspace=tmp_path))]
    assert "Do not install any packages" in task.goal


def test_named_packages_are_written_down_before_being_installed(tmp_path: Path):
    """The requirements file is the record of what was asked for, so the install
    step has a reviewable input rather than a list assembled at run time."""
    task = SetupPythonProject(path=str(tmp_path / "p"), packages=("pandas", "rich"))
    files = dict(task.starter_files())

    assert files[REQUIREMENTS] == "pandas\nrich\n"

    plan = task.reference_plan(Policy(workspace=tmp_path))
    install = [a for a in plan if a.kind == "install_requirements"]

    assert len(install) == 1
    assert install[0].params["requirements"] == str(tmp_path / "p" / REQUIREMENTS)
    assert install[0].params["venv"] == str(tmp_path / "p" / VENV_DIR)
    assert "pandas, rich" in task.goal


def test_packages_from_json_become_a_tuple(tmp_path: Path):
    """Params round-trip through JSON in a trace, where a tuple is a list."""
    task = SetupPythonProject(path=str(tmp_path / "p"), packages=["a", "", " ", "b"])

    assert task.packages == ("a", "b")


# ----------------------------------------------------------------------
# 3. An existing project is refused, not overwritten
# ----------------------------------------------------------------------

def test_a_free_name_resolves_to_a_runnable_request(projects: Path):
    resolved = api.resolve_setup_request(
        "Set up a Python project called test_project and open it in VS Code.")

    assert resolved is not None
    assert resolved.runnable is True
    assert resolved.ambiguous is False
    assert resolved.task_id == TASK
    assert Path(resolved.params["path"]) == projects / "test_project"
    assert resolved.params["packages"] == []
    assert resolved.action == "set up"


def test_an_occupied_directory_is_refused_with_a_usable_sentence(projects: Path):
    existing = projects / "taken"
    existing.mkdir()
    (existing / "main.py").write_text("mine", encoding="utf-8")

    resolved = api.resolve_setup_request("create a python project called taken")

    assert resolved.runnable is False
    assert resolved.ambiguous is False
    assert resolved.action == "set up"
    assert "already" in resolved.detail
    assert str(existing) in resolved.detail
    assert (existing / "main.py").read_text(encoding="utf-8") == "mine"


def test_an_empty_directory_is_not_an_obstacle(projects: Path):
    """An empty folder is what a cancelled earlier run leaves behind. Refusing it
    would make the assistant permanently unable to use that name."""
    (projects / "empty").mkdir()

    resolved = api.resolve_setup_request("create a python project called empty")

    assert resolved.runnable is True


def test_a_file_sitting_at_the_path_is_refused(projects: Path):
    (projects / "clash").write_text("not a directory", encoding="utf-8")

    resolved = api.resolve_setup_request("create a python project called clash")

    assert resolved.runnable is False
    assert "file" in resolved.detail


def test_the_refusal_is_phrased_as_a_creation_not_a_search(projects: Path):
    """"I could not open X: I do not know where it is" describes a search that
    never happened. ``Resolved.action`` is carried for exactly this sentence."""
    from agent_control.response import phrase_no_location

    (projects / "taken").mkdir()
    (projects / "taken" / "f").write_text("x", encoding="utf-8")
    resolved = api.resolve_setup_request("create a python project called taken")

    sentence = phrase_no_location(
        resolved.query, resolved.detail, action=resolved.action)

    assert sentence.startswith("I could not set up taken")


def test_resolution_stays_inside_the_projects_root(projects: Path):
    """Belt and braces: the name pattern already forbids separators, and the
    resolved path is re-checked against the root anyway."""
    resolved = api.resolve_setup_request("create a python project called deep")

    assert Path(resolved.params["path"]).is_relative_to(projects)
    assert Path(resolved.params["path"]) != projects


def test_the_projects_root_is_overridable_and_defaults_off_the_c_drive(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv(api.PROJECTS_ROOT_ENV, raising=False)

    assert api.projects_root() == (api.ROOT / api.PROJECTS_DIR).resolve()

    monkeypatch.setenv(api.PROJECTS_ROOT_ENV, str(tmp_path / "elsewhere"))

    assert api.projects_root() == (tmp_path / "elsewhere").resolve()


# ----------------------------------------------------------------------
# 4. The write root is the projects directory, never the project
# ----------------------------------------------------------------------

def test_the_write_root_is_the_parent_of_the_new_project(projects: Path):
    target = projects / "p"

    assert api.write_root_for_task(TASK, {"path": str(target)}) == projects


def test_the_project_directory_does_not_exist_before_the_run(projects: Path):
    """The whole point of the parent-as-workspace choice. ``Policy`` creates its
    workspace in ``__post_init__``; if that were the project, ``dir_exists``
    would pass before a single action ran."""
    target = projects / "p"
    policy = Policy(workspace=api.write_root_for_task(TASK, {"path": str(target)}),
                    refuse_if_elevated=False)

    assert policy.workspace.exists()
    assert not target.exists()


def test_the_project_path_is_writable_and_its_siblings_are_not(projects: Path):
    target = projects / "p"
    policy = Policy(workspace=api.write_root_for_task(TASK, {"path": str(target)}),
                    refuse_if_elevated=False)

    assert policy.resolve_write_path(str(target)) == target.resolve()
    assert policy.resolve_write_path(str(target / "main.py"))

    with pytest.raises(PolicyDenied):
        policy.resolve_write_path(str(projects.parent / "outside.txt"))


def test_this_task_is_granted_no_extra_read_roots(projects: Path):
    """Everything it reads is inside its own workspace, so a grant would only
    widen the blast radius for no gain."""
    assert api.readable_roots_for_task(TASK, {"path": str(projects / "p")}) == ()
    assert api.readable_roots_for_task(TASK, {}) == ()


def test_an_explicit_workspace_still_wins(projects: Path, tmp_path: Path):
    """``write_root_for_task`` is only consulted when the caller passed none, so
    the benchmark harness keeps its sandbox."""
    import inspect

    source = inspect.getsource(api.run_agent_task)

    assert "workspace is not None" in source
    assert "write_root_for_task" in source


def test_a_run_without_a_path_is_refused_before_planning():
    result = api.run_agent_task("set up a project", task_id=TASK, planner="mock")

    assert result.status is api.TaskStatus.UNSUPPORTED
    assert result.checks == []
    assert "name" in result.detail


# ----------------------------------------------------------------------
# 5. A checkpoint has no opinion about paths outside the project
# ----------------------------------------------------------------------

@pytest.fixture
def task(projects: Path) -> SetupPythonProject:
    return SetupPythonProject(path=str(projects / "demo_app"))


@pytest.fixture
def policy_for(projects: Path) -> Policy:
    return Policy(workspace=projects, refuse_if_elevated=False)


@pytest.mark.parametrize("kind,key", [
    ("create_dir", "path"),
    ("write_file", "path"),
    ("create_venv", "venv"),
    ("install_requirements", "venv"),
])
def test_a_foreign_path_gets_no_opinion(task, policy_for, kind, key, tmp_path):
    """A planner that invented a path elsewhere has already been refused by the
    policy layer, and the action carries that. A verifier cannot read outside its
    roots, so the honest checkpoint is silence, not FAIL."""
    outside = Action(kind=kind, params={key: str(tmp_path / "elsewhere")})

    assert task.verify_checkpoint(policy_for, outside) is None


@pytest.mark.parametrize("kind", ["open_file", "run_command", "click", ""])
def test_an_unrelated_action_gets_no_opinion(task, policy_for, kind):
    assert task.verify_checkpoint(policy_for, Action(kind=kind)) is None


def test_a_missing_path_parameter_gets_no_opinion(task, policy_for):
    assert task.verify_checkpoint(policy_for, Action(kind="create_dir")) is None
    assert task.verify_checkpoint(policy_for, Action(kind="write_file")) is None


def test_a_denied_action_does_not_raise_out_of_the_checkpoint(task, monkeypatch):
    """``verify_checkpoint`` runs even when the action was policy-denied. If a
    ``PolicyDenied`` escaped here it would abort the run inside the runner,
    turning a refusal that was handled into a crash."""
    def deny(*args, **kwargs):
        raise PolicyDenied("nope")

    monkeypatch.setattr(verifiers, "verify_dir", deny)
    policy = Policy(workspace=Path(task.path).parent, refuse_if_elevated=False)
    inside = Action(kind="create_dir", params={"path": task.path})

    assert task.verify_checkpoint(policy, inside) is None


def test_the_venv_checkpoint_does_not_demand_packages_yet(task, policy_for,
                                                          monkeypatch):
    """``create_venv`` runs before ``install_requirements``. A checkpoint that
    asked for the packages here would fail a correct plan at that step and send
    it into recovery."""
    task.packages = ("pandas",)
    seen: list[tuple] = []

    def fake(policy, venv_dir, packages, *, trace=None):
        seen.append(tuple(packages))
        from agent_control.types import VerificationResult
        return VerificationResult(checks=[], label="env")

    monkeypatch.setattr(verifiers, "verify_packages", fake)

    task.verify_checkpoint(policy_for,
                           Action(kind="create_venv",
                                  params={"venv": str(task.venv_dir())}))
    task.verify_checkpoint(policy_for,
                           Action(kind="install_requirements",
                                  params={"venv": str(task.venv_dir())}))

    assert seen == [(), ("pandas",)]


def observation(source: Source, value: object = None, *, ok: bool = True,
                error: str = "") -> Observation:
    """A fabricated reading. ``value`` has no default on ``Observation`` itself --
    a reading with nothing in it has to say so explicitly."""
    return Observation(source=source, query="test", value=value, ok=ok,
                       error=error or None)


def test_the_launch_checkpoint_asks_about_the_editor_only(task, policy_for,
                                                          monkeypatch):
    """Three checks and no file checks: the launch step is judged on the editor,
    because the files were judged when they were written."""
    monkeypatch.setattr(
        verifiers.observe, "process_state",
        lambda **kw: observation(Source.PROCESS,
                                 {"running": True, "count": 1, "pids": [1]}))
    monkeypatch.setattr(
        verifiers.observe, "window_state",
        lambda **kw: observation(Source.WINDOW, {
            "backend": "test", "present": True, "count": 1,
            "windows": [{"title": "demo_app - Visual Studio Code",
                         "visible": True, "pid": 1}]}))

    result = task.verify_checkpoint(policy_for, Action(kind="launch_app"))

    assert result is not None
    assert [c.name for c in result.checks] == [
        "app_process", "app_window", "viewer_window"]
    assert result.verdict is Verdict.PASS
    assert result.label == "editor:demo_app"


# ----------------------------------------------------------------------
# 6. Nothing deletes the deliverable, and every check is distinctly named
# ----------------------------------------------------------------------

def test_setup_and_teardown_touch_nothing(projects: Path, policy_for: Policy):
    """The created project is the deliverable, not scratch space. The runner only
    ever calls the task's own teardown, so a no-op here is the whole guarantee."""
    existing = projects / "p"
    existing.mkdir()
    (existing / "keep.txt").write_text("mine", encoding="utf-8")
    task = SetupPythonProject(path=str(existing))

    task.setup(policy_for)
    task.teardown(policy_for)

    assert (existing / "keep.txt").read_text(encoding="utf-8") == "mine"


def test_every_final_check_has_a_distinct_name(task, policy_for, monkeypatch):
    """``api._derive`` keys checks by name in a dict, so two checks called
    ``file_exists`` would collapse and hide a failure. The prefix is the
    project-relative path, unique by construction."""
    task.packages = ("pandas", "rich")
    monkeypatch.setattr(
        verifiers.observe, "window_state",
        lambda **kw: observation(Source.WINDOW, ok=False, error="no backend"))
    monkeypatch.setattr(
        verifiers.observe, "process_state",
        lambda **kw: observation(Source.PROCESS, ok=False, error="no backend"))

    result = task.verify_final(policy_for)
    names = [check.name for check in result.checks]

    assert len(set(names)) == len(names)
    assert "dir_exists" in names                      # unprefixed, on purpose
    assert "tests/dir_exists" in names
    assert "main.py/file_exists" in names
    assert "tests/test_main.py/file_exists" in names
    assert f"{VENV_DIR}/venv_interpreter" in names
    assert result.label == "project:demo_app"
    # The names the response layer matches on stay unprefixed, or every failure
    # sentence would fall through to the generic count.
    assert {"app_process", "app_window", "viewer_window"} <= set(names)


def test_a_missing_project_fails_rather_than_being_unknown(task, policy_for):
    """Nothing was created, so this is the shape of a failed run: FAIL on the
    directory, and no claim that anything opened."""
    result = task.verify_final(policy_for)

    assert result.verdict is Verdict.FAIL
    assert [c.name for c in result.checks if c.verdict is Verdict.FAIL][0] == \
        "dir_exists"


def test_the_failure_sentence_names_the_innermost_cause(projects: Path):
    """Response phrasing is derived from failed *check names*, so it cannot
    assert a cause verification did not establish -- and it reports the missing
    directory rather than the editor window the missing directory explains."""
    from agent_control.response import phrase_result

    result = api.AgentResult(
        request="create a python project called p", task_id=TASK,
        status=api.TaskStatus.FAILED, target="p", app="vscode",
        failed=("dir_exists", "main.py/file_exists", "app_window"),
    )
    sentence = phrase_result(result)

    assert "could not create the p project folder" in sentence
    assert "VS Code" not in sentence


def test_a_project_that_exists_without_its_files_is_not_called_set_up():
    from agent_control.response import phrase_result

    result = api.AgentResult(
        request="x", task_id=TASK, status=api.TaskStatus.FAILED,
        target="p", app="vscode",
        failed=("main.py/file_exists", "conftest.py/file_min_bytes"),
    )
    sentence = phrase_result(result)

    assert "starter files are missing or empty" in sentence
    assert "main.py" in sentence
    assert "not calling it set up" in sentence


def test_a_missing_environment_is_reported_as_such():
    from agent_control.response import phrase_result

    result = api.AgentResult(
        request="x", task_id=TASK, status=api.TaskStatus.FAILED,
        target="p", app="vscode", failed=(f"{VENV_DIR}/venv_interpreter",),
    )

    assert "virtual environment was not built" in phrase_result(result)


def test_a_package_that_did_not_install_is_named():
    from agent_control.response import phrase_result

    result = api.AgentResult(
        request="x", task_id=TASK, status=api.TaskStatus.FAILED,
        target="p", app="vscode", failed=(f"{VENV_DIR}/package:pandas",),
    )
    sentence = phrase_result(result)

    assert "pandas did not install" in sentence


def test_an_unverified_setup_is_not_described_as_opened():
    """``UNKNOWN`` for this task means "I cannot tell whether it was set up",
    which is a different sentence from the one about opening something."""
    from agent_control.response import phrase_result

    result = api.AgentResult(
        request="x", task_id=TASK, status=api.TaskStatus.UNKNOWN,
        target="p", app="vscode")
    sentence = phrase_result(result)

    assert "was set up" in sentence
    assert "opened" not in sentence


def test_a_verified_setup_is_the_only_thing_called_done():
    from agent_control.response import phrase_result

    result = api.AgentResult(
        request="x", task_id=TASK, status=api.TaskStatus.SUCCESS,
        target="p", app="vscode", completed=("dir_exists", "main.py/file_exists"))

    assert phrase_result(result) == "Set up p in VS Code. Verified 2 checks."


# ----------------------------------------------------------------------
# 7. Checkpoints name a file the way final verification names it
# ----------------------------------------------------------------------

def _derive_groups(*results) -> tuple[list[str], list[str], list[str]]:
    """The report groups ``api._derive`` would actually produce.

    Goes through the real function rather than re-implementing it, because the
    property under test *is* its dict-keyed-by-name behaviour. The last argument
    is the final verification; anything before it is a checkpoint, in order.
    """
    outcome = api.RunOutcome(
        task_id=TASK,
        condition="test",
        trial=0,
        planner_name="test",
        reported_success=True,
        verified=True,
        checkpoints=list(results[:-1]),
        final=results[-1],
    )
    completed, failed, unresolved, _checks = api._derive(outcome)
    return completed, failed, unresolved


def _derive_names(*results) -> list[str]:
    completed, failed, unresolved = _derive_groups(*results)
    return completed + failed + unresolved


def test_a_write_checkpoint_is_named_after_the_file_it_read(task, policy_for):
    """Unprefixed, six ``write_file`` checkpoints would all be called
    ``file_exists`` and five of them would be overwritten in the report."""
    (task.target() / "tests").mkdir(parents=True)
    (task.target() / "main.py").write_text("x = 1\n", encoding="utf-8")

    result = task.verify_checkpoint(
        policy_for,
        Action(kind="write_file", params={"path": str(task.target() / "main.py")}),
    )

    assert result is not None
    assert [c.name for c in result.checks] == [
        "main.py/file_exists", "main.py/file_min_bytes"]
    assert result.verdict is Verdict.PASS


def test_a_nested_write_checkpoint_keeps_the_forward_slash(task, policy_for):
    """``tests/test_main.py`` on every platform: the label is the report's key and
    a backslash on Windows would not match the name ``verify_final`` uses."""
    (task.target() / "tests").mkdir(parents=True)
    (task.target() / "tests" / "test_main.py").write_text("x = 1\n", encoding="utf-8")

    result = task.verify_checkpoint(
        policy_for,
        Action(kind="write_file",
               params={"path": str(task.target() / "tests" / "test_main.py")}),
    )

    assert [c.name for c in result.checks] == [
        "tests/test_main.py/file_exists", "tests/test_main.py/file_min_bytes"]


def test_the_project_root_checkpoint_stays_unprefixed(task, policy_for):
    """``verify_final`` calls the project's own directory check ``dir_exists``, so
    a prefix here would split one directory across two rows."""
    task.target().mkdir(parents=True)

    result = task.verify_checkpoint(
        policy_for, Action(kind="create_dir", params={"path": task.path}))

    assert [c.name for c in result.checks] == ["dir_exists"]


def test_the_tests_directory_checkpoint_is_prefixed(task, policy_for):
    (task.target() / "tests").mkdir(parents=True)

    result = task.verify_checkpoint(
        policy_for,
        Action(kind="create_dir", params={"path": str(task.target() / "tests")}),
    )

    assert [c.name for c in result.checks] == ["tests/dir_exists"]


def test_the_venv_checkpoint_is_prefixed_like_the_final_one(task, policy_for):
    """``_environment`` labels the interpreter ``.venv/venv_interpreter``; the
    checkpoint that created it has to agree. No venv is built here -- the name is
    the property under test, and the verdict is FAIL either way."""
    task.target().mkdir(parents=True)

    result = task.verify_checkpoint(
        policy_for,
        Action(kind="create_venv", params={"venv": str(task.venv_dir())}),
    )

    assert [c.name for c in result.checks] == [f"{VENV_DIR}/venv_interpreter"]


def test_a_failed_checkpoint_survives_into_the_report(task, policy_for,
                                                      monkeypatch):
    """The point of all the prefixing. Six writes, one of them empty: the report
    has to carry that file's failure rather than the last write's success."""
    (task.target() / "tests").mkdir(parents=True)
    monkeypatch.setattr(
        verifiers.observe, "window_state",
        lambda **kw: observation(Source.WINDOW, ok=False, error="no backend"))
    monkeypatch.setattr(
        verifiers.observe, "process_state",
        lambda **kw: observation(Source.PROCESS, ok=False, error="no backend"))

    checkpoints = []
    for relative, content in task.starter_files():
        path = task.target() / relative
        # main.py is written empty; everything else gets its real content.
        path.write_text("" if relative == "main.py" else content, encoding="utf-8")
        checkpoints.append(task.verify_checkpoint(
            policy_for, Action(kind="write_file", params={"path": str(path)})))

    names = _derive_names(*checkpoints, task.verify_final(policy_for))

    assert "main.py/file_min_bytes" in names
    assert "file_min_bytes" not in names          # no unprefixed duplicate row
    assert "file_exists" not in names
    _completed, failed, _unresolved = _derive_groups(
        *checkpoints, task.verify_final(policy_for))
    assert "main.py/file_min_bytes" in failed
    assert "README.md/file_min_bytes" not in failed


def test_only_four_check_names_are_ever_unprefixed(task, policy_for, monkeypatch):
    """The whole-run version of the claim, and the invariant
    ``scripts/smoke_setup_project.py`` prints after a real run.

    Exactly four names may appear bare: the project's own directory, and the three
    the response layer matches on to phrase an editor failure -- prefixing those
    would send every failure sentence through the generic check count. Any other
    bare name is a checkpoint that named a file differently from final
    verification, which is a duplicate row in the report and five hidden writes.
    """
    monkeypatch.setattr(
        verifiers.observe, "window_state",
        lambda **kw: observation(Source.WINDOW, ok=False, error="no backend"))
    monkeypatch.setattr(
        verifiers.observe, "process_state",
        lambda **kw: observation(Source.PROCESS, ok=False, error="no backend"))
    task.packages = ("pandas",)
    (task.target() / "tests").mkdir(parents=True)

    checkpoints = [task.verify_checkpoint(policy_for, action)
                   for action in task.reference_plan(policy_for)]
    names = _derive_names(
        *[c for c in checkpoints if c is not None], task.verify_final(policy_for))

    assert {n for n in names if "/" not in n} == {
        "dir_exists", "app_process", "app_window", "viewer_window"}


# ----------------------------------------------------------------------
# 8. The observation carries the evidence a ``done`` claim needs
# ----------------------------------------------------------------------

def test_every_starter_file_is_observed_by_name(task, policy_for):
    """The planner is told to claim ``done`` only when the state it was handed
    already shows the goal met. A directory listing cannot show
    ``tests/test_main.py`` a level down, so each file is read on its own."""
    state = task.observe(policy_for)

    for relative, _content in task.starter_files():
        assert f"file:{relative}" in state
    assert "file:tests/test_main.py" in state


def test_an_absent_file_is_observed_as_absent_not_as_an_error(task, policy_for):
    """Before the run nothing exists. That is a successful reading of an empty
    world, not a failed reading -- ``ok=False`` would look like a broken sensor."""
    state = task.observe(policy_for)
    reading = state["file:main.py"]

    assert reading.ok is True
    assert reading.value["exists"] is False


def test_a_written_file_is_observed_with_its_size(task, policy_for):
    """Size is the evidence ``file_min_bytes`` turns into a verdict, so the state
    the planner reasons over has to contain it. ``newline=""`` because Windows
    would otherwise translate the newline and make the byte count ambiguous."""
    (task.target() / "tests").mkdir(parents=True)
    (task.target() / "main.py").write_text("x = 1\n", encoding="utf-8", newline="")

    reading = task.observe(policy_for)["file:main.py"]

    assert reading.value["exists"] is True
    assert reading.value["is_file"] is True
    assert reading.value["size"] == 6
    assert "sha256" not in reading.value      # not asked for, so not paid for


def test_requirements_is_observed_only_when_packages_were_requested(projects: Path,
                                                                    policy_for):
    """``starter_files`` is the single source of truth, so the observation follows
    it: no packages means no requirements.txt to write and none to read."""
    bare = SetupPythonProject(path=str(projects / "demo_app"))
    withdeps = SetupPythonProject(path=str(projects / "demo_app"),
                                  packages=("pandas",))

    assert f"file:{REQUIREMENTS}" not in bare.observe(policy_for)
    assert f"file:{REQUIREMENTS}" in withdeps.observe(policy_for)


def test_the_observation_is_recorded_in_the_trace(task, policy_for, trace, events):
    """Every reading the planner sees is also on the record, with the purpose it
    was read for -- otherwise a run cannot be audited after the fact."""
    task.observe(policy_for, trace=trace)

    purposes = [
        e["purpose"] for e in events() if e.get("event") == "observation"]
    assert any("main.py" in p for p in purposes)
    assert any("tests/test_main.py" in p for p in purposes)


def test_a_finished_project_is_observable_as_finished(task, policy_for):
    """The end state the planner has to recognise: every file present and
    non-empty, and the interpreter directory there. If this state were not
    reportable, no run could ever legitimately claim to be done."""
    (task.target() / "tests").mkdir(parents=True)
    for relative, content in task.starter_files():
        (task.target() / relative).write_text(content, encoding="utf-8")
    task.venv_dir().mkdir()

    state = task.observe(policy_for)

    assert state["project_dir"].value["exists"] is True
    assert state["venv_dir"].value["exists"] is True
    assert all(
        state[f"file:{relative}"].value["size"] > 0
        for relative, _content in task.starter_files()
    )



def test_the_plan_is_in_dependency_order(task, policy_for):
    kinds = [action.kind for action in task.reference_plan(policy_for)]

    assert kinds == ["create_dir", "create_dir"] + ["write_file"] * 5 + [
        "create_venv", "launch_app"]


def test_the_tests_directory_is_created_before_the_test_file(task, policy_for):
    plan = task.reference_plan(policy_for)
    made = [a.params["path"] for a in plan if a.kind == "create_dir"]
    written = [a.params["path"] for a in plan if a.kind == "write_file"]

    assert str(task.target() / "tests") in made
    assert made.index(str(task.target() / "tests")) < len(made)
    assert str(task.target() / "tests" / "test_main.py") in written


def test_the_editor_is_opened_last_and_given_time_to_settle(task, policy_for):
    plan = task.reference_plan(policy_for)

    assert plan[-1].kind == "launch_app"
    assert plan[-1].params["app"] == "vscode"
    assert plan[-1].params["open_path"] == task.path
    assert plan[-1].params["settle_s"] >= 5.0


def test_every_written_file_has_real_content(task):
    for relative, content in task.starter_files():
        assert content.strip(), relative
        assert "__NAME__" not in content, relative


def test_the_starter_test_can_actually_run(tmp_path: Path):
    """The starter files are a claim that a fresh project works. Written to disk
    and run with this interpreter, because a template that does not import is a
    project that fails its own test on the user's first try."""
    import subprocess
    import sys

    project = tmp_path / "fresh"
    (project / "tests").mkdir(parents=True)
    for relative, content in SetupPythonProject(path=str(project)).starter_files():
        (project / relative).write_text(content, encoding="utf-8")

    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(project)],
        capture_output=True, text=True, timeout=180,
    )

    assert run.returncode == 0, run.stdout + run.stderr


def test_a_created_project_cannot_hijack_this_repository_s_own_tests():
    """The flip side of the test above, and a real break that happened.

    A created project is a working project: it has a ``conftest.py`` that inserts
    its own root at the front of ``sys.path`` and a ``tests/`` directory. Projects
    land in ``projects/`` inside this repository, so unscoped collection picks
    both up -- and then ``import main`` in this repo's tests resolves to the
    created project's ``main.py``, failing twelve unrelated tests with
    ``module 'main' has no attribute 'cmd_voice_test'``. The suite broke every
    time the assistant did the thing it exists to do, so collection is scoped in
    ``pytest.ini``. Asserted by collecting for real, because the property is about
    what pytest does, not about what the file says.
    """
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only"],
        cwd=root, capture_output=True, text=True, timeout=300,
    )

    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]
    strayed = [
        line for line in run.stdout.splitlines()
        if line.startswith("projects/") or line.startswith("projects\\")
    ]
    assert strayed == [], strayed


def test_a_specimen_builds_without_a_path():
    specimen = build_task(TASK)

    assert specimen.task_id == TASK
    assert specimen.goal
    assert specimen.path == ""
    assert specimen.packages == ()


def test_build_task_passes_parameters_through(projects: Path):
    built = build_task(TASK, path=str(projects / "p"), packages=["pandas"])

    assert built.path == str(projects / "p")
    assert built.packages == ("pandas",)
    assert str(projects / "p") in built.goal


def test_an_invented_parameter_is_a_named_error():
    with pytest.raises(TypeError):
        build_task(TASK, colour="red")


def test_the_task_is_registered_but_not_benchmarked():
    """It writes to disk and needs a name, so an unattended trial row would be
    meaningless -- but a person can still ask for it."""
    assert TASK in TASK_IDS
    assert TASK in INTERACTIVE_IDS
    assert TASK not in BENCHMARK_IDS
    assert TASK in api.registered_tasks()
    assert api.resolve_task(TASK) == TASK


def test_there_is_exactly_one_execution_entry_point():
    """The architectural invariant. A second pipeline is how a voice command and
    a typed command start behaving differently."""
    from agent_control import session

    assert session.api.run_agent_task is api.run_agent_task
