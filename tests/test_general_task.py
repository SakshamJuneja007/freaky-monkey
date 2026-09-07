"""Route 2 -- the general computer action -- as an execution path.

``tests/test_session.py`` covers which route a sentence takes. This module
covers what happens once Route 2 is chosen: that the id it invents can be
spelled as a directory on this platform, that a run which dies during assembly
becomes a reported result instead of ending the session, and that a request
naming a file *type* plus a rule for choosing among them is not read as a
request about a folder with that type's name.

Each test here corresponds to a defect that was reachable from the chat prompt,
so they are written against the same entry points a person uses -- ``Session``
and ``api.resolve_request`` -- rather than against internals that could be made
to pass while the prompt stayed broken.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent_control import api
from agent_control import session as sess
from agent_control.api import AgentResult, TaskStatus
from agent_control.general_task import GeneralTask
from agent_control.memory import Entry, FileMemory
from agent_control.policy import Policy
from agent_control.response import Narrator
from agent_control.session import Session
from agent_control.types import Action, Verdict

#: Characters Windows forbids in a path component. ":" is the one that actually
#: bit -- it reads as a drive separator, so ``mkdir`` raises NotADirectoryError
#: (WinError 267) rather than anything naming the character -- but the id is
#: checked against the whole set, because the property wanted here is "spellable
#: as a directory", not "contains no colon".
_FORBIDDEN = set('<>:"/\\|?*')


@pytest.fixture
def printed() -> list[str]:
    return []


@pytest.fixture
def chat(printed: list[str]) -> Session:
    return Session(narrator=Narrator(speaker=None, write=printed.append),
                   planner="mock")


@pytest.fixture
def index(tmp_path: Path) -> FileMemory:
    """A location index holding a real ``Downloads`` folder with one named PDF
    in it -- the smallest world in which all three routings below have a
    definite answer.

    Built by hand for the same reason ``test_open_project`` and
    ``test_open_named`` build theirs: the resolvers re-check the filesystem, so
    the files have to exist, and a test that consulted *this* machine's index
    would pass or fail for reasons having nothing to do with the gate.
    """
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    named = downloads / "report.pdf"
    named.write_bytes(b"%PDF-1.4\n")

    memory = FileMemory(tmp_path / "idx.json")
    memory.entries = [
        Entry(path=str(path), kind=kind, size=1024, mtime=time.time(),
              depth=len(path.parts), root=str(path.anchor))
        for kind, path in (("dir", downloads), ("file", named))
    ]
    memory.roots = []
    memory.refreshed_at = time.time()
    memory.loaded = True
    return memory


# ----------------------------------------------------------------------
# The id has to survive being used as a path
# ----------------------------------------------------------------------

def test_a_general_task_id_can_be_spelled_as_a_directory():
    """The id is not decoration: it becomes a literal path component of the run
    workspace (:func:`api.default_workspace`), which ``Policy`` then creates."""
    task_id = GeneralTask(request="open the last pdf").task_id

    assert task_id.startswith("general-"), "the route stays legible in the id"
    assert not (set(task_id) & _FORBIDDEN), task_id


def test_the_workspace_for_a_general_task_can_actually_be_created(tmp_path: Path):
    """The regression itself, exercised the way it failed: derive a workspace
    from the id and let the real ``Policy`` create it. Before the fix this raised
    ``NotADirectoryError`` on Windows and took the chat loop down with it."""
    task_id = GeneralTask(request="open the last pdf").task_id

    policy = Policy(workspace=(tmp_path / task_id).resolve(),
                    refuse_if_elevated=False)

    assert policy.workspace.is_dir()


def test_the_default_workspace_path_for_a_general_task_is_creatable(tmp_path: Path):
    """``default_workspace`` is what a run uses when no workspace was named, so
    the id has to survive *that* composition too -- not just a hand-built path.
    Only the trailing component is reused; the timestamped parent is not the
    subject here."""
    task_id = GeneralTask(request="open the last pdf").task_id
    tail = api.default_workspace(task_id).name

    (tmp_path / tail).mkdir(parents=True)

    assert (tmp_path / tail).is_dir()


# ----------------------------------------------------------------------
# A failed run is a result, not the end of the session
# ----------------------------------------------------------------------

def test_a_run_that_could_not_be_assembled_is_reported_not_raised(tmp_path: Path):
    """``run_agent_task`` owns this half: a workspace that cannot be created is
    an ``OSError`` out of ``Policy``, and the caller gets ``UNKNOWN`` with the
    exception preserved rather than the exception itself.

    A file standing where the workspace directory must go is the cheapest
    portable way to make a real ``mkdir`` fail; the Windows original was a colon
    in the path, which is the same failure through a different door.
    """
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("", encoding="utf-8")

    result = api.run_agent_task(
        "open the last pdf",
        task_obj=GeneralTask(request="open the last pdf"),
        workspace=blocked / "inside",
        planner="mock",
    )

    assert result.status is TaskStatus.UNKNOWN
    assert result.ok is False
    assert "could not prepare the run" in result.detail
    assert "Error" in result.detail, "the exception type is kept, not discarded"


def test_the_session_survives_a_run_that_raises(chat: Session, monkeypatch):
    """Containment at the session boundary. ``main.cmd_chat`` catches only
    ``KeyboardInterrupt``, so anything escaping here ends the conversation. The
    turn must come back as an honest failure and the session must stay usable.
    """
    def explode(request, **kwargs):
        raise RuntimeError("planner exploded")

    monkeypatch.setattr(sess.api, "run_agent_task", explode)

    turn = chat.submit("open the last pdf")

    assert turn.result is not None
    assert turn.result.status is TaskStatus.UNKNOWN
    assert turn.ok is False, "a crash is never success"
    assert "RuntimeError" in turn.result.detail
    assert "planner exploded" in turn.result.detail, "the message is preserved"
    assert chat.history[-1] is turn, "the turn was recorded"

    # Still alive: a second request goes through normally.
    monkeypatch.setattr(
        sess.api, "run_agent_task",
        lambda request, **kw: AgentResult(request=request, task_id="x",
                                          status=TaskStatus.SUCCESS,
                                          verified="PASS"),
    )

    assert chat.submit("open the last pdf").ok is True


def test_a_keyboard_interrupt_still_reaches_the_caller(chat: Session, monkeypatch):
    """The containment above must not swallow Ctrl-C. ``except Exception`` does
    not catch ``KeyboardInterrupt``, which is what lets ``main.cmd_chat`` stop a
    session when the person asks it to."""
    def interrupted(request, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(sess.api, "run_agent_task", interrupted)

    with pytest.raises(KeyboardInterrupt):
        chat.submit("open the last pdf")


# ----------------------------------------------------------------------
# Routing: a general action is eligible, and is not a folder request
# ----------------------------------------------------------------------

@pytest.mark.parametrize("request_text", [
    "Open the last PDF.",
    "Open the most recently modified PDF in my Downloads folder.",
    "Find the largest Python file in my project folder and tell me its name "
    "and exact number of lines.",
    "Find a file named definitely_does_not_exist_12345.txt in my Downloads "
    "folder and open it.",
])
def test_a_general_computer_action_reaches_the_pipeline(request_text: str,
                                                        chat: Session,
                                                        monkeypatch):
    """None of these name a registered workflow, and every one of them asks for
    the machine to be touched. They must arrive at the one execution call as a
    ``GeneralTask`` carrying the sentence -- not be refused for having no
    workflow, and not be rewritten into the nearest registered task id.
    """
    seen: list[dict] = []

    def fake(request, **kwargs):
        seen.append({"request": request, **kwargs})
        return AgentResult(request=request, task_id=kwargs.get("task_id", ""),
                           status=TaskStatus.SUCCESS, verified="PASS")

    monkeypatch.setattr(sess.api, "run_agent_task", fake)

    chat.submit(request_text)

    assert len(seen) == 1, "it was neither refused nor asked about"
    assert seen[0]["task_obj"] is not None, "it went as a GeneralTask"
    assert seen[0]["task_id"].startswith("general-")
    # Compared against normalization rather than a re-spelling of it, so this
    # test cannot disagree with `normalize` about trailing punctuation.
    assert seen[0]["request"] == sess.normalize(request_text).text
    assert seen[0]["readable_roots"], "the read grant travels with it"


def test_a_file_type_and_a_selection_are_not_read_as_a_folder_name(
    index: FileMemory,
):
    """The Issue-5 regression. "Open the most recently modified PDF in my
    Downloads folder" contains the word "folder", which is what opens the
    project resolver -- and its readings reduce to the bare word "pdf", so the
    sentence used to be refused with "no folder named pdf". A sentence that picks
    a file *type* by a property names no folder, and must fall through to the
    general route instead.
    """
    text = "Open the most recently modified PDF in my Downloads folder."

    assert api.describes_a_selection(text) is True
    assert api.resolve_request(text, memory=index) is None, (
        "no resolver may claim it, so the session's general route gets it"
    )


def test_without_the_gate_the_same_sentence_is_answered_as_a_folder(
    index: FileMemory, monkeypatch,
):
    """What the gate is actually preventing, shown rather than asserted about.

    With the selection condition switched off, the widest resolver claims the
    sentence and answers it -- and its answer is about a folder, which is not
    what was asked. That is the misread; the gate exists to leave the sentence
    unclaimed instead.
    """
    monkeypatch.setattr(api, "describes_a_selection", lambda text: False)

    resolved = api.resolve_request(
        "Open the most recently modified PDF in my Downloads folder.",
        memory=index,
    )

    assert resolved is not None, "the project resolver would have claimed it"
    assert resolved.runnable is False, "and could only refuse"
    assert "pdf" in (resolved.query + resolved.detail).lower(), (
        "having read the file type as the folder name"
    )


def test_naming_a_file_outright_is_still_resolved_normally(index: FileMemory):
    """The gate takes two signals on purpose: a *named* file carries no
    selection word, so nothing about the change touches the request shape that
    already worked."""
    resolved = api.resolve_request("Open report.pdf", memory=index)

    assert resolved is not None, "still claimed by the named-file resolver"
    assert resolved.task_id == "open_named_file"
    assert resolved.runnable is True


def test_naming_a_folder_outright_is_still_resolved_normally(index: FileMemory):
    """Same point from the other side: the project resolver keeps every sentence
    it could answer before. Only the ones it could only mis-answer are withheld.
    """
    resolved = api.resolve_request("open the downloads folder", memory=index)

    assert resolved is not None
    assert resolved.task_id == "open_project_in_vscode"
    assert resolved.runnable is True


@pytest.mark.parametrize("text,expected", [
    ("Open the most recently modified PDF in my Downloads folder.", True),
    ("Open the last PDF.", True),
    ("Open report.pdf", False),                     # a name, no selection
    ("open the downloads folder", False),           # a folder, no file type
    ("open my chess-ai project in vscode", False),
    ("set up a python project called tools", False),
])
def test_the_selection_gate_needs_both_signals(text: str, expected: bool):
    """Neither signal alone is enough. A selection word without a spoken file
    type is an ordinary sentence, and a file type without a selection word is a
    name -- it takes both to mean "you pick which one"."""
    assert api.describes_a_selection(text) is expected


def test_verified_directory_context_keeps_chained_files_inside_it(
    chat: Session,
    monkeypatch,
    tmp_path: Path,
):
    """Only verified effects become pronoun context, and all turns keep the
    same policy write root so the remembered directory grants no new access.
    """
    chat.workspace = tmp_path / "conversation"
    seen_context: list[dict[str, str]] = []
    written: list[Path] = []

    def execute_and_verify(request: str, **kwargs):
        task = kwargs["task_obj"]
        context = dict(kwargs["recent_context"])
        seen_context.append(context)
        policy = Policy(
            workspace=Path(kwargs["workspace"]),
            readable_roots=kwargs["readable_roots"],
            refuse_if_elevated=False,
        )

        if "BananaTest987" in request:
            target = policy.workspace / "BananaTest987"
            target.mkdir(parents=True)
            action = Action("create_dir", {"path": str(target)})
        else:
            filename = "notes.txt" if "notes.txt" in request else "ideas.txt"
            target = Path(context["last_verified_directory"]) / filename
            target.write_text("three AI project ideas", encoding="utf-8")
            written.append(target)
            action = Action(
                "write_file",
                {"path": str(target), "content": "three AI project ideas"},
            )

        checkpoint = task.verify_checkpoint(policy, action)
        assert checkpoint is not None and checkpoint.verdict is Verdict.PASS
        return AgentResult(
            request=request,
            task_id=task.task_id,
            status=TaskStatus.SUCCESS,
            verified=Verdict.PASS.value,
        )

    monkeypatch.setattr(sess.api, "run_agent_task", execute_and_verify)

    chat.submit("create a folder called BananaTest987")
    chat.submit("make notes.txt in that folder")
    chat.submit("create ideas.txt there")

    banana = (chat.workspace / "BananaTest987").resolve()
    assert seen_context[0] == {}
    assert seen_context[1]["last_verified_directory"] == str(banana)
    assert seen_context[2]["last_verified_directory"] == str(banana)
    assert written == [banana / "notes.txt", banana / "ideas.txt"]
    assert all(path.is_file() for path in written)
    assert chat.recent_context.last_verified_file == banana / "ideas.txt"
    assert chat.recent_context.last_goal == "create ideas.txt there"


def test_file_content_about_an_ai_project_does_not_select_project_setup(
    chat: Session,
    monkeypatch,
):
    request = (
        "Now inside that folder create a file called notes.txt and write three "
        "ideas for an AI project."
    )
    calls: list[dict] = []

    def fake(text, **kwargs):
        calls.append({"request": text, **kwargs})
        return AgentResult(
            request=text,
            task_id=kwargs["task_id"],
            status=TaskStatus.SUCCESS,
            verified=Verdict.PASS.value,
        )

    monkeypatch.setattr(sess.api, "run_agent_task", fake)

    turn = chat.submit(request)

    assert api.parse_setup_candidates(request) == ()
    assert turn.executed is True
    assert len(calls) == 1
    assert calls[0]["task_obj"] is not None
    assert calls[0]["task_id"].startswith("general-")


def test_explicit_named_python_project_still_selects_registered_setup() -> None:
    resolved = api.resolve_request(
        "set up a Python project called ProjectIntentGuard",
        use_memory=False,
    )

    assert resolved is not None
    assert resolved.task_id == "setup_python_project"


def test_verified_equivalent_effect_is_not_executed_twice(
    tmp_path: Path,
) -> None:
    from agent_control.planner.base import PlannerStep, Usage
    from agent_control.runner import RunConfig, run_task
    from agent_control.trace import Trace

    target = (tmp_path / "ws" / "OnlyOnce").resolve()
    action = Action("create_dir", {"path": str(target)})

    class RepeatingPlanner:
        name = "repeating"

        def __init__(self) -> None:
            self.calls = 0
            self.usage = Usage()

        def plan(self, goal, state, history):
            self.calls += 1
            self.usage.add(prompt=0, completion=0)
            if self.calls <= 2:
                return PlannerStep(actions=[action], done=False)
            return PlannerStep(done=True)

    events: list[dict] = []
    task = GeneralTask(request="create a folder called OnlyOnce")
    policy = Policy(workspace=tmp_path / "ws", refuse_if_elevated=False)
    trace = Trace(
        task_id=task.task_id,
        condition="test",
        enabled=False,
        on_event=events.append,
    )

    outcome = run_task(
        task,
        RepeatingPlanner(),
        policy,
        RunConfig(max_steps=3),
        trace=trace,
        teardown=False,
    )

    assert outcome.verified is Verdict.PASS
    assert len([event for event in events if event["event"] == "action"]) == 1
    skipped = [event for event in events if event["event"] == "action_skipped"]
    assert len(skipped) == 1
    assert target.is_dir()


def test_planner_timeout_after_verified_requested_effect_stays_complete(
    tmp_path: Path,
) -> None:
    from agent_control.planner.base import PlannerStep, Usage
    from agent_control.runner import RunConfig, run_task

    target = (tmp_path / "ws" / "TimeoutComplete").resolve()

    class ThenTimeout:
        name = "then-timeout"

        def __init__(self) -> None:
            self.calls = 0
            self.usage = Usage()

        def plan(self, goal, state, history):
            self.calls += 1
            self.usage.add(prompt=0, completion=0)
            if self.calls == 1:
                return PlannerStep(actions=[
                    Action("create_dir", {"path": str(target)})
                ])
            return PlannerStep(error="planner transport error: ReadTimeout")

    task = GeneralTask(request="create a folder called TimeoutComplete")
    outcome = run_task(
        task,
        ThenTimeout(),
        Policy(workspace=tmp_path / "ws", refuse_if_elevated=False),
        RunConfig(max_steps=3),
        trace=None,
        teardown=False,
    )

    assert outcome.verified is Verdict.PASS
    assert outcome.aborted_reason is None
    names = {check.name for check in outcome.final.checks}
    assert any(name.endswith("/dir_exists") for name in names)
    assert not names & {
        ".gitignore/file_exists", ".venv/venv_exists", "README.md/file_exists",
        "conftest.py/file_exists", "main.py/file_exists",
        "tests/test_main.py/file_exists",
    }


def test_planner_timeout_before_all_requested_effects_is_unknown(
    tmp_path: Path,
) -> None:
    from agent_control.planner.base import PlannerStep, Usage
    from agent_control.runner import RunConfig, run_task

    target = (tmp_path / "ws" / "Partial").resolve()

    class PartialThenTimeout:
        name = "partial-timeout"

        def __init__(self) -> None:
            self.calls = 0
            self.usage = Usage()

        def plan(self, goal, state, history):
            self.calls += 1
            self.usage.add(prompt=0, completion=0)
            if self.calls == 1:
                return PlannerStep(actions=[
                    Action("create_dir", {"path": str(target)})
                ])
            return PlannerStep(error="planner transport error: ReadTimeout")

    task = GeneralTask(
        request="create a folder called Partial and write file notes.txt"
    )
    outcome = run_task(
        task,
        PartialThenTimeout(),
        Policy(workspace=tmp_path / "ws", refuse_if_elevated=False),
        RunConfig(max_steps=3),
        trace=None,
        teardown=False,
    )

    assert outcome.verified is Verdict.UNKNOWN
    assert any(
        check.name.endswith("/goal_completion")
        and check.verdict is Verdict.UNKNOWN
        for check in outcome.final.checks
    )
    assert all("README" not in check.name for check in outcome.final.checks)


def test_real_three_turn_file_context_uses_exact_verified_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise the real Session/API/Policy/executor/verifier/ledger path.

    Only the external planner transport is scripted. Filesystem actions and
    independent verification are the production implementations.
    """
    import json

    from agent_control import os_tools
    from agent_control.planner import openai_compat
    from agent_control.planner.base import PlannerStep, Usage

    workspace = (tmp_path / "conversation").resolve()
    emitted: list[Action] = []
    executions: list[dict] = []

    class ScriptedPlanner:
        name = "scripted-context"

        def __init__(self) -> None:
            self.usage = Usage()

        def plan(self, goal, state, history):
            self.usage.add(prompt=0, completion=0)
            if history:
                return PlannerStep(done=True)

            write_root = Path(state["path_permissions"]["write_root"])
            if goal == "create folder TestContext":
                action = Action(
                    "create_dir",
                    {"path": str(write_root / "TestContext")},
                )
            elif goal == "create notes.txt in that folder":
                directory = Path(
                    state["recent_context"]["last_verified_directory"]
                )
                action = Action(
                    "write_file",
                    {"path": str(directory / "notes.txt"), "content": ""},
                )
            elif goal == 'write "updated content" to that file':
                target = Path(state["recent_context"]["last_verified_file"])
                action = Action(
                    "write_file",
                    {"path": str(target), "content": "updated content"},
                )
            else:  # pragma: no cover - the assertion makes an unexpected goal loud
                raise AssertionError(goal)

            emitted.append(action)
            return PlannerStep(actions=[action])

    monkeypatch.setattr(
        openai_compat.LLMClient,
        "from_env",
        classmethod(lambda cls, **kwargs: object()),
    )
    monkeypatch.setattr(
        openai_compat,
        "OpenAICompatPlanner",
        lambda client: ScriptedPlanner(),
    )

    real_execute = os_tools.execute

    def execute_with_audit(policy, action):
        result = real_execute(policy, action)
        executions.append({
            "workspace": policy.workspace,
            "kind": action.kind,
            "raw_params": dict(action.params),
            "resolved_path": result.detail.get("path"),
        })
        return result

    monkeypatch.setattr(os_tools, "execute", execute_with_audit)

    printed: list[str] = []
    chat = Session(
        narrator=Narrator(speaker=None, write=printed.append),
        planner="llm",
        workspace=workspace,
        keep_workspace=True,
        debug=True,
    )

    first = chat.submit("create folder TestContext")
    directory_context = chat.recent_context.last_verified_directory
    second = chat.submit("create notes.txt in that folder")
    file_context = chat.recent_context.last_verified_file
    third = chat.submit('write "updated content" to that file')

    expected_dir = workspace / "TestContext"
    expected_file = expected_dir / "notes.txt"

    assert all(turn.ok for turn in (first, second, third))
    assert [action.kind for action in emitted] == [
        "create_dir", "write_file", "write_file",
    ]
    assert emitted[1].kind == "write_file"
    assert emitted[1].kind != "create_dir"
    assert emitted[1].params["path"] == str(expected_file)

    assert {entry["workspace"] for entry in executions} == {workspace}
    assert executions[0]["resolved_path"] == str(expected_dir)
    assert executions[1]["resolved_path"] == str(expected_file)
    assert executions[2]["resolved_path"] == str(expected_file)

    assert directory_context == expected_dir
    assert file_context == expected_file
    assert chat.recent_context.last_verified_directory == expected_dir
    assert chat.recent_context.last_verified_file == expected_file

    assert expected_dir.is_dir()
    assert expected_file.exists()
    assert expected_file.is_file()
    assert not (workspace / "notes.txt").is_dir()
    assert expected_file.read_text(encoding="utf-8") == "updated content"

    expected_verifier_paths = [expected_dir, expected_file, expected_file]
    for turn, expected in zip(
        (first, second, third), expected_verifier_paths,
    ):
        evidence_paths = {
            Path(check.evidence["path"])
            for check in turn.result.outcome.final.checks
            if "path" in check.evidence
        }
        assert evidence_paths == {expected}

        trace_rows = [
            json.loads(line)
            for line in Path(turn.result.trace_file).read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        recorded = [
            row["effect"] for row in trace_rows
            if row["event"] == "effect_recorded"
        ]
        assert len(recorded) == 1
        assert Path(recorded[0]["target"]) == expected

    debug_output = "\n".join(printed)
    assert f"recent_context.last_verified_directory = {expected_dir}" in debug_output
    assert f"recent_context.last_verified_file = {expected_file}" in debug_output
