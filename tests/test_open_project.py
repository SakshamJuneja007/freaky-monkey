"""Tests for opening a project folder the user named, in VS Code.

The sibling of ``test_open_named.py``, and the path with no coverage until now:
*"open my hermes project in vs code"*. Six claims hold it up.

1. **The parse is lexical and gated twice.** A verb from the closed set is not
   enough -- the sentence must also signal a *directory*, either with a project
   word or by naming the editor. Without the second gate "open notes" becomes a
   folder search and the honest refusal turns into a guess.
2. **A possessive welded onto the name is repaired, not guessed.** This is not
   hypothetical. Reading the scenario's own sentence aloud on this machine comes
   back from the speech provider as ``Open MyHermes project in VS Code``, and the
   first real voice run refused it: *"no folder named myhermes is in the location
   index (16764 entries known)"*. The refusal was correct for the words it was
   given, and the words were wrong. The repair adds a reading; it never replaces
   one, so a folder genuinely called ``myhermes`` still wins.
3. **A remembered folder is a candidate, not a fact.** Every hit is re-checked
   against the disk, an entry that is a *file* is not a folder, and a name that is
   nowhere is reported as nowhere -- with the plainest reading of the sentence in
   it, not the longest.
4. **The read grant is the project's own subtree.** Wider than the single-file
   grant because an editor reads the tree and ``verify_dir`` has to list it;
   narrower than anything else, so a resolution that returned the wrong folder
   cannot become a way to read the drive.
5. **"Opened" has to mean three separate things.** The directory is real, the
   editor process is up, and a visible window is titled after *this* project.
   Each failure means something different, and the window check is the only one
   that distinguishes "VS Code is running" from "VS Code is showing what was
   asked for".
6. **One action, through the existing action layer.** The reference plan is a
   single ``launch_app`` against the resolved path -- no search inside the task,
   and no run at all without a path.

The window and process tests fabricate observations, for the reason
``test_open_named.py`` gives: the property under test is how a *reading* becomes a
verdict, and that part does not need a real editor. What does need a real editor is
covered by ``scripts/smoke_voice_open_project.py``, which opens one.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent_control import api, verifiers
from agent_control import memory as memory_module
from agent_control import session as session_module
from agent_control.api import AgentResult
from agent_control.memory import Entry, FileMemory
from agent_control.policy import Policy, PolicyDenied
from agent_control.response import Narrator
from agent_control.session import Session
from agent_control.tasks.open_project import OpenProjectInEditor
from agent_control.types import Action, Observation, Source, Verdict

TASK = "open_project_in_vscode"


def indexed(dirs: list[Path], files: list[Path] | None = None,
            *, store: Path) -> FileMemory:
    """A ``FileMemory`` holding exactly these locations, touching no disk scan."""
    memory = FileMemory(store)
    memory.entries = [
        Entry(path=str(path), kind=kind, size=1024, mtime=time.time(),
              depth=len(path.parts), root=str(path.anchor))
        for kind, group in (("dir", dirs), ("file", files or []))
        for path in group
    ]
    memory.roots = []
    memory.refreshed_at = time.time()
    memory.loaded = True
    return memory


@pytest.fixture
def one_project(tmp_path: Path) -> tuple[FileMemory, Path]:
    """One folder called ``hermes``, on disk and in the index."""
    project = tmp_path / "projects" / "hermes"
    (project / "tests").mkdir(parents=True)
    (project / "main.py").write_text("x = 1\n", encoding="utf-8")
    return indexed([project], store=tmp_path / "idx.json"), project


@pytest.fixture
def two_projects(tmp_path: Path) -> tuple[FileMemory, Path, Path]:
    """The same folder name in two places -- the shape that must ask, not guess."""
    first = tmp_path / "Downloads" / "hermes"
    second = tmp_path / "work" / "hermes"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    return indexed([first, second], store=tmp_path / "idx.json"), first, second


# ----------------------------------------------------------------------
# 1. The parse is lexical and gated twice
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("open my hermes project in vs code", "hermes project"),
    ("open the chess-ai folder", "chess-ai folder"),
    ("open hermes in vscode", "hermes"),
    ("open project rocket-os", "rocket-os"),
    ('open "rocket os" in vscode', "rocket os"),
])
def test_a_project_request_yields_a_folder_name(text: str, expected: str):
    assert api.parse_project_request(text) == expected


def test_the_longest_reading_comes_first_and_the_shorter_one_follows():
    """The word that signals a project can also be part of its name -- this
    repository is called ``agentic-project`` -- so both readings are offered, in
    the order that costs at most one lookup that finds nothing."""
    candidates = api.parse_project_candidates("open the chess-ai folder")

    assert candidates[0] == "chess-ai folder"
    assert "chess-ai" in candidates
    assert candidates.index("chess-ai folder") < candidates.index("chess-ai")


def test_a_spoken_name_is_offered_in_the_spellings_a_folder_might_use():
    """Speech does not pronounce separators. "chess ai" and "chess-ai" are one
    utterance and only one of them is on disk."""
    candidates = api.parse_project_candidates("open my chess ai project in vscode")

    assert {"chess ai", "chess-ai", "chess_ai", "chessai"} <= set(candidates)


@pytest.mark.parametrize("text", [
    "open notes",                 # no directory signal at all
    "open the folder",            # signals a directory, names none
    "hermes project in vs code",  # names one, has no verb
    "",
])
def test_anything_else_names_no_project(text: str):
    assert api.parse_project_candidates(text) == ()
    assert api.parse_project_request(text) is None


def test_naming_only_the_editor_is_not_naming_a_project():
    """"open vscode" reduces to the editor's own name. Searching the disk for a
    folder called "vs code" would answer a request nobody made."""
    assert api.parse_project_candidates("open vscode") == ()


# ----------------------------------------------------------------------
# 2. A possessive welded onto the name is repaired, not guessed
# ----------------------------------------------------------------------

#: What the speech provider actually returned for the scenario's sentence on this
#: machine, verbatim. The repair exists for this string, not for a category.
HEARD = "Open MyHermes project in VS Code"


def test_the_transcript_this_machine_produces_reaches_the_real_name():
    candidates = api.parse_project_candidates(HEARD)

    assert "hermes" in candidates
    assert "hermes project" in candidates


def test_the_words_as_transcribed_are_searched_for_first():
    """The repair adds a reading; it never replaces one. A folder genuinely called
    ``myhermes`` is looked for before the peeled-back spelling is tried, so being
    wrong costs one lookup that finds nothing."""
    candidates = api.parse_project_candidates(HEARD)

    assert candidates[0] == "myhermes project"
    assert candidates.index("myhermes") < candidates.index("hermes project")


@pytest.mark.parametrize("text, unwanted", [
    # "the" is not a possessive, and no recogniser has been seen to weld one on.
    # Splitting it would invent a "me park" project for every theme park.
    ("open the theme park folder", "me park"),
    # "my" plus two letters is a fragment, not a folder name.
    ("open myx project in vscode", "x"),
])
def test_what_is_not_a_welded_possessive_is_left_alone(text: str, unwanted: str):
    assert unwanted not in api.parse_project_candidates(text)


def test_a_name_beginning_with_a_possessive_keeps_its_literal_readings_first():
    """``mysql notes`` is the adversarial case: a real name whose first two letters
    spell a possessive. Every literal reading is offered before any split one, so
    the folder that exists is found and the split is never reached."""
    candidates = api.parse_project_candidates("open mysql notes folder")
    literal = [word for word in candidates if word.startswith("mysql")]
    split = [word for word in candidates if word.startswith("sql")]

    assert literal and split
    assert (max(candidates.index(word) for word in literal)
            < min(candidates.index(word) for word in split))


# ----------------------------------------------------------------------
# 3. A remembered folder is a candidate, not a fact
# ----------------------------------------------------------------------

#: The scenario's sentence, as a person would type or say it.
REQUEST = "open my hermes project in vs code"


def test_one_live_folder_resolves_to_a_runnable_task(one_project):
    memory, project = one_project

    resolved = api.resolve_project_request(REQUEST, memory=memory)

    assert resolved is not None
    assert resolved.runnable and not resolved.ambiguous
    assert resolved.task_id == TASK
    assert Path(resolved.params["path"]) == project


def test_two_folders_of_the_same_name_produce_a_question(two_projects):
    """The task is known; only the path is not. Guessing between them would be a
    coin toss dressed up as an answer."""
    memory, first, second = two_projects

    resolved = api.resolve_project_request(REQUEST, memory=memory)

    assert resolved.ambiguous and not resolved.runnable
    assert resolved.task_id == TASK
    assert {Path(choice.path) for choice in resolved.choices} == {first, second}
    assert all(choice.label for choice in resolved.choices)


def test_a_remembered_folder_that_is_gone_is_reported_as_out_of_date(tmp_path: Path):
    """The index is a cache written at refresh time. Offering a folder that has
    since moved would present that cache as a fact.

    The reading with the evidence is the one reported, and it is not the first one
    tried: the sentence is parsed longest-first, so ``hermes project`` is looked up
    before ``hermes``, and only the latter ever matched the folder that moved.
    Naming the former here would say "no folder named hermes project is known" --
    true, useless, and pointing at a naming problem instead of a stale index.
    """
    memory = indexed([tmp_path / "projects" / "hermes"], store=tmp_path / "idx.json")

    resolved = api.resolve_project_request(REQUEST, memory=memory)

    assert not resolved.runnable and not resolved.ambiguous
    assert resolved.query == "hermes"
    assert "I remembered 1 location(s) for a hermes folder" in resolved.detail
    assert "index is out of date" in resolved.detail
    assert "memory --refresh" in resolved.detail


def test_a_file_with_the_project_name_is_not_the_project(tmp_path: Path):
    """``hermes`` with no extension is a plausible filename. An editor pointed at
    it would open a text buffer, not a project, and ``verify_dir`` would fail."""
    hermes = tmp_path / "hermes"
    hermes.write_text("not a folder\n", encoding="utf-8")
    memory = indexed([], [hermes], store=tmp_path / "idx.json")

    resolved = api.resolve_project_request(REQUEST, memory=memory)

    assert not resolved.runnable and not resolved.ambiguous
    assert "no folder named hermes" in resolved.detail


def test_a_name_that_is_nowhere_is_reported_in_its_plainest_reading(tmp_path: Path):
    """Every spelling was tried, so which one the user hears back is a presentation
    choice: "no folder named hermes" is recognisable where "no folder named hermes
    project" sounds like an answer to a different question."""
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    memory = indexed([unrelated], store=tmp_path / "idx.json")

    resolved = api.resolve_project_request(REQUEST, memory=memory)

    assert resolved.query == "hermes"
    assert "no folder named hermes is in the location index" in resolved.detail
    assert "hermes project" not in resolved.detail


def test_no_index_at_all_says_so_and_says_how_to_build_one(tmp_path: Path):
    """An empty index is not a negative answer, it is no answer. Reporting it as
    "not found" would blame the user's disk for the assistant's own setup."""
    resolved = api.resolve_project_request(
        REQUEST, memory=indexed([], store=tmp_path / "idx.json"),
    )

    assert not resolved.runnable
    assert "no location index yet" in resolved.detail
    assert "memory --refresh" in resolved.detail


def test_memory_switched_off_is_reported_as_such(one_project):
    """Not the same sentence as "it is not there". The folder may well exist."""
    memory, _ = one_project

    resolved = api.resolve_project_request(REQUEST, use_memory=False, memory=memory)

    assert not resolved.runnable
    assert "file memory is off" in resolved.detail


def test_a_sentence_that_is_not_a_project_request_resolves_to_nothing(one_project):
    """``None`` and "I could not find it" are different answers, and only the
    resolver can tell them apart. Everything downstream depends on the difference."""
    memory, _ = one_project

    assert api.resolve_project_request("what time is it", memory=memory) is None


# ----------------------------------------------------------------------
# 4. The read grant is the project's own subtree
# ----------------------------------------------------------------------

def granted(project: Path, tmp_path: Path) -> Policy:
    """A policy holding exactly what a real run of this task would be granted."""
    return Policy(
        workspace=tmp_path / "ws",
        readable_roots=api.readable_roots_for_task(TASK, {"path": str(project)}),
    )


def test_the_grant_is_the_one_project_directory(one_project):
    _, project = one_project

    assert (api.readable_roots_for_task(TASK, {"path": str(project)})
            == (project.resolve(),))


def test_without_a_path_nothing_is_granted():
    """The refusal comes first, so there is no path to grant and no default to
    fall back on. A task that cannot say what to open reads nothing."""
    assert api.readable_roots_for_task(TASK, {}) == ()
    assert api.readable_roots_for_task(TASK) == ()


def test_the_subtree_is_readable_and_everything_around_it_is_not(one_project,
                                                                tmp_path: Path):
    """Wide enough for an editor that reads the tree and for ``verify_dir`` to list
    it; narrow enough that a resolution which returned the wrong folder cannot
    become a way to read the drive."""
    _, project = one_project
    policy = granted(project, tmp_path)

    assert policy.resolve_read_path(str(project)) == project.resolve()
    assert (policy.resolve_read_path(str(project / "tests"))
            == (project / "tests").resolve())

    for denied in (project.parent, project.parent / "elsewhere", tmp_path):
        with pytest.raises(PolicyDenied):
            policy.resolve_read_path(str(denied))


# ----------------------------------------------------------------------
# 5. "Opened" has to mean three separate things
# ----------------------------------------------------------------------

def processes(*pids: int, ok: bool = True, error: str = "") -> Observation:
    """A fabricated ``process_state``. See the module docstring for why."""
    return Observation(
        source=Source.PROCESS,
        query="name_in=['Code.exe', 'code']",
        ok=ok,
        value=None if not ok else {
            "running": bool(pids), "count": len(pids), "pids": list(pids),
        },
        error=error or None,
    )


def windows(*found: dict, ok: bool = True, error: str = "") -> Observation:
    return Observation(
        source=Source.WINDOW,
        query="title_contains",
        ok=ok,
        value=None if not ok else {
            "backend": "test", "present": bool(found), "count": len(found),
            "windows": list(found),
        },
        error=error or None,
    )


def desktop(*titles: str, visible: bool = True):
    """A ``window_state`` stand-in that really filters on the needle it is given.

    Both window checks go through one function with different needles -- one looks
    for the editor ("Visual Studio Code"), the other for this project ("hermes") --
    so a stand-in that ignored the needle could not tell them apart, and the
    distinction between "VS Code is running" and "VS Code is showing what was
    asked for" is the whole point of having two.
    """
    def state(*, title_contains: str = "", **_: object) -> Observation:
        return windows(*[
            {"title": title, "visible": visible, "pid": 4242}
            for title in titles
            if title_contains.lower() in title.lower()
        ])

    return state


def verdicts(result) -> dict[str, Verdict]:
    return {check.name: check.verdict for check in result.checks}


def test_a_real_directory_passes(one_project, tmp_path: Path):
    _, project = one_project

    result = verifiers.verify_dir(granted(project, tmp_path), str(project))

    assert verdicts(result)["dir_exists"] is Verdict.PASS


def test_a_path_that_is_a_file_fails_the_directory_check(tmp_path: Path):
    """The stale-index case, and the reason this check runs at all: a FAIL here
    means nothing should have been launched."""
    hermes = tmp_path / "hermes"
    hermes.write_text("not a folder\n", encoding="utf-8")

    result = verifiers.verify_dir(granted(hermes, tmp_path), str(hermes))

    assert verdicts(result)["dir_exists"] is Verdict.FAIL


def test_an_editor_that_is_not_running_fails_rather_than_being_unknown(
        one_project, tmp_path: Path, monkeypatch):
    """Enumerating processes worked and found none. That is knowledge, not absence
    of it."""
    _, project = one_project
    monkeypatch.setattr(verifiers.observe, "process_state", lambda **kw: processes())
    monkeypatch.setattr(verifiers.observe, "window_state", desktop())

    result = verifiers.verify_app_running(granted(project, tmp_path), "vscode")

    assert verdicts(result)["app_process"] is Verdict.FAIL


def test_processes_that_cannot_be_enumerated_are_unknown(one_project, tmp_path: Path,
                                                        monkeypatch):
    _, project = one_project
    monkeypatch.setattr(
        verifiers.observe, "process_state",
        lambda **kw: processes(ok=False, error="psutil unavailable"))
    monkeypatch.setattr(verifiers.observe, "window_state", desktop())

    result = verifiers.verify_app_running(granted(project, tmp_path), "vscode")

    assert verdicts(result)["app_process"] is Verdict.UNKNOWN


def test_an_unavailable_window_backend_is_unknown_not_passing(one_project,
                                                             tmp_path: Path,
                                                             monkeypatch):
    """A headless host, or Linux without a window backend. The capability is
    missing, so nothing is claimed either way -- and UNKNOWN does not aggregate to
    success."""
    _, project = one_project
    monkeypatch.setattr(verifiers.observe, "process_state",
                        lambda **kw: processes(4242))
    monkeypatch.setattr(
        verifiers.observe, "window_state",
        lambda **kw: windows(ok=False, error="no window backend"))

    policy = granted(project, tmp_path)

    assert verdicts(verifiers.verify_app_running(policy, "vscode"))["app_window"] \
        is Verdict.UNKNOWN
    assert verdicts(verifiers.verify_opened(policy, str(project),
                                           needle=project.name))["viewer_window"] \
        is Verdict.UNKNOWN


def test_a_visible_window_naming_this_project_is_what_proves_it_opened(
        one_project, tmp_path: Path, monkeypatch):
    _, project = one_project
    monkeypatch.setattr(verifiers.observe, "window_state",
                        desktop("Welcome - hermes - Visual Studio Code"))

    result = verifiers.verify_opened(granted(project, tmp_path), str(project),
                                    needle=project.name)

    assert verdicts(result)["viewer_window"] is Verdict.PASS


def test_an_invisible_window_naming_the_project_is_not_evidence(
        one_project, tmp_path: Path, monkeypatch):
    """The Win32 backend reports every top-level window and most are hidden
    helpers. A substring hit on one of those is a pass with nothing behind it."""
    _, project = one_project
    monkeypatch.setattr(
        verifiers.observe, "window_state",
        desktop("hermes DDE Server Window", visible=False))

    result = verifiers.verify_opened(granted(project, tmp_path), str(project),
                                     needle=project.name)

    assert verdicts(result)["viewer_window"] is Verdict.UNKNOWN


def test_an_editor_showing_a_different_project_is_not_this_project_opened(
        one_project, tmp_path: Path, monkeypatch):
    """The distinction the third verifier exists for. VS Code is up and titled like
    VS Code -- it may have been open all morning -- and nothing on screen names
    hermes, so ``app_window`` passes and ``viewer_window`` does not."""
    _, project = one_project
    monkeypatch.setattr(verifiers.observe, "process_state",
                        lambda **kw: processes(4242))
    monkeypatch.setattr(verifiers.observe, "window_state",
                        desktop("chess-ai - Visual Studio Code"))

    policy = granted(project, tmp_path)
    running = verdicts(verifiers.verify_app_running(policy, "vscode"))
    opened = verdicts(verifiers.verify_opened(policy, str(project),
                                             needle=project.name))

    assert running == {"app_process": Verdict.PASS, "app_window": Verdict.PASS}
    assert opened["viewer_window"] is Verdict.UNKNOWN


def test_the_whole_folder_name_is_what_is_looked_for_not_its_stem(tmp_path: Path,
                                                                monkeypatch):
    """A directory called ``my.project`` has the stem ``my``, which would match
    almost any window on the desktop. The task passes the whole name for this
    reason, and the default would quietly pass on a coincidence."""
    project = tmp_path / "my.project"
    project.mkdir()
    asked: list[str] = []

    def state(*, title_contains: str = "", **_: object) -> Observation:
        asked.append(title_contains)
        return windows()

    monkeypatch.setattr(verifiers.observe, "window_state", state)
    task = OpenProjectInEditor(path=str(project))

    verifiers.verify_opened(granted(project, tmp_path), str(project),
                            needle=task.target().name)

    assert asked[0] == "my.project"


def test_all_four_checks_are_reported_once_each_and_unprefixed(one_project,
                                                              tmp_path: Path,
                                                              monkeypatch):
    """The names reach the user's report, so they are not namespaced: prefixing
    them would put an absolute path into every line a person reads. Four checks,
    each answering a different question."""
    _, project = one_project
    monkeypatch.setattr(verifiers.observe, "process_state",
                        lambda **kw: processes(4242))
    monkeypatch.setattr(verifiers.observe, "window_state",
                        desktop("Welcome - hermes - Visual Studio Code"))

    result = OpenProjectInEditor(path=str(project)).verify_final(
        granted(project, tmp_path))

    assert [check.name for check in result.checks] == [
        "dir_exists", "app_process", "app_window", "viewer_window",
    ]
    assert set(verdicts(result).values()) == {Verdict.PASS}
    assert result.label == "project:hermes"


def test_a_stale_path_fails_the_directory_check_even_when_the_editor_is_up(
        tmp_path: Path, monkeypatch):
    """The false-success shape this task is built to refuse. A window titled after
    the project is on screen -- because the editor had it open from before -- and
    the folder is not there any more. Three PASSes do not outvote the FAIL."""
    project = tmp_path / "projects" / "hermes"
    monkeypatch.setattr(verifiers.observe, "process_state",
                        lambda **kw: processes(4242))
    monkeypatch.setattr(verifiers.observe, "window_state",
                        desktop("hermes - Visual Studio Code"))

    result = OpenProjectInEditor(path=str(project)).verify_final(
        Policy(workspace=tmp_path / "ws", readable_roots=(project,)))

    assert verdicts(result)["dir_exists"] is Verdict.FAIL
    assert result.verdict is not Verdict.PASS


# ----------------------------------------------------------------------
# 6. One action, through the existing action layer
# ----------------------------------------------------------------------

def test_the_reference_plan_is_one_launch_of_one_app_at_one_path(one_project,
                                                               tmp_path: Path):
    _, project = one_project

    plan = OpenProjectInEditor(path=str(project)).reference_plan(
        granted(project, tmp_path))

    assert len(plan) == 1
    action = plan[0]
    assert isinstance(action, Action)
    assert action.kind == "launch_app"
    assert action.params["app"] == "vscode"
    assert action.params["open_path"] == str(project)


def test_the_plan_waits_long_enough_for_a_cold_start_to_name_its_window(one_project,
                                                                      tmp_path: Path):
    """Verification reads a window *title*, and a cold VS Code draws its window
    before it finishes naming the folder in it. Settling too early turns a success
    into an UNKNOWN."""
    _, project = one_project

    plan = OpenProjectInEditor(path=str(project)).reference_plan(
        granted(project, tmp_path))

    assert plan[0].params["settle_s"] >= 8.0


def test_there_is_no_search_inside_the_task(one_project, tmp_path: Path):
    """Turning a name into a path is request resolution. Repeating it here would
    put a second, unpolicied search inside the task."""
    _, project = one_project

    plan = OpenProjectInEditor(path=str(project)).reference_plan(
        granted(project, tmp_path))

    assert {action.kind for action in plan} == {"launch_app"}


def test_the_editor_is_not_a_parameter(one_project):
    """The task id names the application. One that could be pointed elsewhere would
    make every trace line and every spoken sentence about it wrong."""
    _, project = one_project

    assert OpenProjectInEditor(path=str(project)).app == "vscode"


def test_the_task_observes_the_project_and_the_editor_it_may_already_have(
        one_project, tmp_path: Path):
    """``app_process`` cannot distinguish a process this run started from one that
    was already running, so the prior state is read before anything is launched --
    and the planner can see it too.

    Nothing is faked here: both observations are real reads of this machine, and
    the claim is about their shape rather than their contents.
    """
    _, project = one_project

    seen = OpenProjectInEditor(path=str(project)).observe(granted(project, tmp_path))

    assert set(seen) == {"project_dir", "editor_before"}
    assert seen["project_dir"].source is Source.FILESYSTEM
    assert seen["editor_before"].source is Source.PROCESS
    assert seen["project_dir"].value["is_dir"] is True



def test_no_path_means_no_run_at_all():
    """Not a planner call with nothing to open, and not a search: the refusal comes
    before the pipeline reaches either. The sentence it returns names the phrasing
    that *would* work."""
    result = api.run_agent_task(
        "open a project in vscode", task_id=TASK, task_params={},
    )

    assert not result.ok
    assert result.status is api.TaskStatus.UNSUPPORTED
    assert "needs a project folder" in result.detail
    assert result.steps_used == 0


def test_an_empty_task_is_constructible_but_says_it_has_nothing_to_open():
    """``main.py tasks`` builds one specimen of every registered class to read its
    id and goal, so an empty path must not raise -- it must only be unrunnable."""
    specimen = OpenProjectInEditor()

    assert specimen.task_id == TASK
    assert specimen.path == ""
    assert "remembered location" in specimen.goal


# ----------------------------------------------------------------------
# 7. Two folders of one name: the question crosses a turn, the request does not
# ----------------------------------------------------------------------

@pytest.fixture
def printed() -> list[str]:
    return []


@pytest.fixture
def executions(monkeypatch) -> list[dict]:
    """Record every call to the one execution entry point, and execute nothing."""
    seen: list[dict] = []

    def fake(request, **kwargs):
        seen.append({"request": request, **kwargs})
        return AgentResult(
            request=request, task_id=kwargs.get("task_id") or "",
            status=api.TaskStatus.SUCCESS, verified="PASS",
            completed=["dir_exists", "app_process", "app_window", "viewer_window"],
            detail="verified 4 check(s)",
        )

    monkeypatch.setattr(session_module.api, "run_agent_task", fake)
    return seen


@pytest.fixture
def chat(monkeypatch, two_projects, printed: list[str]) -> Session:
    """A session whose project lookups hit the two-place index and nothing else.

    The resolver is real here -- ``resolve_project_request`` runs, ``recall`` runs,
    the disk re-check runs -- because what is under test is the join between real
    resolution and the pending question. Only the execution entry point and the
    speech backend are stood in for.
    """
    memory, _, _ = two_projects
    monkeypatch.setattr(memory_module, "shared", lambda: memory)
    return Session(narrator=Narrator(speaker=None, write=printed.append),
                   planner="mock")


def test_two_projects_of_one_name_are_asked_about_and_nothing_runs(
        chat: Session, two_projects, executions: list[dict], printed: list[str]):
    """Asking is not running, and the turn must not read as an attempt."""
    _, first, second = two_projects

    turn = chat.submit(REQUEST)

    assert executions == []
    assert turn.executed is False and turn.result is None
    assert chat.pending is not None and chat.pending.task_id == TASK
    assert any(str(first) in line for line in printed)
    assert any(str(second) in line for line in printed)


def test_the_question_is_speakable_and_the_paths_are_only_printed(chat: Session,
                                                                two_projects):
    """A path read aloud is noise. It is printed beside the sentence instead."""
    _, first, second = two_projects

    turn = chat.submit(REQUEST)

    assert str(first) not in turn.reply and str(second) not in turn.reply
    assert turn.reply in chat.narrator.spoken


def test_naming_a_folder_resumes_the_original_request(chat: Session, two_projects,
                                                     executions: list[dict]):
    """The requirement, in one test: the answer is a phrase naming a place, the
    request is never repeated, and what runs is the project task against the path
    that phrase chose."""
    _, _, second = two_projects
    chat.submit(REQUEST)

    turn = chat.submit("the one in work")

    assert len(executions) == 1
    assert executions[0]["task_id"] == TASK
    assert executions[0]["task_params"] == {"path": str(second)}
    assert turn.executed is True
    assert chat.pending is None


def test_a_spoken_number_answers_the_question(chat: Session,
                                             executions: list[dict]):
    """Speech-to-text writes numbers as words, and the voice path has to be able to
    answer its own question.

    Position, not spelling: "two" means whichever folder was offered second, and
    which one that is depends on how recall ranked them. Reading the offer back is
    the only honest way to assert this -- hard-coding a side would be asserting the
    ranking, which is not what a number means.
    """
    chat.submit(REQUEST)
    offered_second = chat.pending.choices[1].path

    chat.submit("two", source="voice")

    assert executions[0]["task_params"] == {"path": offered_second}


def test_an_answer_can_only_name_a_place_that_was_offered(chat: Session,
                                                        executions: list[dict]):
    """A pending question is not a general prompt. A third location names none of
    the options, so nothing runs and the question is put once more."""
    chat.submit(REQUEST)

    turn = chat.submit("the one in my documents")

    assert executions == []
    assert turn.executed is False
    assert "did not name one of them" in turn.reply
    assert chat.pending is not None


def test_repeating_the_whole_request_is_read_as_a_fresh_one(chat: Session,
                                                          executions: list[dict]):
    """Saying it again is not answering. Both places still match, so the honest
    outcome is the same question -- not the first option, chosen for them."""
    chat.submit(REQUEST)

    turn = chat.submit(REQUEST)

    assert executions == []
    assert turn.executed is False
    assert chat.pending is not None


def test_answering_still_reaches_the_one_execution_entry_point(chat: Session,
                                                             two_projects,
                                                             executions: list[dict]):
    """The negative claim the whole vertical slice rests on: an answer to a
    clarification is not a second road into execution. It arrives at the same
    function, with the request line it was typed on."""
    chat.submit(REQUEST)
    chat.submit("2")

    assert executions[0]["request"] == "2"
    assert executions[0]["planner"] == "mock"
    assert set(executions[0]) >= {"request", "task_id", "task_params", "planner",
                                  "max_steps", "use_memory"}

