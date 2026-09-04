"""Tests for opening a file the user named, from a remembered location.

The feature is one sentence -- *"open main1.mp4"*, and if the name is in two
places, ask which -- and five claims hold it up:

1. **The parse is lexical, not inference.** A closed verb set and exactly one
   allowlisted filename. Anything else resolves to ``None``, which routes to the
   existing refusal, so the interface never looks more general than the backend.
2. **The extension allowlist is the security boundary.** ``open_file`` hands the
   path to ``os.startfile``, which runs whatever the shell associates with the
   extension. ``.exe`` is refused by name, before any planner sees it.
3. **A remembered path is a candidate, not a fact.** Every hit is re-checked
   against the filesystem, so a stale index cannot produce a runnable request.
4. **The read grant is one file.** Not the folder, not the drive. A sibling in the
   same directory stays unreadable.
5. **"Opened" has to mean something.** The window check returns PASS only on a
   *visible* window, and UNKNOWN -- never PASS, never FAIL -- when it cannot tell.

The window tests fabricate observations. A test that needed a real player running
would prove nothing repeatable, and the property under test is how a *reading* is
turned into a verdict, which is exactly the part that does not need a real window.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent_control import api, verifiers
from agent_control.memory import Entry, FileMemory
from agent_control.policy import Policy
from agent_control.types import Observation, PolicyDenied, Source, Verdict
from benchmark.tasks import (
    BENCHMARK_IDS,
    INTERACTIVE_IDS,
    TASK_IDS,
    OpenNamedFile,
    build_task,
    build_tasks,
)

TASK = "open_named_file"


def indexed(paths: list[Path], *, store: Path) -> FileMemory:
    """A ``FileMemory`` holding exactly these locations, touching no disk scan."""
    memory = FileMemory(store)
    memory.entries = [
        Entry(path=str(path), kind="file",
              size=max(1, path.stat().st_size if path.exists() else 1024),
              mtime=time.time(), depth=len(path.parts), root=str(path.anchor))
        for path in paths
    ]
    memory.roots = []
    memory.refreshed_at = time.time()
    memory.loaded = True
    return memory


@pytest.fixture
def two_places(tmp_path: Path) -> tuple[FileMemory, Path, Path]:
    """The real shape of the request: one name, two real files, two folders."""
    downloads = tmp_path / "Downloads"
    assets = tmp_path / "site" / "src" / "assets"
    downloads.mkdir(parents=True)
    assets.mkdir(parents=True)

    first = downloads / "main1.mp4"
    second = assets / "main1.mp4"
    first.write_bytes(b"\x00" * 2048)
    second.write_bytes(b"\x00" * 4096)

    return indexed([first, second], store=tmp_path / "idx.json"), first, second


# ----------------------------------------------------------------------
# 1. The parse is lexical, not inference
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("open main1.mp4", "main1.mp4"),
    ("hey open the main1.mp4", "main1.mp4"),          # the user's own words
    ("Jarvis, could you please play main1.mp4?", "main1.mp4"),
    ("show notes.txt", "notes.txt"),
    ('open "Last Day.pdf"', "Last Day.pdf"),          # spaces need the quotes
    ("launch report.docx", "report.docx"),
])
def test_an_open_request_yields_its_filename(text: str, expected: str):
    assert api.parse_open_request(text) == expected


@pytest.mark.parametrize("text", [
    "open_last_day_pdf",              # a registered id, not an open-by-name
    "open last day pdf",              # ditto, spoken
    "open the folder",                # names no file
    "delete main1.mp4",               # verb outside the set
    "main1.mp4",                      # no verb at all
    "open a.pdf and b.pdf",           # two files; this task opens one
    "",
])
def test_anything_else_is_not_an_open_request(text: str):
    """``None`` here is what preserves the honest refusal for everything else."""
    assert api.parse_open_request(text) is None
    assert api.resolve_open_request(text) is None


def test_a_registered_task_id_is_left_to_resolve_task(two_places):
    """The two resolvers must not overlap: ``open_last_day_pdf`` is a task id and
    stays one, even though it starts with the word "open"."""
    memory, _, _ = two_places

    assert api.resolve_open_request("open_last_day_pdf", memory=memory) is None
    assert api.resolve_task("open_last_day_pdf") == "open_last_day_pdf"


# ----------------------------------------------------------------------
# 2. The extension allowlist is the security boundary
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "setup.exe", "install.msi", "script.bat", "run.cmd", "payload.ps1",
    "thing.scr", "lib.dll", "hook.vbs", "macro.js",
])
def test_executables_are_refused_by_name(name: str, tmp_path: Path):
    """Refused at the parse, so no policy check, planner or action ever sees it.

    ``open_file`` calls ``os.startfile``, which is "run whatever the shell
    associates with this extension". The allowlist is what keeps that from being a
    way to execute a program by asking politely.
    """
    landmine = tmp_path / name
    landmine.write_bytes(b"MZ")
    memory = indexed([landmine], store=tmp_path / "idx.json")

    assert api.parse_open_request(f"open {name}") is None
    assert api.resolve_open_request(f"open {name}", memory=memory) is None
    assert Path(name).suffix.lower() not in api.OPENABLE_SUFFIXES


def test_the_allowlist_covers_the_document_and_media_types():
    for suffix in (".pdf", ".mp4", ".mkv", ".mp3", ".png", ".jpg", ".txt",
                   ".md", ".csv", ".docx", ".xlsx", ".pptx", ".html", ".json",
                   ".log", ".zip"):
        assert suffix in api.OPENABLE_SUFFIXES


# ----------------------------------------------------------------------
# 3. A remembered path is a candidate, not a fact
# ----------------------------------------------------------------------

def test_one_location_resolves_to_a_runnable_request(tmp_path: Path):
    only = tmp_path / "Downloads" / "notes.txt"
    only.parent.mkdir(parents=True)
    only.write_text("hello", encoding="utf-8")

    resolved = api.resolve_open_request(
        "open notes.txt", memory=indexed([only], store=tmp_path / "idx.json"))

    assert resolved is not None
    assert resolved.runnable is True
    assert resolved.ambiguous is False
    assert resolved.task_id == TASK
    assert Path(resolved.params["path"]) == only


def test_a_near_miss_is_not_a_location_for_this_name(tmp_path: Path):
    """``recall`` is a ranked lookup: asked for ``main1.mp4`` on the real machine it
    also returns ``mainn.mp4.mp4`` and ``classroom.mp4.mp4``. Right for "find me
    something like this", wrong here -- a file not called what the user said is not
    the file they named, and offering it would make the question a false claim
    ("I know 5 places with that name" when the name exists in two)."""
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    exact = downloads / "main1.mp4"
    for name in ("main1.mp4", "mainn.mp4.mp4", "classroom.mp4.mp4", "main1.mkv"):
        (downloads / name).write_bytes(b"\x00" * 64)

    resolved = api.resolve_open_request(
        "open main1.mp4",
        memory=indexed(sorted(downloads.iterdir()), store=tmp_path / "idx.json"))

    assert resolved.runnable is True
    assert resolved.ambiguous is False
    assert Path(resolved.params["path"]) == exact


def test_only_near_misses_is_not_found_rather_than_a_wrong_file(tmp_path: Path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    (downloads / "mainn.mp4.mp4").write_bytes(b"\x00" * 64)

    resolved = api.resolve_open_request(
        "open main1.mp4",
        memory=indexed(sorted(downloads.iterdir()), store=tmp_path / "idx.json"))

    assert resolved.runnable is False
    assert resolved.choices == ()
    assert "is named main1.mp4" in resolved.detail


def test_the_name_match_ignores_case(tmp_path: Path):
    """Windows filenames are case-insensitive; requiring an exact case match would
    refuse a file the user can see in Explorer."""
    target = tmp_path / "Downloads" / "Main1.MP4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\x00" * 64)

    resolved = api.resolve_open_request(
        "open main1.mp4", memory=indexed([target], store=tmp_path / "idx.json"))

    assert resolved.runnable is True
    assert Path(resolved.params["path"]) == target


def test_two_locations_resolve_to_a_question_not_a_guess(two_places):
    """Both are offered, with a short distinguishing label each, and nothing is
    marked runnable -- picking the higher-scoring one would be a guess."""
    memory, first, second = two_places

    resolved = api.resolve_open_request("open main1.mp4", memory=memory)

    assert resolved is not None
    assert resolved.ambiguous is True
    assert resolved.runnable is False
    assert resolved.params == {}
    assert {Path(choice.path) for choice in resolved.choices} == {first, second}
    assert {choice.label for choice in resolved.choices} == {"Downloads", "assets"}


def test_labels_distinguish_and_never_repeat(two_places):
    memory, _, _ = two_places
    labels = [c.label for c in api.resolve_open_request(
        "open main1.mp4", memory=memory).choices]

    assert len(set(labels)) == len(labels)
    assert all(labels)  # a blank label would make the question unanswerable


def test_a_stale_index_cannot_produce_a_runnable_request(tmp_path: Path):
    """The index is a cache. A row whose file has been moved away is a stale fact,
    and offering it would be presenting the cache as the world."""
    moved = tmp_path / "Downloads" / "gone.mp4"
    moved.parent.mkdir(parents=True)
    memory = indexed([moved], store=tmp_path / "idx.json")  # never created

    resolved = api.resolve_open_request("open gone.mp4", memory=memory)

    assert resolved is not None
    assert resolved.runnable is False
    assert resolved.ambiguous is False
    assert "out of date" in resolved.detail
    assert "--refresh" in resolved.detail


def test_a_stale_row_is_dropped_from_a_shortlist(two_places):
    """One of two candidates gone leaves one candidate -- and therefore no
    question, because there is nothing left to choose between."""
    memory, first, second = two_places
    second.unlink()

    resolved = api.resolve_open_request("open main1.mp4", memory=memory)

    assert resolved.runnable is True
    assert Path(resolved.params["path"]) == first


def test_no_index_says_so_instead_of_saying_not_found(tmp_path: Path):
    """"I have no index" and "it is not on this machine" are different facts, and
    only one of them is fixed by running a refresh."""
    empty = FileMemory(tmp_path / "never-written.json")

    resolved = api.resolve_open_request("open main1.mp4", memory=empty)

    assert resolved is not None
    assert resolved.runnable is False
    assert "main.py memory --refresh" in resolved.detail


def test_memory_off_is_reported_rather_than_guessed(two_places):
    memory, _, _ = two_places
    resolved = api.resolve_open_request(
        "open main1.mp4", use_memory=False, memory=memory)

    assert resolved.runnable is False
    assert "memory is off" in resolved.detail


# ----------------------------------------------------------------------
# 4. The read grant is one file
# ----------------------------------------------------------------------

def test_the_readable_root_is_the_chosen_file_alone(tmp_path: Path):
    """The narrowest grant the policy layer can express: ``resolve_read_path``
    tests ``is_relative_to``, and a path is relative to itself."""
    target = tmp_path / "Downloads" / "main1.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\x00" * 16)

    roots = api.readable_roots_for_task(TASK, {"path": str(target)})

    assert roots == (target.resolve(),)


def test_a_sibling_file_stays_unreadable(tmp_path: Path):
    target = tmp_path / "Downloads" / "main1.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\x00" * 16)
    sibling = target.with_name("passwords.txt")
    sibling.write_text("secret", encoding="utf-8")

    policy = Policy(
        workspace=tmp_path / "ws",
        readable_roots=api.readable_roots_for_task(TASK, {"path": str(target)}),
    )

    assert policy.resolve_read_path(str(target)) == target.resolve()

    with pytest.raises(PolicyDenied):
        policy.resolve_read_path(str(sibling))

    with pytest.raises(PolicyDenied):
        policy.resolve_read_path(str(target.parent))


def test_no_path_grants_nothing(tmp_path: Path):
    assert api.readable_roots_for_task(TASK, {}) == ()
    assert api.readable_roots_for_task(TASK, None) == ()


def test_existing_grants_are_unchanged():
    """No task's read scope was widened to make this feature work."""
    assert api.readable_roots_for_task("open_last_day_pdf") == (
        Path.home() / "Downloads",
    )
    assert api.readable_roots_for_task("some_other_task", {"path": "D:\\x"}) == ()


# ----------------------------------------------------------------------
# 5. "Opened" has to mean something
# ----------------------------------------------------------------------

@pytest.fixture
def opened_file(tmp_path: Path) -> tuple[Policy, Path]:
    target = tmp_path / "main1.mp4"
    target.write_bytes(b"\x00" * 32)
    return Policy(workspace=tmp_path / "ws", readable_roots=(target,)), target


def windows(*found: dict, ok: bool = True, error: str = "") -> Observation:
    return Observation(
        source=Source.WINDOW,
        query="title_contains=main1",
        ok=ok,
        value=None if not ok else {
            "backend": "test", "present": bool(found), "count": len(found),
            "windows": list(found),
        },
        error=error or None,
    )


def only(name: str, verdict: Verdict, result) -> None:
    assert [check.name for check in result.checks] == [name]
    assert result.checks[0].verdict is verdict
    assert result.verdict is verdict


def test_a_visible_window_titled_like_the_file_passes(opened_file, monkeypatch):
    policy, target = opened_file
    monkeypatch.setattr(
        verifiers.observe, "window_state",
        lambda **kw: windows({"title": "main1.mp4 - VLC media player",
                              "visible": True, "pid": 42}))

    only("viewer_window", Verdict.PASS,
         verifiers.verify_opened(policy, str(target)))


def test_an_invisible_window_is_not_evidence(opened_file, monkeypatch):
    """Measured on this machine: the Win32 backend reports ~114 top-level windows,
    most of them hidden helpers (``DDE Server Window``). A substring hit on one of
    those is a pass with nothing behind it."""
    policy, target = opened_file
    monkeypatch.setattr(
        verifiers.observe, "window_state",
        lambda **kw: windows({"title": "main1.mp4 DDE Server Window",
                              "visible": False, "pid": 7}))

    only("viewer_window", Verdict.UNKNOWN,
         verifiers.verify_opened(policy, str(target)))


def test_no_matching_window_is_unknown_not_failed(opened_file, monkeypatch):
    """The honest verdict. A player that titles its window "Films & TV" really did
    open the file; a FAIL here would be a false negative, and a PASS a false
    positive. UNKNOWN aggregates to a non-PASS verdict, so nothing is claimed."""
    policy, target = opened_file
    monkeypatch.setattr(verifiers.observe, "window_state", lambda **kw: windows())

    result = verifiers.verify_opened(policy, str(target))
    only("viewer_window", Verdict.UNKNOWN, result)
    assert "either nothing opened" in result.checks[0].reason


def test_an_unmatched_window_check_reports_what_is_on_screen(opened_file,
                                                             monkeypatch):
    """Measured on this machine: Windows 11's default ``.mp4`` handler titles its
    window plain ``Media Player``, so this branch is the *normal* case for a
    successful open. Naming the visible windows is what lets a person tell "nothing
    opened" from "the handler does not name the file" -- and it is an observation,
    not an inference, so the verdict is untouched."""
    policy, target = opened_file
    seen: list[dict] = []

    def fake(**kw):
        seen.append(kw)
        if kw.get("title_contains"):
            return windows()                                   # no title match
        return windows({"title": "Media Player", "visible": True, "pid": 99},
                       {"title": "DDE Server Window", "visible": False})

    monkeypatch.setattr(verifiers.observe, "window_state", fake)

    result = verifiers.verify_opened(policy, str(target))

    only("viewer_window", Verdict.UNKNOWN, result)
    assert "either nothing opened" in result.checks[0].reason
    assert "Media Player" in result.checks[0].reason
    assert "DDE Server Window" not in result.checks[0].reason
    assert result.checks[0].evidence["visible_titles"] == ["Media Player"]
    assert [bool(kw.get("title_contains")) for kw in seen] == [True, False]


def test_a_matched_window_asks_nothing_further(opened_file, monkeypatch):
    """The context observation is for the UNKNOWN branch only; a PASS already
    knows what it is looking at."""
    policy, target = opened_file
    calls: list[dict] = []

    def fake(**kw):
        calls.append(kw)
        return windows({"title": "main1.mp4 - VLC media player", "visible": True})

    monkeypatch.setattr(verifiers.observe, "window_state", fake)

    only("viewer_window", Verdict.PASS, verifiers.verify_opened(policy, str(target)))
    assert len(calls) == 1


def test_an_unavailable_backend_is_unknown(opened_file, monkeypatch):
    policy, target = opened_file
    monkeypatch.setattr(
        verifiers.observe, "window_state",
        lambda **kw: windows(ok=False, error="no window backend"))

    result = verifiers.verify_opened(policy, str(target))
    only("viewer_window", Verdict.UNKNOWN, result)
    assert "no window backend" in result.checks[0].reason


def test_a_two_character_stem_cannot_identify_a_window(tmp_path: Path, monkeypatch):
    """``a.txt`` would match any title containing an "a"."""
    target = tmp_path / "ab.txt"
    target.write_text("x", encoding="utf-8")
    policy = Policy(workspace=tmp_path / "ws", readable_roots=(target,))

    called: list[bool] = []
    monkeypatch.setattr(verifiers.observe, "window_state",
                        lambda **kw: called.append(True) or windows())

    only("viewer_window", Verdict.UNKNOWN,
         verifiers.verify_opened(policy, str(target)))
    assert called == []  # refused before observing, not after


def test_the_task_verifies_content_and_presence_together(opened_file, monkeypatch):
    """Three checks, distinct names, no absolute path inside any of them -- the
    reason the two results are merged directly instead of through ``combine``."""
    policy, target = opened_file
    monkeypatch.setattr(
        verifiers.observe, "window_state",
        lambda **kw: windows({"title": "main1 - Films & TV", "visible": True}))

    result = OpenNamedFile(path=str(target)).verify_final(policy)

    assert [check.name for check in result.checks] == [
        "file_exists", "file_min_bytes", "viewer_window"]
    assert result.verdict is Verdict.PASS
    assert all(str(target) not in check.name for check in result.checks)


def test_an_empty_file_fails_even_with_a_window(opened_file, monkeypatch):
    """The classic false success: something is on screen, the file is a stub."""
    policy, target = opened_file
    target.write_bytes(b"")
    monkeypatch.setattr(
        verifiers.observe, "window_state",
        lambda **kw: windows({"title": "main1.mp4", "visible": True}))

    result = OpenNamedFile(path=str(target)).verify_final(policy)

    assert result.verdict is Verdict.FAIL
    assert [c.name for c in result.checks if c.verdict is Verdict.FAIL] == [
        "file_min_bytes"]


def test_checkpoint_verification_only_answers_for_the_open(opened_file):
    policy, target = opened_file
    task = OpenNamedFile(path=str(target))
    from agent_control.types import Action

    assert task.verify_checkpoint(policy, Action(kind="create_dir")) is None
    assert task.verify_checkpoint(policy, Action(kind="open_file")) is not None


# ----------------------------------------------------------------------
# The task, the registry, and the benchmark set
# ----------------------------------------------------------------------

def test_the_reference_plan_is_one_open_of_the_chosen_path(tmp_path: Path):
    target = tmp_path / "main1.mp4"
    target.write_bytes(b"\x00")
    policy = Policy(workspace=tmp_path / "ws", readable_roots=(target,))

    plan = OpenNamedFile(path=str(target)).reference_plan(policy)

    assert [action.kind for action in plan] == ["open_file"]
    assert plan[0].params["path"] == str(target)


def test_a_specimen_builds_without_a_path():
    """``main.py tasks`` builds one of every class to read its id and goal."""
    specimen = build_task(TASK)

    assert specimen.task_id == TASK
    assert specimen.goal
    assert specimen.path == ""


def test_build_task_passes_parameters_through(tmp_path: Path):
    target = tmp_path / "main1.mp4"
    built = build_task(TASK, path=str(target))

    assert built.path == str(target)
    assert str(target) in built.goal


def test_an_invented_parameter_is_a_named_error():
    with pytest.raises(TypeError):
        build_task(TASK, colour="red")


def test_an_unparameterized_run_is_refused_not_attempted():
    """The empty-path specimen is constructible but not runnable. Refusing here is
    what keeps it from reaching a planner with nothing to open."""
    result = api.run_agent_task("open something", task_id=TASK, planner="mock")

    assert result.status is api.TaskStatus.UNSUPPORTED
    assert "needs a filename" in result.detail
    assert result.checks == []


def test_an_unknown_task_id_is_refused():
    result = api.run_agent_task("x", task_id="no_such_task", planner="mock")
    assert result.status is api.TaskStatus.UNSUPPORTED


def test_the_interactive_task_is_not_in_the_benchmark_set():
    """``--tasks all`` expands to ``BENCHMARK_IDS``. A task that needs a filename
    cannot run unattended, and a trial row it produced would mean nothing."""
    assert TASK in TASK_IDS
    assert TASK in INTERACTIVE_IDS
    assert TASK not in BENCHMARK_IDS
    assert [task.task_id for task in build_tasks()] == list(BENCHMARK_IDS)


def test_the_harness_measures_the_benchmark_set_only():
    from benchmark import harness

    assert tuple(harness.TASK_IDS) == BENCHMARK_IDS
    assert TASK not in harness.TASK_IDS


def test_the_task_is_still_advertised_to_a_person():
    """It cannot be benchmarked and it can be asked for. Both are true."""
    assert TASK in api.registered_tasks()
    assert api.resolve_task(TASK) == TASK
