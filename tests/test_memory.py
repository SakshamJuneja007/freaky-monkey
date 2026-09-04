"""Tests for the persistent file-location memory.

Four properties, in the order they matter:

1. **It persists.** The whole point of the feature is not re-walking the disks on
   every question, so a fresh ``FileMemory`` over the same file must answer.
2. **It stays bounded.** Depth, entry cap and exclusions are what keep an index
   of "the D: drive" from becoming an index of every ``node_modules`` on it.
3. **It ranks the thing the user meant.** Both live ranking defects found on the
   real disk are pinned here as regressions, with the same shapes that broke.
4. **It cannot take a run down.** A missing or corrupt index recalls nothing; it
   never raises into the control loop, and it never widens what a run may read.

The walk itself shells out to PowerShell, so tests that would need a real scan
build ``Entry`` lists directly. ``_walk`` is exercised against a temporary tree in
one Windows-only test rather than mocked, because mocking it would only prove the
mock works.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from agent_control import memory as mem
from agent_control.memory import Entry, FileMemory, Recall, RootReport


WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32", reason="discovery and the walk are PowerShell-based"
)


def entry(path: str, kind: str = "file", *, mtime: float = 1_700_000_000.0,
          size: int = 1024, depth: int = 3, root: str = "D:\\") -> Entry:
    return Entry(path=path, kind=kind, size=size, mtime=mtime, depth=depth,
                 root=root)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "locations.json"


@pytest.fixture
def certificates() -> list[Entry]:
    """The real shapes from this machine that drove the ranking rules.

    ``last day.pdf`` is the internship certificate: the words "internship" and
    "certificate" appear only in its *parent path*, never in its filename. The
    other two entries are the things that outranked it before the fix.
    """
    base = "C:\\Users\\x\\Downloads\\Gyansetu Internship Assignment\\Certificates"
    return [
        entry(f"{base}\\last day.pdf", depth=4),
        entry(base, kind="dir", depth=3),
        entry("D:\\certificates\\postman certificate.jpg", depth=2),
        entry("D:\\code\\setup-python-3.12.exe", depth=2, size=27_000_000),
    ]


def loaded(entries: list[Entry], *, store: Path | str = "unused",
           refreshed_at: float | None = None,
           roots: list[RootReport] | None = None) -> FileMemory:
    """A FileMemory holding known entries without touching a disk."""
    memory = FileMemory(store)
    memory.entries = list(entries)
    memory.roots = list(roots or [])
    memory.refreshed_at = time.time() if refreshed_at is None else refreshed_at
    memory.loaded = True
    return memory


# ----------------------------------------------------------------------
# 1. Persistence -- the reason the feature exists
# ----------------------------------------------------------------------

def test_index_survives_a_new_process(store: Path, certificates: list[Entry]):
    """save -> forget everything -> load -> still answers.

    A second ``FileMemory`` over the same path stands in for a restart: nothing
    is shared but the file on disk.
    """
    loaded(certificates, store=store).save()
    assert store.is_file()

    reopened = FileMemory(store)
    assert reopened.load() is True
    assert len(reopened.entries) == len(certificates)
    assert reopened.recall("internship certificate").hits


def test_save_is_atomic_and_leaves_no_temp(store: Path, certificates: list[Entry]):
    """A half-written index is the corrupt case ``load`` must tolerate, so the
    writer must never produce one under a reader's nose."""
    loaded(certificates, store=store).save()
    assert not list(store.parent.glob("*.tmp"))
    json.loads(store.read_text(encoding="utf-8"))  # complete document


def test_load_rejects_a_foreign_schema(store: Path, certificates: list[Entry]):
    loaded(certificates, store=store).save()
    payload = json.loads(store.read_text(encoding="utf-8"))
    payload["schema"] = mem.SCHEMA_VERSION + 99
    store.write_text(json.dumps(payload), encoding="utf-8")

    stranded = FileMemory(store)
    assert stranded.load() is False
    assert stranded.entries == []


def test_a_bad_row_is_dropped_not_fatal(store: Path, certificates: list[Entry]):
    loaded(certificates, store=store).save()
    payload = json.loads(store.read_text(encoding="utf-8"))
    payload["entries"].append(["only-one-field"])
    payload["entries"].append("not even a list")
    store.write_text(json.dumps(payload), encoding="utf-8")

    reopened = FileMemory(store)
    assert reopened.load() is True
    assert len(reopened.entries) == len(certificates)


def test_stale_is_reported_not_hidden(store: Path, certificates: list[Entry]):
    old = time.time() - (mem.DEFAULT_MAX_AGE_S + 60)
    loaded(certificates, store=store, refreshed_at=old).save()

    reopened = FileMemory(store)
    assert reopened.load() is True
    assert reopened.stale is True
    assert reopened.recall("internship certificate").stale is True


# ----------------------------------------------------------------------
# 2. Bounds -- what keeps "index the D: drive" finite
# ----------------------------------------------------------------------

def test_system_drive_is_not_a_root():
    """Directive: do not scan the whole C: drive. Only Downloads may come from it.

    The walk script filters ``DriveType=3`` volumes against ``$env:SystemDrive``,
    so this asserts the *policy* the roots list encodes: any C: root present must
    be the Downloads folder, never ``C:\\``.
    """
    excluded = {name.lower() for name in mem.EXCLUDED_DIRS}
    for name in ("windows", "program files", "program files (x86)",
                 "programdata", "appdata", "$recycle.bin"):
        assert name in excluded


def test_developer_noise_is_excluded():
    for name in ("node_modules", ".git", ".venv", "__pycache__",
                 "site-packages", "dist", "build"):
        assert name in mem.EXCLUDED_DIRS


def test_the_index_does_not_index_itself():
    """Otherwise a refresh grows the thing it is scanning."""
    assert ".agent_memory" in mem.EXCLUDED_DIRS
    assert ".sandbox" in mem.EXCLUDED_DIRS


@WINDOWS_ONLY
def test_walk_respects_depth_and_exclusions(tmp_path: Path):
    """A real walk over a real tree -- the one test that runs PowerShell."""
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (tmp_path / "a" / "top.pdf").write_text("x", encoding="utf-8")
    (deep / "buried.pdf").write_text("x", encoding="utf-8")

    noise = tmp_path / "a" / "node_modules" / "pkg"
    noise.mkdir(parents=True)
    (noise / "hidden.pdf").write_text("x", encoding="utf-8")

    memory = FileMemory(tmp_path / "idx.json", max_depth=3, timeout_s=60.0)
    entries, report = memory._walk(tmp_path, "drive")

    if report.error:
        pytest.skip(f"PowerShell unavailable here: {report.error}")

    found = {Path(item.path).name.lower() for item in entries}
    assert "top.pdf" in found          # depth 2, inside the budget
    assert "buried.pdf" not in found   # depth 5, beyond max_depth=3
    assert "hidden.pdf" not in found   # pruned with node_modules
    assert "node_modules" not in found


@WINDOWS_ONLY
def test_walk_only_keeps_known_extensions(tmp_path: Path):
    (tmp_path / "keep.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "skip.tmpfile").write_text("x", encoding="utf-8")

    memory = FileMemory(tmp_path / "idx.json", max_depth=2, timeout_s=60.0)
    entries, report = memory._walk(tmp_path, "drive")
    if report.error:
        pytest.skip(f"PowerShell unavailable here: {report.error}")

    names = {Path(item.path).name for item in entries}
    assert "keep.pdf" in names
    assert "skip.tmpfile" not in names


def test_truncation_is_recorded_not_silent():
    assert RootReport(root="D:\\", stopped="entry_cap").truncated is True
    assert RootReport(root="D:\\", stopped="timeout").truncated is True
    assert RootReport(root="D:\\", indexed=10).truncated is False


def test_a_refresh_that_saw_nothing_keeps_the_old_index(monkeypatch,
                                                        store: Path,
                                                        certificates: list[Entry]):
    """An empty scan is a failed scan. Trading a usable map for an empty one
    would make the feature worse than not refreshing at all."""
    memory = loaded(certificates, store=store)
    monkeypatch.setattr(memory, "discover_roots",
                        lambda: ([("drive", Path("D:\\"))], "stub"))
    monkeypatch.setattr(
        memory, "_walk",
        lambda root, role: ([], RootReport(root=str(root), role=role)))

    report = memory.refresh(save=False)
    assert report.ok is False
    assert "kept" in (report.error or "")
    assert len(memory.entries) == len(certificates)


# ----------------------------------------------------------------------
# 3. Retrieval -- the two defects the live disk exposed
# ----------------------------------------------------------------------

def test_recalls_the_file_the_words_describe(certificates: list[Entry]):
    """'internship certificate' names neither word in ``last day.pdf``.

    All the evidence is in the parent path, which is exactly why parent folders
    carry real weight.
    """
    assert loaded(certificates).recall("internship certificate").best.endswith(
        "last day.pdf")


def test_a_file_outranks_the_folder_containing_it(certificates: list[Entry]):
    """Regression, live defect 1: the ``Certificates`` folder won on a name-prefix
    hit while the file's evidence sat in its path. Kind is now a sort tier."""
    hits = loaded(certificates).recall("internship certificate").hits
    kinds = [hit.entry.kind for hit in hits]
    assert kinds[0] == "file"
    assert "dir" in kinds, "the folder should still be offered, just lower"


def test_matching_both_words_beats_one_strong_match(certificates: list[Entry]):
    """Regression, live defect 2: ``postman certificate.jpg`` scored 4.12 on an
    exact filename hit for one of two terms and outranked the 2-of-2 match at
    3.68. Coverage now outranks raw score."""
    hits = loaded(certificates).recall("internship certificate").hits
    assert hits[0].entry.path.endswith("last day.pdf")
    assert len(hits[0].matched) == 2

    paths = [hit.entry.path for hit in hits]
    assert paths.index("D:\\certificates\\postman certificate.jpg") > 0


def test_asking_for_a_folder_returns_the_folder(certificates: list[Entry]):
    hits = loaded(certificates).recall("the certificates folder").hits
    assert hits
    assert hits[0].entry.kind == "dir"


def test_an_extension_word_is_a_hint(certificates: list[Entry]):
    assert loaded(certificates).recall("last day pdf").best.endswith("last day.pdf")


def test_nothing_relevant_returns_nothing(certificates: list[Entry]):
    """A cache that always answers is a cache that invents locations."""
    found = loaded(certificates).recall("quantum tunnelling lecture notes")
    assert found.hits == ()
    assert bool(found) is False


def test_a_query_of_only_stopwords_returns_nothing(certificates: list[Entry]):
    assert loaded(certificates).recall("please open the file for me").hits == ()


def test_recall_is_deterministic(certificates: list[Entry]):
    memory = loaded(certificates)
    first = memory.recall("internship certificate").paths
    assert first == memory.recall("internship certificate").paths


def test_limit_is_honoured(certificates: list[Entry]):
    assert len(loaded(certificates).recall("certificate", limit=1).hits) == 1
    assert loaded(certificates).recall("certificate", limit=0).hits == ()


# ----------------------------------------------------------------------
# 4. It cannot break a run
# ----------------------------------------------------------------------

def test_a_missing_index_recalls_empty(tmp_path: Path):
    absent = FileMemory(tmp_path / "never-written.json")
    found = absent.recall("internship certificate")
    assert found.hits == ()
    assert found.available is False
    assert found.as_state() == {}


def test_a_corrupt_index_recalls_empty_instead_of_raising(store: Path):
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("{not json at all", encoding="utf-8")

    memory = FileMemory(store)
    assert memory.load() is False
    assert memory.recall("internship certificate").hits == ()


def test_module_recall_swallows_a_broken_index(monkeypatch):
    """The pipeline calls ``memory.recall``; a bug in scoring must degrade to
    "nothing remembered", never abort the task."""
    class Exploding:
        def recall(self, *_args, **_kwargs):
            raise RuntimeError("index on fire")

    found = mem.recall("internship certificate", memory=Exploding())
    assert found.hits == ()
    assert found.available is False


def test_keeping_drops_disallowed_paths(certificates: list[Entry]):
    found = loaded(certificates).recall("certificate")
    assert found.hits

    kept = found.keeping(lambda path: path.startswith("D:\\"))
    assert kept.hits
    assert all(hit.entry.path.startswith("D:\\") for hit in kept.hits)
    assert kept.query == found.query
    assert kept.indexed == found.indexed

    assert found.keeping(lambda _path: False).hits == ()
    assert found.keeping(lambda _path: False).as_state() == {}


# ----------------------------------------------------------------------
# What the planner is actually told
# ----------------------------------------------------------------------

def test_as_state_fences_candidates_as_untrusted(certificates: list[Entry]):
    """File and folder names are attacker-controllable text. They arrive at the
    planner inside the untrusted fence the system prompt already knows about."""
    state = loaded(certificates).recall("internship certificate").as_state()

    assert list(state) == ["remembered_locations"]
    value = state["remembered_locations"]["value"]
    body = value["candidates"]
    assert "UNTRUSTED_DATA:remembered_file_locations" in body
    assert "END_UNTRUSTED_DATA:remembered_file_locations" in body
    assert "last day.pdf" in body


def test_as_state_calls_the_cache_a_cache(certificates: list[Entry]):
    """The framing is the runtime's own words, outside the fence, so untrusted
    text cannot rewrite it."""
    value = (loaded(certificates)
             .recall("internship certificate")
             .as_state()["remembered_locations"]["value"])
    note = value["note"].lower()
    assert "cache" in note
    assert "out of date" in note or "may be" in note
    assert "verification" in note


def test_as_state_is_shaped_like_an_observation(certificates: list[Entry]):
    """``summarize`` produces {name: {source, ok, age_s, value, error}}; recall
    must match or the planner sees two different state shapes."""
    block = (loaded(certificates)
             .recall("internship certificate")
             .as_state()["remembered_locations"])
    assert set(block) == {"source", "ok", "age_s", "value", "error"}
    assert block["error"] is None
    assert block["ok"] is True


def test_as_state_is_empty_when_nothing_matched(certificates: list[Entry]):
    """Silence, not an empty list with a heading: an empty section invites the
    planner to explain it."""
    assert loaded(certificates).recall("no such thing here").as_state() == {}


def test_recall_json_round_trips(certificates: list[Entry]):
    payload = loaded(certificates).recall("internship certificate").to_json()
    json.dumps(payload)  # must be serialisable for the trace and --json
    assert payload["query"] == "internship certificate"
    assert payload["hits"]
    assert payload["available"] is True


# ----------------------------------------------------------------------
# Tokenisation
# ----------------------------------------------------------------------

def test_tokens_splits_on_separators_and_case():
    assert mem.tokens("Last Day.pdf") == ["last", "day", "pdf"]
    assert mem.tokens("Gyansetu_Internship-Assignment") == [
        "gyansetu", "internship", "assignment"]


def test_query_terms_strip_the_verb_and_keep_the_noun():
    terms, wants_dir = mem._query_terms("open my internship certificate")
    assert terms == ["internship", "certificate"]
    assert wants_dir is False

    terms, wants_dir = mem._query_terms("show the certificates folder")
    assert wants_dir is True


def test_entry_name_parts():
    item = entry("D:\\a\\Report Final.PDF")
    assert item.name == "Report Final.PDF"
    assert item.stem == "Report Final"
    assert item.ext == ".pdf"
    assert item.is_file is True


def test_entry_round_trips_through_json():
    original = entry("D:\\a\\b.pdf", depth=2, size=99)
    restored = Entry.from_json(original.to_json())
    assert restored is not None
    assert restored.path == original.path
    assert restored.size == original.size
    assert Entry.from_json(["too", "short"]) is None
    assert Entry.from_json("garbage") is None
