"""Observation / state acquisition layer (plan S5).

Every reader returns an :class:`Observation` stamped with source and time, so
freshness is checkable and stale-state failures are measurable after the fact.

The critical convention: a *successful reading of an absent thing* is
``ok=True, value={"exists": False}``. A *failed reading* is ``ok=False``. Only
the first may produce FAIL; the second must produce UNKNOWN.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import psutil

from .platform_window import get_backend
from .types import Observation, Source

MAX_HASH_BYTES = 8 * 1024 * 1024


def _observe(source: Source, query: str, fn) -> Observation:
    """Run a reader, converting exceptions into an honest ``ok=False``."""
    started = time.time()
    try:
        return Observation(source=source, query=query, value=fn(), observed_at=started)
    except Exception as exc:
        return Observation(
            source=source,
            query=query,
            value=None,
            observed_at=started,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


# -- filesystem ------------------------------------------------------------
def file_state(path: str | os.PathLike, *, want_hash: bool = True) -> Observation:
    target = Path(path)

    def read() -> dict[str, Any]:
        if not target.exists():
            return {"path": str(target), "exists": False}
        stat = target.stat()
        value: dict[str, Any] = {
            "path": str(target),
            "exists": True,
            "is_file": target.is_file(),
            "is_dir": target.is_dir(),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        }
        if want_hash and target.is_file() and stat.st_size <= MAX_HASH_BYTES:
            digest = hashlib.sha256()
            with open(target, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    digest.update(chunk)
            value["sha256"] = digest.hexdigest()
        return value

    return _observe(Source.FILESYSTEM, f"file_state({target})", read)


def dir_state(path: str | os.PathLike) -> Observation:
    target = Path(path)

    def read() -> dict[str, Any]:
        if not target.exists():
            return {"path": str(target), "exists": False}
        if not target.is_dir():
            return {"path": str(target), "exists": True, "is_dir": False}
        entries = sorted(p.name for p in target.iterdir())
        return {
            "path": str(target),
            "exists": True,
            "is_dir": True,
            "entry_count": len(entries),
            "entries": entries[:50],
        }

    return _observe(Source.FILESYSTEM, f"dir_state({target})", read)


# -- process ---------------------------------------------------------------
def process_state(*, name_contains: str | None = None, pid: int | None = None,
                  name_in: Iterable[str] | None = None) -> Observation:
    """Running processes matching a name fragment, an exact name, and/or a pid.

    ``name_in`` is exact basename matching and is what verifiers should use.
    Substring matching is too loose to verify with: a needle like ``"code"``
    matches unrelated executables and would inflate the success rate, which is
    the shallow-verification failure plan S22 calls a kill condition.
    """
    exact = {n.lower() for n in (name_in or ())}

    def read() -> dict[str, Any]:
        matches: list[dict[str, Any]] = []
        needle = (name_contains or "").lower()
        for proc in psutil.process_iter(["pid", "name", "exe", "cmdline", "create_time"]):
            try:
                info = proc.info
                if pid is not None and info["pid"] != pid:
                    continue
                if exact and (info.get("name") or "").lower() not in exact:
                    continue
                if needle:
                    haystack = " ".join(
                        filter(None, [info.get("name") or "", info.get("exe") or ""])
                    ).lower()
                    if needle not in haystack:
                        continue
                matches.append(
                    {
                        "pid": info["pid"],
                        "name": info.get("name"),
                        "exe": info.get("exe"),
                        "cmdline": (info.get("cmdline") or [])[:8],
                        "create_time": info.get("create_time"),
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return {
            "query": {"name_contains": name_contains, "pid": pid, "name_in": sorted(exact)},
            "running": bool(matches),
            "count": len(matches),
            "pids": [m["pid"] for m in matches][:20],
            "matches": matches[:10],
        }

    return _observe(
        Source.PROCESS, f"process_state(name={name_contains},pid={pid},in={sorted(exact)})", read
    )


# -- window / accessibility ------------------------------------------------
def window_state(*, title_contains: str | None = None, pid: int | None = None) -> Observation:
    """Window state via the platform seam.

    If the backend is unavailable the read *fails* (``ok=False``) rather than
    reporting "no windows", so dependent checks resolve to UNKNOWN.
    """
    backend = get_backend()

    def read() -> dict[str, Any]:
        if not backend.available:
            raise RuntimeError(f"window backend {backend.name!r} unavailable")
        needle = (title_contains or "").lower()
        windows = [
            w
            for w in backend.list_windows()
            if (not needle or needle in w.title.lower()) and (pid is None or w.pid == pid)
        ]
        return {
            "backend": backend.name,
            "query": {"title_contains": title_contains, "pid": pid},
            "present": bool(windows),
            "count": len(windows),
            "windows": [w.to_json() for w in windows[:10]],
        }

    return _observe(Source.WINDOW, f"window_state(title={title_contains},pid={pid})", read)


def accessibility_state(handle: int) -> Observation:
    """Accessibility tree for a window. Unimplemented backends -> ok=False."""
    backend = get_backend()

    def read() -> dict[str, Any]:
        tree = backend.accessibility_tree(handle)
        if tree is None:
            raise NotImplementedError(
                f"accessibility unavailable on backend {backend.name!r} (measured gap, plan S27)"
            )
        return tree

    return _observe(Source.ACCESSIBILITY, f"accessibility_state({handle})", read)


#: Titles Windows gives the "How do you want to open this file?" dialog. Matched
#: as lowercase substrings because the wording carries the file type on some
#: builds ("How do you want to open this .pdf file?").
CHOOSER_TITLES = (
    "how do you want to open",
    "open with",
)


def chooser_state() -> Observation:
    """Whether an application-chooser dialog is on screen -- and nothing more.

    Presence is observable: the chooser is an ordinary visible top-level window
    with a stable title, so the same backend that answers ``window_state`` can
    say it is there. Its *contents* are not observable. ``accessibility_tree``
    returns None on the Win32 backend, and no action kind in ``os_tools`` can
    click a list item, so the applications the dialog is offering can be neither
    read nor chosen from.

    ``options`` is therefore always ``None``, never ``[]``. The distinction is
    the whole point: an empty list would read as "the dialog offered nothing",
    which is a claim about what is on screen; ``None`` says the list could not
    be read. Callers turning this into a question for the user must leave
    ``Clarification.options`` empty and name the limit in ``unobservable``
    rather than invent plausible application names (plan S27).
    """
    backend = get_backend()

    def read() -> dict[str, Any]:
        if not backend.available:
            raise RuntimeError(f"window backend {backend.name!r} unavailable")
        found = [
            w
            for w in backend.list_windows()
            if w.visible and any(t in w.title.lower() for t in CHOOSER_TITLES)
        ]
        return {
            "backend": backend.name,
            "present": bool(found),
            "count": len(found),
            "title": found[0].title if found else None,
            #: Not readable on any current backend -- see the docstring.
            "options": None,
            "titles": [w.title for w in found[:5]],
        }

    return _observe(Source.WINDOW, "chooser_state()", read)


# -- python environment ----------------------------------------------------
def venv_python(venv_dir: str | os.PathLike) -> Path:
    """Interpreter path inside a venv, for the host layout."""
    base = Path(venv_dir)
    return base / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


_ENV_PROBE = (
    "import json,sys,sysconfig;"
    "print(json.dumps({'executable':sys.executable,'version':sys.version.split()[0],"
    "'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
    "'purelib':sysconfig.get_paths().get('purelib')}))"
)


def python_env_state(venv_dir: str | os.PathLike) -> Observation:
    """Interpreter identity of a venv -- the environment half of plan S8's install check."""
    interpreter = venv_python(venv_dir)

    def read() -> dict[str, Any]:
        if not interpreter.exists():
            return {"venv": str(venv_dir), "interpreter_exists": False}
        proc = subprocess.run(
            [str(interpreter), "-c", _ENV_PROBE],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"interpreter probe exit={proc.returncode}: {proc.stderr[:300]}")
        payload = json.loads(proc.stdout.strip())
        payload.update(
            {
                "venv": str(venv_dir),
                "interpreter_exists": True,
                # A real venv has base_prefix != prefix; this catches a dir that
                # merely looks like a venv.
                "is_isolated": payload["prefix"] != payload["base_prefix"],
            }
        )
        return payload

    return _observe(Source.SHELL, f"python_env_state({venv_dir})", read)


_PKG_PROBE = """
import importlib, importlib.metadata as md, json, sys
out = {}
for name in json.loads(sys.argv[1]):
    entry = {"import_ok": False, "version": None, "error": None}
    try:
        entry["version"] = md.version(name)
    except Exception as exc:
        entry["error"] = f"metadata: {type(exc).__name__}"
    try:
        importlib.import_module(name.replace("-", "_"))
        entry["import_ok"] = True
    except Exception as exc:
        entry["error"] = f"{entry['error'] or ''} import: {type(exc).__name__}".strip()
    out[name] = entry
print(json.dumps(out))
"""


def package_state(venv_dir: str | os.PathLike, packages: Iterable[str]) -> Observation:
    """Import + version check inside the target venv (plan S8).

    Runs in the *target* interpreter, not the runtime's own, so a package
    installed into the wrong environment reads as missing -- which is the whole
    point of the check.
    """
    interpreter = venv_python(venv_dir)
    wanted = list(packages)

    def read() -> dict[str, Any]:
        if not interpreter.exists():
            return {"venv": str(venv_dir), "interpreter_exists": False, "packages": {}}
        proc = subprocess.run(
            [str(interpreter), "-c", _PKG_PROBE, json.dumps(wanted)],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"package probe exit={proc.returncode}: {proc.stderr[:300]}")
        results = json.loads(proc.stdout.strip())
        return {
            "venv": str(venv_dir),
            "interpreter_exists": True,
            "packages": results,
            "all_importable": all(v["import_ok"] for v in results.values()) if results else False,
        }

    return _observe(Source.SHELL, f"package_state({venv_dir},{wanted})", read)


def recent_entries(roots: Iterable[str | os.PathLike], *, limit: int = 20) -> Observation:
    """Immediate children of ``roots``, most-recently-modified first.

    Read-only, one level deep, and confined to exactly the roots the caller
    passes in -- for the general-task route (plan: general-task routing),
    that is always ``(policy.workspace, *policy.readable_roots)``, so this
    reader can never show the planner a path ``resolve_read_path`` would not
    already let it open. It is a *listing* primitive, not a wider grant.

    A root that does not exist or is not a directory is skipped rather than
    failing the whole read: ``readable_roots`` may legitimately be empty or
    point somewhere not yet created, and one missing root should not make
    every other root's listing come back as UNKNOWN.
    """
    root_paths = [Path(root) for root in roots]

    def read() -> dict[str, Any]:
        found: list[dict[str, Any]] = []
        for root in root_paths:
            if not root.exists() or not root.is_dir():
                continue
            try:
                children = list(root.iterdir())
            except OSError:
                continue
            for child in children:
                try:
                    stat = child.stat()
                except OSError:
                    continue
                found.append(
                    {
                        "path": str(child),
                        "name": child.name,
                        "is_dir": child.is_dir(),
                        "mtime": stat.st_mtime,
                    }
                )
        found.sort(key=lambda entry: entry["mtime"], reverse=True)
        return {
            "roots": [str(root) for root in root_paths],
            "count": len(found),
            "entries": found[:limit],
        }

    label = ",".join(str(root) for root in root_paths)
    return _observe(Source.FILESYSTEM, f"recent_entries({label})", read)


def summarize(observations: dict[str, Observation]) -> dict[str, Any]:
    """Planner-facing view of state, ages included.

    Ages are exposed on purpose: the planner is allowed to *see* that state is
    old, but it is the runtime -- not the planner -- that refuses to act on it.
    """
    return {
        name: {
            "source": obs.source.value,
            "ok": obs.ok,
            "age_s": round(obs.age(), 3),
            "value": obs.value,
            "error": obs.error,
        }
        for name, obs in observations.items()
    }