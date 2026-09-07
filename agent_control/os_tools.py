"""
Semantic OS action layer (plan S7).

Every function here expresses intent -- ``create_dir``, ``install_requirements``,
``open_file`` -- never mouse coordinates. Each one:

1. asks the policy layer for permission (defence in depth; the runner asks too),
2. performs the narrowest operation that achieves the intent,
3. returns evidence in an :class:`ActionResult`.

What it deliberately does **not** do is decide whether it worked. That is
verifiers.py, re-reading the world from scratch (plan S8).

`shell=True` is never used anywhere in this module: commands are argv lists, so
a filename or URL carrying shell metacharacters is an argument, not syntax.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import httpx

from .observe import venv_python
from .policy import Policy
from .types import Action, ActionResult, FailureClass, PolicyDenied


#: Semantic app name -> how to launch it, and how to recognise it afterwards.
#: The recognition fields are consumed by verifiers.py, so "launched" and
#: "verified running" cannot drift apart.
APP_REGISTRY: dict[str, dict[str, Any]] = {
    "vscode": {
        "executables": ["code.cmd", "code", "Code.exe"],
        "windows_candidates": [
            r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe",
            r"%PROGRAMFILES%\Microsoft VS Code\Code.exe",
        ],
        #: Walked up from a `bin/` shim found on PATH to the real binary.
        "real_binary_relative": ["../Code.exe", "Code.exe"],
        #: Exact process basenames.
        "process_names": ["Code.exe", "code"],
        "window_title_contains": "Visual Studio Code",
    },
    "chrome": {
        "executables": ["chrome.exe", "chrome"],
        "windows_candidates": [
            r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe",
            r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe",
            r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
        ],
        "real_binary_relative": [],
        #: Exact Chrome process basenames.
        "process_names": ["chrome.exe", "chrome"],
        "window_title_contains": "Google Chrome",
    },
}


def _result(
    action: Action,
    started: float,
    *,
    ok: bool,
    detail: dict[str, Any] | None = None,
    error: str | None = None,
    failure_class: FailureClass | None = None,
) -> ActionResult:
    return ActionResult(
        action=action,
        ok=ok,
        detail=detail or {},
        error=error,
        failure_class=failure_class,
        duration_s=time.time() - started,
    )


def _guard(fn: Callable[..., ActionResult]) -> Callable[..., ActionResult]:
    """Turn policy denials and unexpected errors into classified ActionResults."""

    @wraps(fn)
    def wrapper(policy: Policy, action: Action) -> ActionResult:
        started = time.time()

        try:
            policy.enforce(action)
            return fn(policy, action)

        except PolicyDenied as exc:
            return _result(
                action,
                started,
                ok=False,
                error=str(exc),
                failure_class=FailureClass.PERMISSION_DENIED,
            )

        except subprocess.TimeoutExpired as exc:
            return _result(
                action,
                started,
                ok=False,
                error=f"timeout: {exc}",
                failure_class=FailureClass.TRANSIENT,
            )

        except FileNotFoundError as exc:
            return _result(
                action,
                started,
                ok=False,
                error=f"missing: {exc}",
                failure_class=FailureClass.PRECONDITION_FAILED,
            )

        except Exception as exc:
            return _result(
                action,
                started,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                failure_class=FailureClass.ACTION_FAILED,
            )

    return wrapper


# -- filesystem ------------------------------------------------------------


@_guard
def create_dir(policy: Policy, action: Action) -> ActionResult:
    started = time.time()

    target = policy.resolve_write_path(action.params["path"])
    existed = target.exists()

    target.mkdir(parents=True, exist_ok=True)

    return _result(
        action,
        started,
        ok=True,
        detail={
            "path": str(target),
            "already_existed": existed,
        },
    )


@_guard
def write_file(policy: Policy, action: Action) -> ActionResult:
    started = time.time()

    target = policy.resolve_write_path(action.params["path"])
    content = action.params.get("content", "")

    data = (
        content.encode("utf-8")
        if isinstance(content, str)
        else bytes(content)
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    return _result(
        action,
        started,
        ok=True,
        detail={
            "path": str(target),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    )


@_guard
def fetch_file(policy: Policy, action: Action) -> ActionResult:
    """Retrieve a file over HTTPS rather than clicking through a browser."""

    started = time.time()

    url = policy.check_url(action.params["url"])
    dest = policy.resolve_write_path(action.params["dest"])

    dest.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[bytes] = []
    total = 0

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()

            for chunk in response.iter_bytes():
                total += len(chunk)

                if total > policy.max_download_bytes:
                    raise PolicyDenied(
                        f"download exceeded "
                        f"{policy.max_download_bytes} bytes: {url}"
                    )

                chunks.append(chunk)

            status = response.status_code

    body = b"".join(chunks)
    dest.write_bytes(body)

    return _result(
        action,
        started,
        ok=True,
        detail={
            "url": url,
            "dest": str(dest),
            "status": status,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "untrusted_preview": body[:512].decode("utf-8", "replace"),
        },
    )


@_guard
def open_file(policy: Policy, action: Action) -> ActionResult:
    """
    Open an existing file using the operating system's default application.

    This is intentionally semantic: the caller supplies a file path, not
    a specific executable or mouse coordinates.
    """

    started = time.time()

    target = policy.resolve_read_path(action.params["path"])

    if not target.exists():
        return _result(
            action,
            started,
            ok=False,
            error=f"file does not exist: {target}",
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"path": str(target)},
        )

    if not target.is_file():
        return _result(
            action,
            started,
            ok=False,
            error=f"path is not a file: {target}",
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"path": str(target)},
        )

    if sys.platform == "win32":
        os.startfile(str(target))

    elif sys.platform == "darwin":
        subprocess.Popen(
            ["open", str(target)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            shell=False,
        )

    else:
        subprocess.Popen(
            ["xdg-open", str(target)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            shell=False,
        )

    settle = float(action.params.get("settle_s", 3.0))

    if settle > 0:
        time.sleep(settle)

    return _result(
        action,
        started,
        ok=True,
        detail={
            "path": str(target),
            "suffix": target.suffix.lower(),
            "settled_s": settle,
        },
    )


# -- read-only inspection --------------------------------------------------

_MAX_DIRECTORY_ENTRIES = 200
_MAX_READ_LINES = 500
_MAX_READ_BYTES = 256 * 1024
_MAX_SEARCH_RESULTS = 100
_MAX_SEARCH_FILES = 2_000
_MAX_SEARCH_FILE_BYTES = 512 * 1024
_SEARCH_SKIP_DIRS = frozenset({".git", ".venv", "__pycache__", "node_modules"})


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


@_guard
def list_directory(policy: Policy, action: Action) -> ActionResult:
    """Return a bounded, non-recursive listing of a permitted directory."""
    started = time.time()
    target = policy.resolve_read_path(action.params["path"])
    if not target.exists():
        return _result(action, started, ok=False, error=f"directory does not exist: {target}",
                       failure_class=FailureClass.PRECONDITION_FAILED, detail={"path": str(target)})
    if not target.is_dir():
        return _result(action, started, ok=False, error=f"path is not a directory: {target}",
                       failure_class=FailureClass.PRECONDITION_FAILED, detail={"path": str(target)})
    limit = _bounded_int(action.params.get("max_entries"), default=_MAX_DIRECTORY_ENTRIES,
                         minimum=1, maximum=_MAX_DIRECTORY_ENTRIES)
    entries = []
    truncated = False
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        # A readable parent must not turn a sensitive child into planner-visible data.
        try:
            resolved = policy.resolve_read_path(child)
        except PolicyDenied:
            continue
        if len(entries) >= limit:
            truncated = True
            break
        entries.append({
            "name": resolved.name,
            "kind": "directory" if resolved.is_dir() else "file",
            "suffix": resolved.suffix.lower() if resolved.is_file() else "",
        })
    return _result(action, started, ok=True, detail={
        "path": str(target), "entry_count": len(entries), "entries": entries,
        "truncated": truncated,
    })


@_guard
def read_text_file(policy: Policy, action: Action) -> ActionResult:
    """Read a bounded line window from a permitted text file."""
    started = time.time()
    target = policy.resolve_read_path(action.params["path"])
    if not target.exists():
        return _result(action, started, ok=False, error=f"file does not exist: {target}",
                       failure_class=FailureClass.PRECONDITION_FAILED, detail={"path": str(target)})
    if not target.is_file():
        return _result(action, started, ok=False, error=f"path is not a file: {target}",
                       failure_class=FailureClass.PRECONDITION_FAILED, detail={"path": str(target)})
    size = target.stat().st_size
    if size > _MAX_READ_BYTES:
        return _result(action, started, ok=False,
                       error=f"file exceeds read limit ({_MAX_READ_BYTES} bytes): {target}",
                       failure_class=FailureClass.PRECONDITION_FAILED,
                       detail={"path": str(target), "bytes": size, "max_bytes": _MAX_READ_BYTES})
    raw = target.read_bytes()
    if b"\x00" in raw:
        return _result(action, started, ok=False, error=f"binary file refused: {target}",
                       failure_class=FailureClass.PRECONDITION_FAILED, detail={"path": str(target)})
    try:
        text = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        return _result(action, started, ok=False, error=f"non-UTF-8 text file refused: {target}",
                       failure_class=FailureClass.PRECONDITION_FAILED, detail={"path": str(target)})
    lines = text.splitlines()
    start = _bounded_int(action.params.get("start_line"), default=1, minimum=1, maximum=max(1, len(lines) or 1))
    limit = _bounded_int(action.params.get("max_lines"), default=300, minimum=1, maximum=_MAX_READ_LINES)
    window = lines[start - 1:start - 1 + limit]
    end = start + len(window) - 1 if window else start - 1
    return _result(action, started, ok=True, detail={
        "path": str(target), "encoding": encoding, "start_line": start, "end_line": end,
        "total_lines": len(lines), "truncated": end < len(lines), "content": "\n".join(window),
    })


@_guard
def search_files(policy: Policy, action: Action) -> ActionResult:
    """Search bounded UTF-8 text files under a permitted directory."""
    started = time.time()
    root = policy.resolve_read_path(action.params["path"])
    query = action.params.get("query")
    if not isinstance(query, str) or not query:
        return _result(action, started, ok=False, error="search_files requires non-empty 'query'",
                       failure_class=FailureClass.PRECONDITION_FAILED)
    if not root.exists() or not root.is_dir():
        return _result(action, started, ok=False, error=f"path is not an existing directory: {root}",
                       failure_class=FailureClass.PRECONDITION_FAILED, detail={"path": str(root)})
    max_results = _bounded_int(action.params.get("max_results"), default=20, minimum=1, maximum=_MAX_SEARCH_RESULTS)
    matches, scanned, truncated = [], 0, False
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in _SEARCH_SKIP_DIRS]
        for name in sorted(files):
            if scanned >= _MAX_SEARCH_FILES or len(matches) >= max_results:
                truncated = True
                break
            candidate = Path(current) / name
            try:
                target = policy.resolve_read_path(candidate)
            except PolicyDenied:
                continue
            try:
                if not target.is_file() or target.stat().st_size > _MAX_SEARCH_FILE_BYTES:
                    continue
                raw = target.read_bytes()
                scanned += 1
                if b"\x00" in raw:
                    continue
                text = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                if query in line:
                    matches.append({"path": str(target), "line": line_no, "text": line[:1000]})
                    if len(matches) >= max_results:
                        truncated = True
                        break
        if truncated:
            break
    return _result(action, started, ok=True, detail={
        "path": str(root), "query": query, "matches": matches,
        "match_count": len(matches), "files_scanned": scanned, "truncated": truncated,
    })


# -- shell / environment ---------------------------------------------------


@_guard
def run_command(policy: Policy, action: Action) -> ActionResult:
    started = time.time()

    argv = [str(arg) for arg in action.params["argv"]]

    if not argv:
        return _result(
            action,
            started,
            ok=False,
            error="argv must not be empty",
            failure_class=FailureClass.PRECONDITION_FAILED,
        )

    policy.check_executable(argv)

    cwd = policy.resolve_write_path(
        action.params.get("cwd") or policy.workspace
    )

    timeout = float(action.params.get("timeout", 300))

    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        shell=False,
    )

    return _result(
        action,
        started,
        ok=proc.returncode == 0,
        detail={
            "argv": argv,
            "cwd": str(cwd),
            "exit_code": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        },
        error=None if proc.returncode == 0 else f"exit_code={proc.returncode}",
        failure_class=(
            None
            if proc.returncode == 0
            else FailureClass.ACTION_FAILED
        ),
    )


@_guard
def create_venv(policy: Policy, action: Action) -> ActionResult:
    started = time.time()

    target = policy.resolve_write_path(action.params["venv"])

    argv = [
        sys.executable,
        "-m",
        "venv",
        str(target),
    ]

    policy.check_executable(argv)

    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        shell=False,
    )

    return _result(
        action,
        started,
        ok=proc.returncode == 0,
        detail={
            "venv": str(target),
            "exit_code": proc.returncode,
            "interpreter": str(venv_python(target)),
            "stderr_tail": proc.stderr[-1000:],
        },
        error=(
            None
            if proc.returncode == 0
            else f"venv exit_code={proc.returncode}"
        ),
        failure_class=(
            None
            if proc.returncode == 0
            else FailureClass.ACTION_FAILED
        ),
    )


@_guard
def install_requirements(policy: Policy, action: Action) -> ActionResult:
    """Install into the named environment. Exit code is evidence, not proof."""

    started = time.time()

    venv_dir = policy.resolve_write_path(action.params["venv"])
    req = policy.resolve_read_path(action.params["requirements"])

    interpreter = venv_python(venv_dir)

    if not interpreter.exists():
        return _result(
            action,
            started,
            ok=False,
            error=f"no interpreter at {interpreter}",
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"venv": str(venv_dir)},
        )

    argv = [
        str(interpreter),
        "-m",
        "pip",
        "install",
        "--requirement",
        str(req),
        "--no-input",
        "--disable-pip-version-check",
    ]

    policy.check_executable(argv)

    proc = subprocess.run(
        argv,
        cwd=str(policy.workspace),
        capture_output=True,
        text=True,
        timeout=float(action.params.get("timeout", 900)),
        check=False,
        shell=False,
    )

    return _result(
        action,
        started,
        ok=proc.returncode == 0,
        detail={
            "venv": str(venv_dir),
            "requirements": str(req),
            "exit_code": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        },
        error=(
            None
            if proc.returncode == 0
            else f"pip exit_code={proc.returncode}"
        ),
        failure_class=(
            None
            if proc.returncode == 0
            else FailureClass.ACTION_FAILED
        ),
    )


# -- application lifecycle -------------------------------------------------


def resolve_app_executable(app: str) -> Path | None:
    """Find a registered app's launcher, or None if it is not installed."""

    spec = APP_REGISTRY.get(app)

    if not spec:
        return None

    windows_candidates = (
        spec.get("windows_candidates", [])
        if sys.platform == "win32"
        else []
    )

    for raw in windows_candidates:
        candidate = Path(os.path.expandvars(raw))

        if candidate.exists():
            return candidate

    for name in spec.get("executables", []):
        found = shutil.which(name)

        if found:
            return _prefer_real_binary(Path(found), spec)

    return None


def _prefer_real_binary(shim: Path, spec: dict[str, Any]) -> Path:
    """Prefer the real executable over a launcher shim when available."""

    if shim.suffix.lower() not in {".cmd", ".bat", ""}:
        return shim

    for relative in spec.get("real_binary_relative", []):
        candidate = (shim.parent / relative).resolve()

        if (
            candidate.exists()
            and candidate.suffix.lower() == ".exe"
        ):
            return candidate

    return shim


@_guard
def launch_app(policy: Policy, action: Action) -> ActionResult:
    """Launch through the OS process layer, not by clicking an icon."""

    started = time.time()

    app = action.params["app"]

    if app not in APP_REGISTRY:
        return _result(
            action,
            started,
            ok=False,
            error=f"unregistered app {app!r}",
            failure_class=FailureClass.PRECONDITION_FAILED,
        )

    executable = resolve_app_executable(app)

    if executable is None:
        return _result(
            action,
            started,
            ok=False,
            error=f"{app!r} is not installed on this host",
            failure_class=FailureClass.ENVIRONMENT,
            detail={"app": app},
        )

    argv = [str(executable)]

    if action.params.get("open_path"):
        argv.append(
            str(
                policy.resolve_read_path(
                    action.params["open_path"]
                )
            )
        )

    proc = subprocess.Popen(
        argv,
        cwd=str(policy.workspace),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        shell=False,
    )

    settle = float(action.params.get("settle_s", 6.0))

    if settle > 0:
        time.sleep(settle)

    return _result(
        action,
        started,
        ok=True,
        detail={
            "app": app,
            "executable": str(executable),
            "launcher_pid": proc.pid,
            "settled_s": settle,
        },
    )


#: Semantic action kind -> executor.
DISPATCH: dict[str, Callable[[Policy, Action], ActionResult]] = {
    "create_dir": create_dir,
    "write_file": write_file,
    "fetch_file": fetch_file,
    "open_file": open_file,
    "list_directory": list_directory,
    "read_text_file": read_text_file,
    "search_files": search_files,
    "run_command": run_command,
    "create_venv": create_venv,
    "install_requirements": install_requirements,
    "launch_app": launch_app,
}


def execute(policy: Policy, action: Action) -> ActionResult:
    """Dispatch a semantic action, refusing unknown kinds."""

    handler = DISPATCH.get(action.kind)

    if handler is None:
        return ActionResult(
            action=action,
            ok=False,
            error=(
                f"unknown action kind {action.kind!r}; "
                f"allowed: {sorted(DISPATCH)}"
            ),
            failure_class=FailureClass.PRECONDITION_FAILED,
        )

    return handler(policy, action)