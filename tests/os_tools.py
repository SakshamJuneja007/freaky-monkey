"""
Semantic OS action layer (plan S7).

Every function here expresses intent -- ``create_dir``, ``install_requirements``,
``open_file`` -- never mouse coordinates. Each one:

1. asks the policy layer for permission (defence in depth; the runner asks too),
2. performs the narrowest operation that achieves the intent,
3. returns evidence in an ActionResult.

What it deliberately does **not** do is decide whether it worked. That is
verifiers.py, re-reading the world from scratch (plan S8).

shell=True is never used anywhere in this module: commands are argv lists, so
a filename or URL carrying shell metacharacters is an argument, not syntax.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import httpx
import psutil

from .observe import venv_python
from .policy import BLOCKED_APP_EXECUTABLES, Policy
from .types import Action, ActionResult, FailureClass, PolicyDenied


# Semantic app name -> how to launch it, and how to recognise it afterwards.
# The recognition fields are consumed by verifiers.py, so "launched" and
# "verified running" cannot drift apart.
APP_REGISTRY: dict[str, dict[str, Any]] = {
    "vscode": {
        "executables": ["code.cmd", "code", "Code.exe"],
        "windows_candidates": [
            r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe",
            r"%PROGRAMFILES%\Microsoft VS Code\Code.exe",
        ],
        # Walked up from a ``bin/`` shim found on PATH to the real binary.
        "real_binary_relative": ["../Code.exe", "Code.exe"],
        # Exact process basenames.
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
        # Exact Chrome process basenames.
        "process_names": ["chrome.exe", "chrome"],
        "window_title_contains": "Google Chrome",
    },
}


_MAX_DIRECTORY_ENTRIES = 200
_MAX_READ_LINES = 500
_MAX_READ_BYTES = 256 * 1024
_MAX_SEARCH_RESULTS = 100
_MAX_SEARCH_FILES = 2_000
_MAX_SEARCH_FILE_BYTES = 512 * 1024
_MAX_COMMAND_TIMEOUT = 900.0
_MAX_INSTALL_TIMEOUT = 1_800.0
_MAX_OPEN_SETTLE_SECONDS = 30.0
_MAX_APP_SETTLE_SECONDS = 30.0

_SEARCH_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)


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


def _guard(
    fn: Callable[..., ActionResult],
) -> Callable[..., ActionResult]:
    """
    Turn policy denials and unexpected errors into classified ActionResults.

    Policy enforcement remains here as defence in depth even when the runner
    already performs permission checks.
    """

    @wraps(fn)
    def wrapper(
        policy: Policy,
        action: Action,
    ) -> ActionResult:
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


def _bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default

    return max(minimum, min(parsed, maximum))


def _bounded_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default

    return max(minimum, min(parsed, maximum))


def _is_http_url(value: str) -> bool:
    """
    Return True only for HTTP(S) URLs.

    This deliberately does not treat arbitrary ``scheme:...`` strings as URLs.
    Application launch targets should be explicit rather than guessing whether a
    string is a URL or a filesystem path.
    """
    return value.startswith("https://") or value.startswith("http://")


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------


@_guard
def create_dir(
    policy: Policy,
    action: Action,
) -> ActionResult:
    started = time.time()

    target = policy.resolve_write_path(
        action.params["path"]
    )

    existed = target.exists()

    if existed and not target.is_dir():
        return _result(
            action,
            started,
            ok=False,
            error=f"path exists but is not a directory: {target}",
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"path": str(target)},
        )

    target.mkdir(
        parents=True,
        exist_ok=True,
    )

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
def write_file(
    policy: Policy,
    action: Action,
) -> ActionResult:
    """
    Write a file inside a permitted location.

    Existing files are not overwritten unless ``overwrite=True`` is explicitly
    supplied by the caller.
    """
    started = time.time()

    target = policy.resolve_write_path(
        action.params["path"]
    )

    overwrite = bool(
        action.params.get("overwrite", False)
    )

    existed = target.exists()

    if existed:
        if not target.is_file():
            return _result(
                action,
                started,
                ok=False,
                error=f"path exists but is not a file: {target}",
                failure_class=FailureClass.PRECONDITION_FAILED,
                detail={"path": str(target)},
            )

        if not overwrite:
            return _result(
                action,
                started,
                ok=False,
                error=(
                    f"refusing to overwrite existing file "
                    f"without overwrite=True: {target}"
                ),
                failure_class=FailureClass.PRECONDITION_FAILED,
                detail={"path": str(target)},
            )

    content = action.params.get("content", "")

    if isinstance(content, str):
        data = content.encode("utf-8")
    else:
        data = bytes(content)

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_bytes(data)

    return _result(
        action,
        started,
        ok=True,
        detail={
            "path": str(target),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "overwritten": existed and overwrite,
        },
    )


@_guard
def fetch_file(
    policy: Policy,
    action: Action,
) -> ActionResult:
    """
    Retrieve a file over HTTPS without browser interaction.

    The response is streamed directly to a temporary file so the complete
    download is never unnecessarily buffered in memory. The completed file is
    atomically moved into place only after the full response succeeds.
    """
    started = time.time()

    url = policy.check_url(
        action.params["url"]
    )

    dest = policy.resolve_write_path(
        action.params["dest"]
    )

    overwrite = bool(
        action.params.get("overwrite", False)
    )

    if dest.exists() and not overwrite:
        return _result(
            action,
            started,
            ok=False,
            error=(
                f"refusing to overwrite existing file "
                f"without overwrite=True: {dest}"
            ),
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"dest": str(dest)},
        )

    dest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    max_bytes = int(
        getattr(
            policy,
            "max_download_bytes",
            100 * 1024 * 1024,
        )
    )

    total = 0
    digest = hashlib.sha256()
    preview = bytearray()
    temp_path: Path | None = None
    status: int | None = None
    final_url = url

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=str(dest.parent),
            prefix=f".{dest.name}.",
            suffix=".part",
        ) as temp_file:
            temp_path = Path(temp_file.name)

            with httpx.Client(
                timeout=30.0,
                follow_redirects=False,
            ) as client:
                current_url = url
                redirects = 0
                max_redirects = 5

                while True:
                    with client.stream(
                        "GET",
                        current_url,
                    ) as response:

                        # Every redirect target must pass policy validation.
                        if 300 <= response.status_code < 400:
                            location = response.headers.get(
                                "location"
                            )

                            if not location:
                                response.raise_for_status()

                            redirects += 1

                            if redirects > max_redirects:
                                raise PolicyDenied(
                                    "too many redirects"
                                )

                            next_url = str(
                                response.url.join(
                                    location
                                )
                            )

                            current_url = policy.check_url(
                                next_url
                            )

                            continue

                        response.raise_for_status()

                        status = response.status_code
                        final_url = str(response.url)

                        # Validate the final resolved URL too.
                        policy.check_url(final_url)

                        for chunk in response.iter_bytes():
                            total += len(chunk)

                            if total > max_bytes:
                                raise PolicyDenied(
                                    f"download exceeded "
                                    f"{max_bytes} bytes: "
                                    f"{final_url}"
                                )

                            digest.update(chunk)
                            temp_file.write(chunk)

                            remaining_preview = (
                                512 - len(preview)
                            )

                            if remaining_preview > 0:
                                preview.extend(
                                    chunk[
                                        :remaining_preview
                                    ]
                                )

                        break

        if temp_path is None:
            raise RuntimeError(
                "temporary download file was not created"
            )

        os.replace(
            str(temp_path),
            str(dest),
        )

    except Exception:
        if (
            temp_path is not None
            and temp_path.exists()
        ):
            try:
                temp_path.unlink()
            except OSError:
                pass

        raise

    return _result(
        action,
        started,
        ok=True,
        detail={
            "url": final_url,
            "dest": str(dest),
            "status": status,
            "bytes": total,
            "sha256": digest.hexdigest(),
            "untrusted_preview": bytes(
                preview
            ).decode(
                "utf-8",
                "replace",
            ),
        },
    )


@_guard
def open_file(
    policy: Policy,
    action: Action,
) -> ActionResult:
    """
    Ask the operating system to open an existing file using its default
    application.

    The action only reports that the launch request was issued. Verification
    that the file actually opened belongs to the verifier layer.
    """
    started = time.time()

    target = policy.resolve_read_path(
        action.params["path"]
    )

    if not target.exists():
        return _result(
            action,
            started,
            ok=False,
            error=f"file does not exist: {target}",
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"path": str(target)},
        )

    if not target.is_file() and not target.is_dir():
        return _result(
            action,
            started,
            ok=False,
            error=f"path is not a file or directory: {target}",
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

    settle = _bounded_float(
        action.params.get(
            "settle_s",
            3.0,
        ),
        default=3.0,
        minimum=0.0,
        maximum=_MAX_OPEN_SETTLE_SECONDS,
    )

    if settle > 0:
        time.sleep(settle)

    return _result(
        action,
        started,
        ok=True,
        detail={
            "path": str(target),
            "suffix": target.suffix.lower(),
            "launch_requested": True,
            "settled_s": settle,
        },
    )


# ---------------------------------------------------------------------------
# Read-only inspection
# ---------------------------------------------------------------------------


@_guard
def list_directory(
    policy: Policy,
    action: Action,
) -> ActionResult:
    """
    Return a bounded, non-recursive listing of a permitted directory.
    """
    started = time.time()

    target = policy.resolve_read_path(
        action.params["path"]
    )

    if not target.exists():
        return _result(
            action,
            started,
            ok=False,
            error=f"directory does not exist: {target}",
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"path": str(target)},
        )

    if not target.is_dir():
        return _result(
            action,
            started,
            ok=False,
            error=f"path is not a directory: {target}",
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"path": str(target)},
        )

    limit = _bounded_int(
        action.params.get("max_entries"),
        default=_MAX_DIRECTORY_ENTRIES,
        minimum=1,
        maximum=_MAX_DIRECTORY_ENTRIES,
    )

    entries: list[dict[str, str]] = []
    truncated = False

    for child in sorted(
        target.iterdir(),
        key=lambda p: (
            not p.is_dir(),
            p.name.lower(),
        ),
    ):
        try:
            resolved = policy.resolve_read_path(
                child
            )
        except PolicyDenied:
            continue

        if len(entries) >= limit:
            truncated = True
            break

        entries.append(
            {
                "name": resolved.name,
                "kind": (
                    "directory"
                    if resolved.is_dir()
                    else "file"
                ),
                "suffix": (
                    resolved.suffix.lower()
                    if resolved.is_file()
                    else ""
                ),
            }
        )

    return _result(
        action,
        started,
        ok=True,
        detail={
            "path": str(target),
            "entry_count": len(entries),
            "entries": entries,
            "truncated": truncated,
        },
    )


@_guard
def read_text_file(
    policy: Policy,
    action: Action,
) -> ActionResult:
    """
    Read a bounded line window from a permitted UTF-8 text file.
    """
    started = time.time()

    target = policy.resolve_read_path(
        action.params["path"]
    )

    if not target.exists():
        return _result(
            action,
            started,
            ok=False,
            error=f"file does not exist: {target}",
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"path": str(target)},
        )

    if not target.is_file() and not target.is_dir():
        return _result(
            action,
            started,
            ok=False,
            error=f"path is not a file or directory: {target}",
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"path": str(target)},
        )

    size = target.stat().st_size

    if size > _MAX_READ_BYTES:
        return _result(
            action,
            started,
            ok=False,
            error=(
                f"file exceeds read limit "
                f"({_MAX_READ_BYTES} bytes): {target}"
            ),
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={
                "path": str(target),
                "bytes": size,
                "max_bytes": _MAX_READ_BYTES,
            },
        )

    raw = target.read_bytes()

    if b"\x00" in raw:
        return _result(
            action,
            started,
            ok=False,
            error=f"binary file refused: {target}",
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"path": str(target)},
        )

    try:
        text = raw.decode("utf-8")

    except UnicodeDecodeError:
        return _result(
            action,
            started,
            ok=False,
            error=(
                f"non-UTF-8 text file refused: "
                f"{target}"
            ),
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"path": str(target)},
        )

    lines = text.splitlines()

    start = _bounded_int(
        action.params.get("start_line"),
        default=1,
        minimum=1,
        maximum=max(1, len(lines) or 1),
    )

    limit = _bounded_int(
        action.params.get("max_lines"),
        default=300,
        minimum=1,
        maximum=_MAX_READ_LINES,
    )

    window = lines[
        start - 1:
        start - 1 + limit
    ]

    end = (
        start + len(window) - 1
        if window
        else start - 1
    )

    return _result(
        action,
        started,
        ok=True,
        detail={
            "path": str(target),
            "encoding": "utf-8",
            "start_line": start,
            "end_line": end,
            "total_lines": len(lines),
            "truncated": end < len(lines),
            "content": "\n".join(window),
        },
    )


@_guard
def search_files(
    policy: Policy,
    action: Action,
) -> ActionResult:
    """
    Search bounded UTF-8 text files under a permitted directory.

    Both filesystem traversal and file-content reads are independently bounded.
    """
    started = time.time()

    root = policy.resolve_read_path(
        action.params["path"]
    )

    query = action.params.get("query")

    if not isinstance(query, str) or not query:
        return _result(
            action,
            started,
            ok=False,
            error=(
                "search_files requires "
                "non-empty 'query'"
            ),
            failure_class=FailureClass.PRECONDITION_FAILED,
        )

    if not root.exists() or not root.is_dir():
        return _result(
            action,
            started,
            ok=False,
            error=(
                f"path is not an existing directory: "
                f"{root}"
            ),
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"path": str(root)},
        )

    max_results = _bounded_int(
        action.params.get("max_results"),
        default=20,
        minimum=1,
        maximum=_MAX_SEARCH_RESULTS,
    )

    matches: list[dict[str, Any]] = []
    visited = 0
    scanned = 0
    truncated = False

    for current, dirs, files in os.walk(
        root,
        followlinks=False,
    ):
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in _SEARCH_SKIP_DIRS
        ]

        for name in sorted(files):
            visited += 1

            if visited > _MAX_SEARCH_FILES:
                truncated = True
                break

            if len(matches) >= max_results:
                truncated = True
                break

            candidate = Path(current) / name

            try:
                target = policy.resolve_read_path(
                    candidate
                )
            except PolicyDenied:
                continue

            try:
                if not target.is_file():
                    continue

                if (
                    target.stat().st_size
                    > _MAX_SEARCH_FILE_BYTES
                ):
                    continue

                raw = target.read_bytes()
                scanned += 1

                if b"\x00" in raw:
                    continue

                text = raw.decode("utf-8")

            except (
                OSError,
                UnicodeDecodeError,
            ):
                continue

            for line_no, line in enumerate(
                text.splitlines(),
                1,
            ):
                if query in line:
                    matches.append(
                        {
                            "path": str(target),
                            "line": line_no,
                            "text": line[:1000],
                        }
                    )

                    if len(matches) >= max_results:
                        truncated = True
                        break

        if truncated:
            break

    return _result(
        action,
        started,
        ok=True,
        detail={
            "path": str(root),
            "query": query,
            "matches": matches,
            "match_count": len(matches),
            "files_visited": visited,
            "files_scanned": scanned,
            "truncated": truncated,
        },
    )


# ---------------------------------------------------------------------------
# Shell / environment
# ---------------------------------------------------------------------------


@_guard
def run_command(
    policy: Policy,
    action: Action,
) -> ActionResult:
    """
    Execute a policy-approved argv command.

    Commands are never passed through a shell.
    """
    started = time.time()

    raw_argv = action.params.get("argv")

    if not isinstance(raw_argv, (list, tuple)):
        return _result(
            action,
            started,
            ok=False,
            error="argv must be a list or tuple",
            failure_class=FailureClass.PRECONDITION_FAILED,
        )

    argv = [
        str(arg)
        for arg in raw_argv
    ]

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
        action.params.get("cwd")
        or policy.workspace
    )

    if not cwd.exists():
        return _result(
            action,
            started,
            ok=False,
            error=f"working directory does not exist: {cwd}",
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"cwd": str(cwd)},
        )

    if not cwd.is_dir():
        return _result(
            action,
            started,
            ok=False,
            error=f"working path is not a directory: {cwd}",
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"cwd": str(cwd)},
        )

    timeout = _bounded_float(
        action.params.get(
            "timeout",
            300,
        ),
        default=300.0,
        minimum=1.0,
        maximum=_MAX_COMMAND_TIMEOUT,
    )

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
        error=(
            None
            if proc.returncode == 0
            else f"exit_code={proc.returncode}"
        ),
        failure_class=(
            None
            if proc.returncode == 0
            else FailureClass.ACTION_FAILED
        ),
    )


@_guard
def create_venv(
    policy: Policy,
    action: Action,
) -> ActionResult:
    started = time.time()

    target = policy.resolve_write_path(
        action.params["venv"]
    )

    if target.exists() and any(
        target.iterdir()
    ):
        return _result(
            action,
            started,
            ok=False,
            error=(
                f"refusing to create venv in "
                f"non-empty directory: {target}"
            ),
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={"venv": str(target)},
        )

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
            "interpreter": str(
                venv_python(target)
            ),
            "stderr_tail": proc.stderr[-1000:],
        },
        error=(
            None
            if proc.returncode == 0
            else (
                f"venv exit_code="
                f"{proc.returncode}"
            )
        ),
        failure_class=(
            None
            if proc.returncode == 0
            else FailureClass.ACTION_FAILED
        ),
    )


@_guard
def install_requirements(
    policy: Policy,
    action: Action,
) -> ActionResult:
    """
    Install requirements into the named environment.

    A successful process exit is execution evidence. The verifier layer decides
    whether the requested environment state was actually achieved.
    """
    started = time.time()

    venv_dir = policy.resolve_write_path(
        action.params["venv"]
    )

    req = policy.resolve_read_path(
        action.params["requirements"]
    )

    if not req.exists() or not req.is_file():
        return _result(
            action,
            started,
            ok=False,
            error=(
                f"requirements file does not exist: "
                f"{req}"
            ),
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={
                "requirements": str(req),
            },
        )

    interpreter = venv_python(
        venv_dir
    )

    if not interpreter.exists():
        return _result(
            action,
            started,
            ok=False,
            error=(
                f"no interpreter at "
                f"{interpreter}"
            ),
            failure_class=FailureClass.PRECONDITION_FAILED,
            detail={
                "venv": str(venv_dir),
            },
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

    timeout = _bounded_float(
        action.params.get(
            "timeout",
            900,
        ),
        default=900.0,
        minimum=1.0,
        maximum=_MAX_INSTALL_TIMEOUT,
    )

    proc = subprocess.run(
        argv,
        cwd=str(policy.workspace),
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
            "venv": str(venv_dir),
            "requirements": str(req),
            "exit_code": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        },
        error=(
            None
            if proc.returncode == 0
            else (
                f"pip exit_code="
                f"{proc.returncode}"
            )
        ),
        failure_class=(
            None
            if proc.returncode == 0
            else FailureClass.ACTION_FAILED
        ),
    )


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


def _normalize_app_name(app: str) -> str:
    name = " ".join(str(app).strip().split()).lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name


def _blocked_app_name(app: str) -> bool:
    normalized = _normalize_app_name(app)
    return normalized in BLOCKED_APP_EXECUTABLES


def _app_aliases(app: str) -> tuple[str, ...]:
    normalized = _normalize_app_name(app)
    aliases = {normalized}
    aliases.update({
        "vscode" if normalized in {"visual studio code", "vs code"} else normalized,
        "chrome" if normalized == "google chrome" else normalized,
        "edge" if normalized in {"microsoft edge", "edge browser"} else normalized,
        "mspaint" if normalized in {"paint", "microsoft paint"} else normalized,
        "calc" if normalized == "calculator" else normalized,
    })
    return tuple(sorted(aliases))


def _registry_app_executable(app: str) -> Path | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None

    aliases = _app_aliases(app)
    roots = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
    views = (0,)
    if hasattr(winreg, "KEY_WOW64_64KEY") and hasattr(winreg, "KEY_WOW64_32KEY"):
        views = (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY)

    for root in roots:
        for view in views:
            for alias in aliases:
                key_path = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{alias}.exe"
                try:
                    with winreg.OpenKey(root, key_path, 0, winreg.KEY_READ | view) as key:
                        value, _ = winreg.QueryValueEx(key, "")
                except OSError:
                    continue
                candidate = Path(os.path.expandvars(str(value).strip().strip('"')))
                if candidate.is_file():
                    return candidate
    return None


def _common_windows_candidates(app: str) -> list[Path]:
    if sys.platform != "win32":
        return []
    aliases = _app_aliases(app)
    roots = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32",
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")),
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))),
    ]
    # Direct system executables are common for Notepad/Paint/etc.; installed
    # application trees are searched only to a bounded depth by name.
    for alias in aliases:
        direct = roots[0] / f"{alias}.exe"
        if direct.is_file():
            return [direct]

    found: list[Path] = []
    wanted = {alias.replace(" ", "").lower() for alias in aliases}
    for root in roots[1:]:
        if not root.is_dir():
            continue
        try:
            for candidate in root.rglob("*.exe"):
                try:
                    relative_depth = len(candidate.relative_to(root).parts)
                except ValueError:
                    continue
                if relative_depth > 5:
                    continue
                stem = re.sub(r"[^a-z0-9]+", "", candidate.stem.lower())
                if stem in wanted and candidate.is_file():
                    found.append(candidate)
                    if len(found) >= 20:
                        return found
        except OSError:
            continue
    return found


def _chrome_configuration() -> dict[str, str]:
    """Read the existing DEIMOS Chrome environment configuration.

    These values are intentionally environment-only: they are the existing
    deployment seam for selecting the user's real Chrome installation/profile.
    No second config file or implicit username/path is introduced.
    """
    values = {
        "executable": os.getenv("DEIMOS_CHROME_EXECUTABLE", "").strip(),
        "user_data": os.getenv("DEIMOS_CHROME_USER_DATA", "").strip(),
        "profile": os.getenv("DEIMOS_CHROME_PROFILE", "").strip(),
        "debug_port": os.getenv("DEBUG_PORT", "").strip(),
    }
    return {key: value for key, value in values.items() if value}


def _chrome_process_matches_configuration(config: dict[str, str]) -> bool:
    """Return True only when an existing Chrome process matches configured state."""
    if sys.platform != "win32":
        return False
    user_data = os.path.normcase(os.path.normpath(config.get("user_data", ""))) if config.get("user_data") else ""
    profile = config.get("profile", "").casefold()
    debug_port = config.get("debug_port", "")
    try:
        for proc in psutil.process_iter(["name", "exe", "cmdline"]):
            info = proc.info
            name = str(info.get("name") or "").casefold()
            if name != "chrome.exe" and not name.endswith("\\chrome.exe"):
                continue
            cmdline = [str(x) for x in (info.get("cmdline") or [])]
            joined = " ".join(cmdline)
            folded = joined.casefold()
            if user_data:
                normalized_joined = os.path.normcase(folded.replace('\\\\', '/'))
                normalized_user = os.path.normcase(user_data.replace('\\\\', '/')).casefold()
                if normalized_user not in normalized_joined:
                    continue
            if profile:
                explicit_profile = ""
                for arg in cmdline:
                    low = arg.casefold()
                    if low.startswith("--profile-directory="):
                        explicit_profile = arg.split("=", 1)[1].strip().strip('"').casefold()
                        break
                # When user-data-dir identifies the configured profile store,
                # Chrome may omit the default profile switch. In that case the
                # browser session selector remains authoritative.
                if explicit_profile and explicit_profile != profile:
                    continue
                if not explicit_profile and not user_data:
                    continue
            if debug_port and f"--remote-debugging-port={debug_port}".casefold() not in folded:
                continue
            return True
    except Exception:
        return False
    return False


def resolve_app_executable(app: str) -> Path | None:
    """Resolve a semantic GUI app name without accepting an executable path."""
    if not isinstance(app, str) or not app.strip() or _blocked_app_name(app):
        return None

    normalized = _normalize_app_name(app)
    spec = APP_REGISTRY.get(normalized)
    if spec:
        windows_candidates = spec.get("windows_candidates", []) if sys.platform == "win32" else []
        for raw in windows_candidates:
            candidate = Path(os.path.expandvars(raw))
            if candidate.is_file():
                return candidate
        for name in spec.get("executables", []):
            found = shutil.which(name)
            if found:
                return Path(found)

    registry_candidate = _registry_app_executable(normalized)
    if registry_candidate:
        return registry_candidate

    for candidate in _common_windows_candidates(normalized):
        return candidate

    # PATH is useful on non-Windows and for portable desktop applications.
    for alias in _app_aliases(normalized):
        found = shutil.which(alias) or shutil.which(f"{alias}.exe")
        if found:
            return Path(found).resolve()
    return None


def _dynamic_app_spec(app: str, executable: Path) -> dict[str, Any]:
    process = executable.name
    display = " ".join(str(app).strip().split())
    return {
        "executables": [process],
        "windows_candidates": [str(executable)],
        "real_binary_relative": [],
        "process_names": [process],
        "window_title_contains": display if len(display) >= 3 else None,
    }


def _chrome_agent_profile(
    policy: Policy,
) -> Path:
    """
    Return an isolated Chrome profile for the agent.

    The agent must not silently inherit the user's authenticated Default Chrome
    profile unless that is implemented as an explicit higher-level permission.
    """
    profile = (
        Path(policy.workspace)
        / ".hermes"
        / "chrome-profile"
    )

    resolved = policy.resolve_write_path(
        profile
    )

    resolved.mkdir(
        parents=True,
        exist_ok=True,
    )

    return resolved


@_guard
def launch_app(
    policy: Policy,
    action: Action,
) -> ActionResult:
    """
    Launch a registered application through the OS process layer.

    The caller supplies a semantic application name rather than an arbitrary
    executable path.

    For Chrome, an ``open_path`` may be either a permitted local filesystem
    path or an HTTP(S) URL. URLs are validated by the network policy and are
    never passed through filesystem path resolution.
    """
    started = time.time()

    app = action.params["app"]

    if not isinstance(app, str) or not app.strip():
        return _result(action, started, ok=False, error="launch_app requires a non-empty app name",
                       failure_class=FailureClass.PRECONDITION_FAILED)
    if _blocked_app_name(app):
        return _result(action, started, ok=False,
                       error=f"shell/interpreter application is blocked: {app!r}",
                       failure_class=FailureClass.PERMISSION_DENIED)

    chrome_config = _chrome_configuration() if _normalize_app_name(app) == "chrome" else {}
    configured_executable = chrome_config.get("executable")
    if configured_executable:
        executable = Path(os.path.expandvars(configured_executable)).expanduser()
        if not executable.is_file():
            return _result(
                action, started, ok=False,
                error=f"configured Chrome executable does not exist: {configured_executable!r}",
                failure_class=FailureClass.ENVIRONMENT,
                detail={"app": "chrome", "executable": configured_executable},
            )
    else:
        executable = resolve_app_executable(app)

    if executable is None:
        return _result(
            action,
            started,
            ok=False,
            error=(
                f"could not resolve installed application {app!r} on this host"
            ),
            failure_class=FailureClass.ENVIRONMENT,
            detail={"app": app},
        )

    argv = [str(executable)]

    # Dynamically resolved GUI applications still pass through the deterministic
    # application-specific policy gate. This never widens run_command.
    executable = policy.check_app_executable(executable)

    normalized_app = _normalize_app_name(app)
    if normalized_app not in APP_REGISTRY:
        APP_REGISTRY[normalized_app] = _dynamic_app_spec(app, executable)
        app = normalized_app

    if app == "chrome":
        # Use the deployment's configured real Chrome profile when supplied.
        # Never synthesize a username, user-data directory, or profile name.
        argv.extend([
            "--no-first-run",
            "--no-default-browser-check",
        ])
        if chrome_config.get("user_data"):
            argv.append(f"--user-data-dir={chrome_config['user_data']}")
        if chrome_config.get("profile"):
            argv.append(f"--profile-directory={chrome_config['profile']}")
        if chrome_config.get("debug_port"):
            argv.append(f"--remote-debugging-port={chrome_config['debug_port']}")

    # launch_app is intentionally application-only. File paths and URLs have
    # their own semantic action kinds so the planner cannot accidentally turn a
    # target into a different operation.
    if action.params.get("open_path") or action.params.get("url"):
        return _result(
            action, started, ok=False,
            error="launch_app accepts only an app name; use open_file or open_url for targets",
            failure_class=FailureClass.PRECONDITION_FAILED,
        )

    reused_existing = app == "chrome" and bool(chrome_config) and _chrome_process_matches_configuration(chrome_config)
    proc = None
    if not reused_existing:
        proc = subprocess.Popen(
            argv,
            cwd=str(policy.workspace),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            shell=False,
        )

    settle = _bounded_float(
        action.params.get(
            "settle_s",
            6.0,
        ),
        default=6.0,
        minimum=0.0,
        maximum=_MAX_APP_SETTLE_SECONDS,
    )

    if settle > 0:
        time.sleep(settle)

    detail: dict[str, Any] = {
        "app": app,
        "executable": str(executable),
        "launcher_pid": proc.pid if proc is not None else None,
        "launch_requested": not reused_existing,
        "reused_existing": reused_existing,
        "settled_s": settle,
    }

    if app == "chrome":
        detail["chrome_configuration"] = dict(chrome_config)

    return _result(
        action,
        started,
        ok=True,
        detail=detail,
    )


@_guard
def type_text(policy: Policy, action: Action) -> ActionResult:
    """Type text into a semantically identified desktop application window."""
    started = time.time()
    app = str(action.params.get("app", "")).strip()
    text = action.params.get("text")
    if not app:
        return _result(action, started, ok=False, error="type_text requires a target app", failure_class=FailureClass.PRECONDITION_FAILED)
    if not isinstance(text, str) or not text:
        return _result(action, started, ok=False, error="type_text requires non-empty text", failure_class=FailureClass.PRECONDITION_FAILED)
    from .platform_window import get_backend
    backend = get_backend()
    if not backend.available:
        return _result(action, started, ok=False, error=f"window backend {backend.name!r} unavailable", failure_class=FailureClass.UNKNOWN)
    normalized = _normalize_app_name(app)
    spec = APP_REGISTRY.get(normalized)
    if spec is None:
        executable = resolve_app_executable(app)
        if executable is not None:
            spec = _dynamic_app_spec(app, executable)
    if not spec:
        return _result(action, started, ok=False, error=f"cannot resolve semantic application {app!r}", failure_class=FailureClass.PRECONDITION_FAILED)
    names = {str(n).casefold() for n in (spec.get("process_names") or [])}
    needle = str(spec.get("window_title_contains") or app).casefold()
    candidates = []
    for window in backend.list_windows():
        if not window.visible:
            continue
        if names and window.pid:
            try:
                import psutil
                proc_name = (psutil.Process(window.pid).name() or "").casefold()
            except Exception:
                proc_name = ""
            if proc_name not in names and needle not in window.title.casefold():
                continue
        elif needle not in window.title.casefold():
            continue
        candidates.append(window)
    if not candidates:
        return _result(action, started, ok=False, error=f"no visible {app} window found", failure_class=FailureClass.PRECONDITION_FAILED)
    # Multiple instances are valid. Prefer the currently focused matching
    # window, otherwise use the first semantic match returned by the backend.
    # No coordinates or OCR are involved.
    focused = [candidate for candidate in candidates if candidate.focused]
    window = focused[0] if focused else candidates[0]
    if not backend.focus(window.handle):
        return _result(action, started, ok=False, error=f"could not focus {app} window", failure_class=FailureClass.ACTION_FAILED)
    if not backend.type_text(text):
        return _result(action, started, ok=False, error=f"platform text input failed for {app}", failure_class=FailureClass.ACTION_FAILED)
    return _result(action, started, ok=True, detail={"app": app, "window_handle": window.handle, "typed_length": len(text)})


@_guard
def open_url(policy: Policy, action: Action) -> ActionResult:
    """Open an HTTP(S) URL using the user's default browser.

    This is deliberately separate from ``launch_app`` and ``run_command``. The
    URL is validated as browser navigation, then handed to the platform's URL
    association without a shell command. Independent verification remains the
    task-level verifier's responsibility.
    """
    started = time.time()
    url = policy.check_browser_url(str(action.params.get("url", "")))

    if sys.platform == "win32":
        os.startfile(url)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, shell=False)
    else:
        subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, shell=False)

    settle = _bounded_float(action.params.get("settle_s", 3.0), default=3.0,
                            minimum=0.0, maximum=_MAX_OPEN_SETTLE_SECONDS)
    if settle > 0:
        time.sleep(settle)
    return _result(action, started, ok=True, detail={"url": url, "settled_s": settle})


# Semantic action kind -> executor.
DISPATCH: dict[
    str,
    Callable[
        [Policy, Action],
        ActionResult,
    ],
] = {
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
    "type_text": type_text,
    "open_url": open_url,
}


def execute(
    policy: Policy,
    action: Action,
) -> ActionResult:
    """
    Dispatch a semantic action, refusing unknown action kinds.
    """
    handler = DISPATCH.get(
        action.kind
    )

    if handler is None:
        return ActionResult(
            action=action,
            ok=False,
            error=(
                f"unknown action kind "
                f"{action.kind!r}; "
                f"allowed: {sorted(DISPATCH)}"
            ),
            failure_class=(
                FailureClass.PRECONDITION_FAILED
            ),
        )

    return handler(
        policy,
        action,
    )