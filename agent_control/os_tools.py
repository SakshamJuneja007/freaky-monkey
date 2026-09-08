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
import shutil
import subprocess
import sys
import tempfile
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import httpx

from .observe import venv_python
from .policy import Policy
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

    if not target.is_file():
        return _result(
            action,
            started,
            ok=False,
            error=f"path is not a file: {target}",
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


def resolve_app_executable(
    app: str,
) -> Path | None:
    """
    Find a registered app's launcher, or None if it is not installed.
    """
    spec = APP_REGISTRY.get(app)

    if not spec:
        return None

    windows_candidates = (
        spec.get(
            "windows_candidates",
            [],
        )
        if sys.platform == "win32"
        else []
    )

    for raw in windows_candidates:
        candidate = Path(
            os.path.expandvars(raw)
        )

        if candidate.exists():
            return candidate

    for name in spec.get(
        "executables",
        [],
    ):
        found = shutil.which(name)

        if found:
            return _prefer_real_binary(
                Path(found),
                spec,
            )

    return None


def _prefer_real_binary(
    shim: Path,
    spec: dict[str, Any],
) -> Path:
    """
    Prefer the real executable over a launcher shim when available.
    """
    if shim.suffix.lower() not in {
        ".cmd",
        ".bat",
        "",
    }:
        return shim

    for relative in spec.get(
        "real_binary_relative",
        [],
    ):
        candidate = (
            shim.parent / relative
        ).resolve()

        if (
            candidate.exists()
            and candidate.suffix.lower()
            == ".exe"
        ):
            return candidate

    return shim


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

    if app not in APP_REGISTRY:
        return _result(
            action,
            started,
            ok=False,
            error=(
                f"unregistered app {app!r}"
            ),
            failure_class=FailureClass.PRECONDITION_FAILED,
        )

    executable = resolve_app_executable(
        app
    )

    if executable is None:
        return _result(
            action,
            started,
            ok=False,
            error=(
                f"{app!r} is not installed "
                f"on this host"
            ),
            failure_class=FailureClass.ENVIRONMENT,
            detail={"app": app},
        )

    argv = [str(executable)]

    # Registered applications still pass through executable policy validation.
    policy.check_executable(argv)

    if app == "chrome":
        chrome_profile = (
            _chrome_agent_profile(policy)
        )

        argv.extend(
            [
                (
                    "--user-data-dir="
                    f"{chrome_profile}"
                ),
                "--no-first-run",
                "--no-default-browser-check",
            ]
        )

    open_path = action.params.get(
        "open_path"
    )

    url = action.params.get(
        "url"
    )

    # The action must not contain two competing launch targets.
    if open_path and url:
        return _result(
            action,
            started,
            ok=False,
            error=(
                "provide either 'url' or 'open_path', "
                "not both"
            ),
            failure_class=FailureClass.PRECONDITION_FAILED,
        )

    if url:
        if app != "chrome":
            return _result(
                action,
                started,
                ok=False,
                error=(
                    f"app {app!r} does not support "
                    f"URL launching"
                ),
                failure_class=FailureClass.PRECONDITION_FAILED,
            )

        argv.append(
            policy.check_url(
                str(url)
            )
        )

    elif open_path:
        target = str(open_path)

        # Backwards compatibility: previous planners may still send URLs in
        # ``open_path``. Detect HTTP(S) explicitly and keep the target out of
        # filesystem path resolution.
        if _is_http_url(target):
            if app != "chrome":
                return _result(
                    action,
                    started,
                    ok=False,
                    error=(
                        f"app {app!r} does not support "
                        f"URL launching"
                    ),
                    failure_class=FailureClass.PRECONDITION_FAILED,
                )

            argv.append(
                policy.check_url(target)
            )

        else:
            argv.append(
                str(
                    policy.resolve_read_path(
                        target
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
        "launcher_pid": proc.pid,
        "launch_requested": True,
        "settled_s": settle,
    }

    if app == "chrome":
        detail["profile"] = str(
            _chrome_agent_profile(policy)
        )

    if url:
        detail["url"] = str(url)

    elif open_path and _is_http_url(str(open_path)):
        detail["url"] = str(open_path)

    elif open_path:
        detail["open_path"] = str(open_path)

    return _result(
        action,
        started,
        ok=True,
        detail=detail,
    )


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