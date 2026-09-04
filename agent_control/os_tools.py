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