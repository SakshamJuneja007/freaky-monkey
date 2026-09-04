"""Independent verification layer (plan S8).

Two rules make this layer worth having:

1. **Verifiers never read** :class:`~agent_control.types.ActionResult`. They
   re-observe the world from scratch. The action layer's belief that it
   succeeded is not evidence, so it is not even in scope here.
2. **UNKNOWN is not success.** A check that could not be evaluated -- window
   backend unavailable, interpreter unreadable -- returns UNKNOWN, which
   aggregates to a non-PASS verdict. Coercing UNKNOWN to PASS is exactly the
   shallow-verification failure the plan's kill condition (S22) warns about.

The gap between "reported success" and these verdicts is the false-success rate
in the metrics table (plan S19).
"""

from __future__ import annotations

from typing import Any, Iterable

from . import observe
from .os_tools import APP_REGISTRY
from .policy import Policy
from .trace import Trace
from .types import Check, Observation, VerificationResult, Verdict


def _record(trace: Trace | None, obs: Observation, purpose: str) -> Observation:
    return trace.observation(obs, purpose=purpose) if trace else obs


def _unknown(name: str, obs: Observation, reason: str) -> Check:
    """A reading that failed -> UNKNOWN, carrying the reader's own error."""
    return Check(
        name=name,
        verdict=Verdict.UNKNOWN,
        evidence={"observation": obs.to_json()},
        reason=f"{reason}: {obs.error}",
    )


# -- filesystem ------------------------------------------------------------
def verify_dir(policy: Policy, path: str, *, trace: Trace | None = None,
               min_entries: int | None = None) -> VerificationResult:
    """Directory exists and is a directory (plan S17 row 2)."""
    resolved = policy.resolve_read_path(path)
    obs = _record(trace, observe.dir_state(resolved), f"verify_dir({resolved})")
    checks: list[Check] = []
    if not obs.ok:
        checks.append(_unknown("dir_readable", obs, "could not read directory state"))
        return VerificationResult(checks=checks, label=f"dir:{resolved}")

    value = obs.value or {}
    exists, is_dir = bool(value.get("exists")), bool(value.get("is_dir"))
    checks.append(Check(
        name="dir_exists",
        verdict=Verdict.PASS if exists and is_dir else Verdict.FAIL,
        evidence={"path": str(resolved), "exists": exists, "is_dir": is_dir},
        reason="" if exists and is_dir else "directory absent or not a directory",
    ))
    if min_entries is not None:
        count = int(value.get("entry_count", 0))
        checks.append(Check(
            name="dir_min_entries",
            verdict=Verdict.PASS if count >= min_entries else Verdict.FAIL,
            evidence={"entry_count": count, "min_entries": min_entries},
            reason="" if count >= min_entries else f"{count} entries < {min_entries}",
        ))
    return VerificationResult(checks=checks, label=f"dir:{resolved}")


def verify_file(policy: Policy, path: str, *, trace: Trace | None = None,
                expected_sha256: str | None = None,
                must_contain: Iterable[str] = (),
                min_bytes: int | None = None) -> VerificationResult:
    """File exists at the right path with the right *content* (plan S17 row 3).

    Content, not just presence: a truncated or error-page download that lands at
    the right path is the classic false success this check exists to catch.
    """
    resolved = policy.resolve_read_path(path)
    obs = _record(trace, observe.file_state(resolved), f"verify_file({resolved})")
    checks: list[Check] = []
    if not obs.ok:
        checks.append(_unknown("file_readable", obs, "could not read file state"))
        return VerificationResult(checks=checks, label=f"file:{resolved}")

    value = obs.value or {}
    present = bool(value.get("exists")) and bool(value.get("is_file"))
    checks.append(Check(
        name="file_exists",
        verdict=Verdict.PASS if present else Verdict.FAIL,
        evidence={"path": str(resolved), "exists": value.get("exists"),
                  "is_file": value.get("is_file"), "size": value.get("size")},
        reason="" if present else "file absent or not a regular file",
    ))
    if not present:
        return VerificationResult(checks=checks, label=f"file:{resolved}")

    if min_bytes is not None:
        size = int(value.get("size", 0))
        checks.append(Check(
            name="file_min_bytes",
            verdict=Verdict.PASS if size >= min_bytes else Verdict.FAIL,
            evidence={"size": size, "min_bytes": min_bytes},
            reason="" if size >= min_bytes else f"{size} bytes < {min_bytes}",
        ))
    if expected_sha256:
        actual = value.get("sha256")
        if actual is None:
            checks.append(Check(
                name="file_sha256", verdict=Verdict.UNKNOWN,
                evidence={"expected": expected_sha256},
                reason="file too large to hash; cannot confirm content",
            ))
        else:
            match = actual.lower() == expected_sha256.lower()
            checks.append(Check(
                name="file_sha256",
                verdict=Verdict.PASS if match else Verdict.FAIL,
                evidence={"expected": expected_sha256, "actual": actual},
                reason="" if match else "content hash mismatch",
            ))
    return VerificationResult(
        checks=checks + _content_checks(resolved, must_contain),
        label=f"file:{resolved}",
    )


def _content_checks(resolved, must_contain: Iterable[str]) -> list[Check]:
    needles = list(must_contain)
    if not needles:
        return []
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [Check(
            name="file_contains", verdict=Verdict.UNKNOWN,
            evidence={"needles": needles}, reason=f"unreadable: {exc}",
        )]
    missing = [n for n in needles if n not in text]
    return [Check(
        name="file_contains",
        verdict=Verdict.PASS if not missing else Verdict.FAIL,
        evidence={"needles": needles, "missing": missing},
        reason="" if not missing else f"missing {missing}",
    )]


# -- application -----------------------------------------------------------
def verify_app_running(policy: Policy, app: str, *,
                       trace: Trace | None = None) -> VerificationResult:
    """Process exists **and** expected window state (plan S17 row 1).

    Process and window are separate checks on purpose. A process that started and
    immediately crashed back to a splash screen passes the first and fails the
    second, and the trace shows which.
    """
    spec = APP_REGISTRY.get(app, {})
    checks: list[Check] = []
    names = spec.get("process_names") or [app]

    proc_obs = _record(
        trace,
        observe.process_state(name_in=names),
        f"verify_app_running:process({app})",
    )
    if not proc_obs.ok:
        checks.append(_unknown("app_process", proc_obs, "could not enumerate processes"))
    else:
        running = bool((proc_obs.value or {}).get("running"))
        checks.append(Check(
            name="app_process",
            verdict=Verdict.PASS if running else Verdict.FAIL,
            evidence={"app": app, "process_names": names,
                      "count": (proc_obs.value or {}).get("count"),
                      "pids": (proc_obs.value or {}).get("pids")},
            reason="" if running else f"no process named one of {names}",
        ))
    return VerificationResult(
        checks=checks + _window_checks(app, spec, trace), label=f"app:{app}",
    )


def _window_checks(app: str, spec: dict[str, Any], trace: Trace | None) -> list[Check]:
    needle = spec.get("window_title_contains")
    if not needle:
        return []
    obs = _record(
        trace, observe.window_state(title_contains=needle), f"verify_app_running:window({app})"
    )
    if not obs.ok:
        # Backend unavailable (e.g. headless, or Linux without wmctrl). UNKNOWN,
        # never PASS -- this is the measured accessibility/window gap (plan S27).
        return [_unknown("app_window", obs, "window state unavailable")]
    present = bool((obs.value or {}).get("present"))
    return [Check(
        name="app_window",
        verdict=Verdict.PASS if present else Verdict.FAIL,
        evidence={"title_contains": needle, "count": (obs.value or {}).get("count"),
                  "backend": (obs.value or {}).get("backend")},
        reason="" if present else f"no window titled like {needle!r}",
    )]


#: Below this a filename stem cannot identify a window. ``a.txt`` would match any
#: title containing an "a", and a check that passes on a coincidence is worth less
#: than one that admits it could not tell.
MIN_WINDOW_STEM_CHARS = 3


def verify_opened(policy: Policy, path: str, *, needle: str | None = None,
                  trace: Trace | None = None) -> VerificationResult:
    """A *visible* window titled like the file: evidence that it is on screen.

    This is the check that separates "a launch was issued" from "something opened".
    ``open_file`` returns ok as soon as ``os.startfile`` returns, which is before
    any application has drawn anything, so the action layer's belief is not
    evidence here -- as everywhere in this module, the world is re-read.

    Absence is UNKNOWN rather than FAIL, and the distinction is the honest one:
    a media player may title its window ``main1.mp4 - VLC media player`` or plain
    ``Media Player`` -- which is what Windows 11's own default ``.mp4`` handler
    does, measured on this machine -- so a missing title match genuinely does not
    distinguish "the open failed" from "this handler does not name the file".
    UNKNOWN aggregates to a non-PASS verdict, so nothing is claimed either way.

    Only visible windows count. The Win32 backend reports every top-level window,
    and most of them on a real desktop are invisible helpers (``DDE Server
    Window``, ``System tray overflow window.``); a substring hit on one of those
    would be a pass with nothing behind it.

    ``needle`` overrides what is looked for. The default -- the filename stem --
    is right for a file, because a handler that names the file in its title names
    it without the extension. It is wrong for a directory: ``my.project`` has the
    stem ``my``, which would match almost any window on the desktop. Callers
    opening a folder pass the folder's whole name.
    """
    resolved = policy.resolve_read_path(path)
    stem = needle if needle is not None else resolved.stem
    label = f"opened:{resolved.name}"

    if len(stem) < MIN_WINDOW_STEM_CHARS:
        return VerificationResult(label=label, checks=[Check(
            name="viewer_window",
            verdict=Verdict.UNKNOWN,
            evidence={"stem": stem, "min_chars": MIN_WINDOW_STEM_CHARS},
            reason=(f"{stem!r} is too short to identify a window by title; "
                    "cannot tell whether anything opened"),
        )])

    obs = _record(trace, observe.window_state(title_contains=stem),
                  f"verify_opened({stem})")

    if not obs.ok:
        return VerificationResult(
            label=label,
            checks=[_unknown("viewer_window", obs, "window state unavailable")],
        )

    value = obs.value or {}
    visible = [w for w in (value.get("windows") or []) if w.get("visible")]

    if visible:
        return VerificationResult(label=label, checks=[Check(
            name="viewer_window",
            verdict=Verdict.PASS,
            evidence={"title_contains": stem, "visible_count": len(visible),
                      "titles": [w.get("title") for w in visible[:3]],
                      "backend": value.get("backend")},
        )])

    #: Which of the two worlds this UNKNOWN is in ("nothing opened" or "the
    #: handler does not name the file") is decided by one more observation, so the
    #: report can name what *is* on screen instead of leaving a person to guess.
    #: Measured on this machine: Windows 11's default ``.mp4`` handler titles its
    #: window plain ``Media Player``, so this branch is the normal case for a
    #: successful open, not an exotic one. It stays UNKNOWN either way -- a visible
    #: window that does not name the file is not evidence about *this* file.
    others = _record(trace, observe.window_state(), f"verify_opened({stem}) context")
    onscreen = [
        str(w.get("title") or "")
        for w in ((others.value or {}).get("windows") or [])
        if w.get("visible") and str(w.get("title") or "").strip()
    ][:5]

    return VerificationResult(label=label, checks=[Check(
        name="viewer_window",
        verdict=Verdict.UNKNOWN,
        evidence={"title_contains": stem, "matched_count": value.get("count"),
                  "visible_count": 0, "backend": value.get("backend"),
                  "visible_titles": onscreen},
        reason=(f"no visible window titled like {stem!r}: either nothing opened, "
                "or the application that handles this file type does not put the "
                "filename in its window title"
                + (f" (visible windows include {', '.join(onscreen)})"
                   if onscreen else "")),
    )])


# -- python environment ----------------------------------------------------
def verify_packages(policy: Policy, venv_dir: str, packages: Iterable[str], *,
                    trace: Trace | None = None) -> VerificationResult:
    """Exit code is not enough: import + version inside the target env (plan S17 row 4)."""
    resolved = policy.resolve_read_path(venv_dir)
    wanted = list(packages)
    checks: list[Check] = []

    env_obs = _record(trace, observe.python_env_state(resolved), f"verify_env({resolved})")
    if not env_obs.ok:
        checks.append(_unknown("venv_interpreter", env_obs, "could not probe interpreter"))
        return VerificationResult(checks=checks, label=f"env:{resolved}")

    env = env_obs.value or {}
    isolated = bool(env.get("interpreter_exists")) and bool(env.get("is_isolated"))
    checks.append(Check(
        name="venv_interpreter",
        verdict=Verdict.PASS if isolated else Verdict.FAIL,
        evidence={k: env.get(k) for k in ("interpreter_exists", "is_isolated", "version", "prefix")},
        reason="" if isolated else "no isolated interpreter at that path",
    ))
    if not isolated:
        return VerificationResult(checks=checks, label=f"env:{resolved}")
    return VerificationResult(
        checks=checks + _package_checks(resolved, wanted, trace), label=f"env:{resolved}",
    )


def _package_checks(resolved, wanted: list[str], trace: Trace | None) -> list[Check]:
    if not wanted:
        return []
    obs = _record(trace, observe.package_state(resolved, wanted), f"verify_packages({wanted})")
    if not obs.ok:
        return [_unknown("packages_importable", obs, "could not probe packages")]
    results = (obs.value or {}).get("packages", {})
    checks: list[Check] = []
    for name in wanted:
        entry = results.get(name, {})
        ok = bool(entry.get("import_ok"))
        checks.append(Check(
            name=f"package:{name}",
            verdict=Verdict.PASS if ok else Verdict.FAIL,
            evidence={"import_ok": ok, "version": entry.get("version")},
            reason="" if ok else f"not importable in target env: {entry.get('error')}",
        ))
    return checks


# -- composition -----------------------------------------------------------
def combine(label: str, parts: Iterable[VerificationResult]) -> VerificationResult:
    """Merge sub-verifications, namespacing check names so evidence stays legible.

    Used for the multi-step task, where every checkpoint *and* the final goal are
    verified independently (plan S8 last bullet).
    """
    merged: list[Check] = []
    for part in parts:
        for check in part.checks:
            merged.append(Check(
                name=f"{part.label}/{check.name}" if part.label else check.name,
                verdict=check.verdict,
                evidence=check.evidence,
                reason=check.reason,
            ))
    return VerificationResult(checks=merged, label=label)
