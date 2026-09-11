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


def _record(
    trace: Trace | None,
    obs: Observation,
    purpose: str,
) -> Observation:
    return trace.observation(obs, purpose=purpose) if trace else obs


def _unknown(
    name: str,
    obs: Observation,
    reason: str,
) -> Check:
    """A reading that failed -> UNKNOWN, carrying the reader's own error."""
    return Check(
        name=name,
        verdict=Verdict.UNKNOWN,
        evidence={"observation": obs.to_json()},
        reason=f"{reason}: {obs.error}",
    )


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

def verify_dir(
    policy: Policy,
    path: str,
    *,
    trace: Trace | None = None,
    min_entries: int | None = None,
) -> VerificationResult:
    """Directory exists and is a directory."""
    resolved = policy.resolve_read_path(path)

    obs = _record(
        trace,
        observe.dir_state(resolved),
        f"verify_dir({resolved})",
    )

    checks: list[Check] = []

    if not obs.ok:
        checks.append(
            _unknown(
                "dir_readable",
                obs,
                "could not read directory state",
            )
        )
        return VerificationResult(
            checks=checks,
            label=f"dir:{resolved}",
        )

    value = obs.value or {}

    exists = bool(value.get("exists"))
    is_dir = bool(value.get("is_dir"))

    checks.append(
        Check(
            name="dir_exists",
            verdict=(
                Verdict.PASS
                if exists and is_dir
                else Verdict.FAIL
            ),
            evidence={
                "path": str(resolved),
                "exists": exists,
                "is_dir": is_dir,
            },
            reason=(
                ""
                if exists and is_dir
                else "directory absent or not a directory"
            ),
        )
    )

    if min_entries is not None:
        count = int(value.get("entry_count", 0))

        checks.append(
            Check(
                name="dir_min_entries",
                verdict=(
                    Verdict.PASS
                    if count >= min_entries
                    else Verdict.FAIL
                ),
                evidence={
                    "entry_count": count,
                    "min_entries": min_entries,
                },
                reason=(
                    ""
                    if count >= min_entries
                    else f"{count} entries < {min_entries}"
                ),
            )
        )

    return VerificationResult(
        checks=checks,
        label=f"dir:{resolved}",
    )


def verify_file(
    policy: Policy,
    path: str,
    *,
    trace: Trace | None = None,
    expected_sha256: str | None = None,
    must_contain: Iterable[str] = (),
    min_bytes: int | None = None,
) -> VerificationResult:
    """Verify a file exists and optionally verify its content."""

    resolved = policy.resolve_read_path(path)

    obs = _record(
        trace,
        observe.file_state(resolved),
        f"verify_file({resolved})",
    )

    checks: list[Check] = []

    if not obs.ok:
        checks.append(
            _unknown(
                "file_readable",
                obs,
                "could not read file state",
            )
        )
        return VerificationResult(
            checks=checks,
            label=f"file:{resolved}",
        )

    value = obs.value or {}

    present = (
        bool(value.get("exists"))
        and bool(value.get("is_file"))
    )

    checks.append(
        Check(
            name="file_exists",
            verdict=(
                Verdict.PASS
                if present
                else Verdict.FAIL
            ),
            evidence={
                "path": str(resolved),
                "exists": value.get("exists"),
                "is_file": value.get("is_file"),
                "size": value.get("size"),
            },
            reason=(
                ""
                if present
                else "file absent or not a regular file"
            ),
        )
    )

    if not present:
        return VerificationResult(
            checks=checks,
            label=f"file:{resolved}",
        )

    if min_bytes is not None:
        size = int(value.get("size", 0))

        checks.append(
            Check(
                name="file_min_bytes",
                verdict=(
                    Verdict.PASS
                    if size >= min_bytes
                    else Verdict.FAIL
                ),
                evidence={
                    "size": size,
                    "min_bytes": min_bytes,
                },
                reason=(
                    ""
                    if size >= min_bytes
                    else f"{size} bytes < {min_bytes}"
                ),
            )
        )

    if expected_sha256:
        actual = value.get("sha256")

        if actual is None:
            checks.append(
                Check(
                    name="file_sha256",
                    verdict=Verdict.UNKNOWN,
                    evidence={
                        "expected": expected_sha256,
                    },
                    reason=(
                        "file too large to hash; "
                        "cannot confirm content"
                    ),
                )
            )
        else:
            match = (
                actual.lower()
                == expected_sha256.lower()
            )

            checks.append(
                Check(
                    name="file_sha256",
                    verdict=(
                        Verdict.PASS
                        if match
                        else Verdict.FAIL
                    ),
                    evidence={
                        "expected": expected_sha256,
                        "actual": actual,
                    },
                    reason=(
                        ""
                        if match
                        else "content hash mismatch"
                    ),
                )
            )

    return VerificationResult(
        checks=checks
        + _content_checks(
            resolved,
            must_contain,
        ),
        label=f"file:{resolved}",
    )


def _content_checks(
    resolved,
    must_contain: Iterable[str],
) -> list[Check]:
    needles = list(must_contain)

    if not needles:
        return []

    try:
        text = resolved.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return [
            Check(
                name="file_contains",
                verdict=Verdict.UNKNOWN,
                evidence={
                    "needles": needles,
                },
                reason=f"unreadable: {exc}",
            )
        ]

    missing = [
        needle
        for needle in needles
        if needle not in text
    ]

    return [
        Check(
            name="file_contains",
            verdict=(
                Verdict.PASS
                if not missing
                else Verdict.FAIL
            ),
            evidence={
                "needles": needles,
                "missing": missing,
            },
            reason=(
                ""
                if not missing
                else f"missing {missing}"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def verify_app_running(
    policy: Policy,
    app: str,
    *,
    trace: Trace | None = None,
    window_title_contains: str | None = None,
) -> VerificationResult:
    """Verify both the application process and its visible window."""

    spec = APP_REGISTRY.get(app, {})

    checks: list[Check] = []

    names = spec.get("process_names") or [app]

    proc_obs = _record(
        trace,
        observe.process_state(
            name_in=names,
        ),
        f"verify_app_running:process({app})",
    )

    if not proc_obs.ok:
        checks.append(
            _unknown(
                "app_process",
                proc_obs,
                "could not enumerate processes",
            )
        )
    else:
        running = bool(
            (proc_obs.value or {}).get("running")
        )

        checks.append(
            Check(
                name="app_process",
                verdict=(
                    Verdict.PASS
                    if running
                    else Verdict.FAIL
                ),
                evidence={
                    "app": app,
                    "process_names": names,
                    "count": (
                        proc_obs.value or {}
                    ).get("count"),
                    "pids": (
                        proc_obs.value or {}
                    ).get("pids"),
                },
                reason=(
                    ""
                    if running
                    else (
                        "no process named one of "
                        f"{names}"
                    )
                ),
            )
        )

    return VerificationResult(
        checks=checks
        + _window_checks(
            app,
            spec,
            trace,
            window_title_contains=window_title_contains,
        ),
        label=f"app:{app}",
    )


def _window_checks(
    app: str,
    spec: dict[str, Any],
    trace: Trace | None,
    *,
    window_title_contains: str | None = None,
) -> list[Check]:

    needle = (
        window_title_contains
        or spec.get("window_title_contains")
    )

    if not needle:
        return []

    obs = _record(
        trace,
        observe.window_state(
            title_contains=needle,
        ),
        f"verify_app_running:window({app})",
    )

    if not obs.ok:
        return [
            _unknown(
                "app_window",
                obs,
                "window state unavailable",
            )
        ]

    present = bool(
        (obs.value or {}).get("present")
    )

    return [
        Check(
            name="app_window",
            verdict=(
                Verdict.PASS
                if present
                else Verdict.FAIL
            ),
            evidence={
                "title_contains": needle,
                "count": (
                    obs.value or {}
                ).get("count"),
                "backend": (
                    obs.value or {}
                ).get("backend"),
            },
            reason=(
                ""
                if present
                else (
                    f"no window titled like {needle!r}"
                )
            ),
        )
    ]


#: Below this a filename stem cannot reliably identify a window.
MIN_WINDOW_STEM_CHARS = 3


def verify_opened(
    policy: Policy,
    path: str,
    *,
    needle: str | None = None,
    trace: Trace | None = None,
) -> VerificationResult:
    """Verify that a visible window appears to represent the opened target.

    A successful ``os.startfile`` call only proves that Windows accepted the
    request. It does not prove that the target actually appeared on screen.

    Therefore this verifier independently inspects the visible windows.

    Missing title evidence is deliberately UNKNOWN rather than PASS.
    """

    resolved = policy.resolve_read_path(path)

    stem = (
        needle
        if needle is not None
        else resolved.stem
    )

    label = f"opened:{resolved.name}"

    if len(stem) < MIN_WINDOW_STEM_CHARS:
        return VerificationResult(
            label=label,
            checks=[
                Check(
                    name="viewer_window",
                    verdict=Verdict.UNKNOWN,
                    evidence={
                        "stem": stem,
                        "min_chars": MIN_WINDOW_STEM_CHARS,
                    },
                    reason=(
                        f"{stem!r} is too short to identify "
                        "a window by title; cannot tell "
                        "whether anything opened"
                    ),
                )
            ],
        )

    obs = _record(
        trace,
        observe.window_state(
            title_contains=stem,
        ),
        f"verify_opened({stem})",
    )

    if not obs.ok:
        return VerificationResult(
            label=label,
            checks=[
                _unknown(
                    "viewer_window",
                    obs,
                    "window state unavailable",
                )
            ],
        )

    value = obs.value or {}

    visible = [
        window
        for window in (
            value.get("windows") or []
        )
        if window.get("visible")
    ]

    if visible:
        return VerificationResult(
            label=label,
            checks=[
                Check(
                    name="viewer_window",
                    verdict=Verdict.PASS,
                    evidence={
                        "title_contains": stem,
                        "visible_count": len(visible),
                        "titles": [
                            window.get("title")
                            for window in visible[:3]
                        ],
                        "backend": value.get(
                            "backend"
                        ),
                    },
                )
            ],
        )

    # No exact filename/title match.
    #
    # Take another independent window observation so that the result can
    # tell us what is actually visible. This remains UNKNOWN rather than
    # PASS because the visible application may simply use a generic title.

    others = _record(
        trace,
        observe.window_state(),
        f"verify_opened({stem}) context",
    )

    onscreen = [
        str(window.get("title") or "")
        for window in (
            (others.value or {}).get("windows")
            or []
        )
        if (
            window.get("visible")
            and str(window.get("title") or "").strip()
        )
    ][:5]

    return VerificationResult(
        label=label,
        checks=[
            Check(
                name="viewer_window",
                verdict=Verdict.UNKNOWN,
                evidence={
                    "title_contains": stem,
                    "matched_count": value.get(
                        "count"
                    ),
                    "visible_count": 0,
                    "backend": value.get(
                        "backend"
                    ),
                    "visible_titles": onscreen,
                },
                reason=(
                    f"no visible window titled like "
                    f"{stem!r}: either nothing opened, "
                    "or the application that handles this "
                    "file type does not put the filename "
                    "in its window title"
                    + (
                        " (visible windows include "
                        + ", ".join(onscreen)
                        + ")"
                        if onscreen
                        else ""
                    )
                ),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Python environments
# ---------------------------------------------------------------------------

def verify_packages(
    policy: Policy,
    venv_dir: str,
    packages: Iterable[str],
    *,
    trace: Trace | None = None,
) -> VerificationResult:
    """Verify an isolated Python environment."""

    resolved = policy.resolve_read_path(
        venv_dir
    )

    wanted = list(packages)

    checks: list[Check] = []

    env_obs = _record(
        trace,
        observe.python_env_state(resolved),
        f"verify_env({resolved})",
    )

    if not env_obs.ok:
        checks.append(
            _unknown(
                "venv_interpreter",
                env_obs,
                "could not probe interpreter",
            )
        )

        return VerificationResult(
            checks=checks,
            label=f"env:{resolved}",
        )

    env = env_obs.value or {}

    isolated = (
        bool(env.get("interpreter_exists"))
        and bool(env.get("is_isolated"))
    )

    checks.append(
        Check(
            name="venv_interpreter",
            verdict=(
                Verdict.PASS
                if isolated
                else Verdict.FAIL
            ),
            evidence={
                key: env.get(key)
                for key in (
                    "interpreter_exists",
                    "is_isolated",
                    "version",
                    "prefix",
                )
            },
            reason=(
                ""
                if isolated
                else (
                    "no isolated interpreter "
                    "at that path"
                )
            ),
        )
    )

    if not isolated:
        return VerificationResult(
            checks=checks,
            label=f"env:{resolved}",
        )

    return VerificationResult(
        checks=checks
        + _package_checks(
            resolved,
            wanted,
            trace,
        ),
        label=f"env:{resolved}",
    )


def _package_checks(
    resolved,
    wanted: list[str],
    trace: Trace | None,
) -> list[Check]:

    if not wanted:
        return []

    obs = _record(
        trace,
        observe.package_state(
            resolved,
            wanted,
        ),
        f"verify_packages({wanted})",
    )

    if not obs.ok:
        return [
            _unknown(
                "packages_importable",
                obs,
                "could not probe packages",
            )
        ]

    results = (
        (obs.value or {}).get(
            "packages",
            {}
        )
    )

    checks: list[Check] = []

    for name in wanted:
        entry = results.get(
            name,
            {},
        )

        ok = bool(
            entry.get("import_ok")
        )

        checks.append(
            Check(
                name=f"package:{name}",
                verdict=(
                    Verdict.PASS
                    if ok
                    else Verdict.FAIL
                ),
                evidence={
                    "import_ok": ok,
                    "version": entry.get(
                        "version"
                    ),
                },
                reason=(
                    ""
                    if ok
                    else (
                        "not importable in target "
                        "env: "
                        f"{entry.get('error')}"
                    )
                ),
            )
        )

    return checks


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def combine(
    label: str,
    parts: Iterable[VerificationResult],
) -> VerificationResult:
    """Merge sub-verifications while preserving individual evidence."""

    merged: list[Check] = []

    for part in parts:
        for check in part.checks:
            merged.append(
                Check(
                    name=(
                        f"{part.label}/{check.name}"
                        if part.label
                        else check.name
                    ),
                    verdict=check.verdict,
                    evidence=check.evidence,
                    reason=check.reason,
                )
            )

    return VerificationResult(
        checks=merged,
        label=label,
    )