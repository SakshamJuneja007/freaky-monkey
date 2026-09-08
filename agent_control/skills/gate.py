"""Planner exposure gate for DEIMOS skills.

The LLM must never be responsible for deciding whether a suspicious skill is
safe. This module makes exposure a deterministic decision based on the skill's
audit status.
"""

from __future__ import annotations

from typing import Iterable

from .audit import SkillAuditReport
from .security import Capability
from .status import SkillStatus


def is_planner_visible(
    report: SkillAuditReport,
) -> bool:
    """Return whether a skill may be exposed to the planner."""
    return (
        report.status
        is SkillStatus.APPROVED
    )


def planner_visible_reports(
    reports: Iterable[SkillAuditReport],
) -> tuple[SkillAuditReport, ...]:
    """Return only reports whose skills may reach the planner."""

    return tuple(
        report
        for report in reports
        if is_planner_visible(report)
    )


def blocked_reason(
    report: SkillAuditReport,
) -> str:
    """Return a deterministic human-readable reason for blocking exposure.

    Explains *what was detected*, not merely that the skill was rejected --
    e.g. "the implementation calls subprocess.run with shell=True", not just
    "unsafe". This lets a user or admin actually understand and act on the
    audit result rather than trust it blindly.
    """

    if is_planner_visible(report):
        return ""

    if not report.source_available:
        return (
            "the skill's implementation (its adapter class, executor, or "
            "verifier) could not be fully inspected, so it was not "
            "approved"
        )

    critical = [
        finding
        for finding in report.findings
        if finding.severity.value == "CRITICAL"
    ]

    if critical:
        details = "; ".join(
            finding.message for finding in critical
        )

        return (
            "blocked because the implementation contains disallowed "
            f"behavior: {details}"
        )

    if report.status is SkillStatus.QUARANTINED:
        dynamic = [
            finding
            for finding in report.findings
            if finding.capability is Capability.DYNAMIC_CODE
        ]

        details = (
            "; ".join(finding.message for finding in dynamic)
            or "dynamic code execution was detected"
        )

        return (
            "requires human review before it can be approved: "
            f"{details}"
        )

    unexpected = sorted(
        capability.value
        for capability
        in report.unexpected_capabilities
    )

    if unexpected:
        return (
            "undeclared capabilities detected: "
            + ", ".join(unexpected)
        )

    if report.status is SkillStatus.BLOCKED:
        return (
            "critical security findings prevented "
            "skill exposure"
        )

    return (
        "skill did not pass the planner exposure gate"
    )