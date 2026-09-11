"""Skill registry and security-aware planner exposure.

Registration and planner exposure are deliberately separate:

    register skill
        ↓
    audit implementation
        ↓
    store audit report
        ↓
    expose only APPROVED skills to the planner

The planner never decides whether a skill is safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .audit import SkillAuditReport, audit_skill
from .gate import is_planner_visible
from .status import SkillStatus


# ``Skill`` is imported only for type checking/validation; concrete skills may
# still use duck-typed executors and verifiers behind the stable interface.
from .base import Skill


@dataclass
class SkillRegistry:
    """Registry of installed skills and their security audit state."""

    _skills: dict[str, Skill] = field(
        default_factory=dict
    )

    _reports: dict[str, SkillAuditReport] = field(
        default_factory=dict
    )

    def register(self, skill: Skill) -> SkillAuditReport:
        """Register and audit a skill.

        Registration does not imply planner exposure. Every skill receives a
        deterministic audit report first. The report is stored alongside the
        skill and controls whether the skill may later be exposed to the LLM.
        """

        if not isinstance(skill, Skill):
            raise TypeError("cannot register a non-Skill object")

        info = getattr(skill, "info", None)

        if info is None:
            raise ValueError(
                "cannot register skill without .info metadata"
            )

        name = getattr(info, "name", None)

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "cannot register skill with empty name"
            )

        name = name.strip()

        if name in self._skills:
            raise ValueError(
                f"skill already registered: {name!r}"
            )

        report = audit_skill(skill)

        self._skills[name] = skill
        self._reports[name] = report

        return report

    def unregister(self, name: str) -> Skill | None:
        """Remove a skill and its audit report."""

        self._reports.pop(name, None)

        return self._skills.pop(name, None)

    def find(self, name: str) -> Skill | None:
        """Return a registered skill by name, or ``None`` if absent."""

        return self._skills.get(name)

    def get(self, name: str) -> Skill:
        """Return a registered skill by name.

        Unlike :meth:`find`, this strict accessor raises ``KeyError`` when the
        name is unknown.
        """

        try:
            return self._skills[name]
        except KeyError:
            raise KeyError(f"unknown skill: {name!r}") from None

    def names(self) -> tuple[str, ...]:
        """Return registered skill names in deterministic order."""

        return tuple(sorted(self._skills))

    def find_for_action(self, kind: str) -> Skill | None:
        """Find the sole approved skill capable of executing ``kind``.

        Registration alone never makes a skill dispatchable. Only skills whose
        current audit report passes the planner exposure gate participate. If
        multiple approved skills claim the same action kind, dispatch is
        ambiguous and fails closed rather than depending on registration order.
        """

        matches: list[Skill] = []

        for name, skill in self._skills.items():
            report = self._reports.get(name)
            if report is None or not is_planner_visible(report):
                continue

            supports = getattr(skill, "supports", None)
            if callable(supports):
                if supports(kind):
                    matches.append(skill)
                continue

            info = getattr(skill, "info", None)
            actions = getattr(info, "actions", ()) if info is not None else ()
            if any(getattr(action, "kind", None) == kind for action in actions):
                matches.append(skill)

        if not matches:
            return None

        if len(matches) > 1:
            names = sorted(getattr(skill, "name", type(skill).__name__) for skill in matches)
            raise ValueError(
                f"multiple skills support action {kind!r}: {names}"
            )

        return matches[0]

    def resolve(self, kind: str) -> Skill | None:
        """Resolve an action kind to its approved owning skill."""
        return self.find_for_action(kind)

    def capabilities(self) -> tuple[str, ...]:
        """Return capabilities advertised by approved registered skills."""
        capabilities: set[str] = set()
        for skill in self.planner_visible():
            capabilities.update(
                capability.value
                for capability in skill.manifest.capabilities
            )
        return tuple(sorted(capabilities))

    def report(
        self,
        name: str,
    ) -> SkillAuditReport | None:
        """Return the latest audit report for a skill."""

        return self._reports.get(name)

    def status(
        self,
        name: str,
    ) -> SkillStatus | None:
        """Return the security status of a registered skill."""

        report = self.report(name)

        if report is None:
            return None

        return report.status

    def all_skills(self) -> tuple[Skill, ...]:
        """Return every registered skill.

        This is for runtime management and inspection only. It does not imply
        that every returned skill is safe to expose to the planner.
        """

        return tuple(self._skills.values())

    def reports(self) -> tuple[SkillAuditReport, ...]:
        """Return audit reports for all registered skills."""

        return tuple(self._reports.values())

    def planner_visible(self) -> tuple[Skill, ...]:
        """Return only skills approved for planner exposure."""

        visible: list[Skill] = []

        for name, skill in self._skills.items():
            report = self._reports.get(name)

            if report is None:
                continue

            if is_planner_visible(report):
                visible.append(skill)

        return tuple(visible)

    def restricted(self) -> tuple[Skill, ...]:
        """Return registered skills hidden from the planner."""

        hidden: list[Skill] = []

        for name, skill in self._skills.items():
            report = self._reports.get(name)

            if report is None:
                hidden.append(skill)
                continue

            if not is_planner_visible(report):
                hidden.append(skill)

        return tuple(hidden)

    def action_kinds(self) -> tuple[str, ...]:
        """Return all unique action kinds exposed by registered skills.

        This is an inspection API; unlike ``planner_action_kinds`` it includes
        non-approved skills. Runtime dispatch still uses ``find_for_action``
        and therefore remains approval-gated.
        """

        action_kinds: set[str] = set()

        for skill in self._skills.values():
            info = getattr(skill, "info", None)
            actions = getattr(info, "actions", ()) if info is not None else ()
            for action in actions:
                kind = getattr(action, "kind", None)
                if isinstance(kind, str) and kind:
                    action_kinds.add(kind)

        return tuple(sorted(action_kinds))

    def planner_action_kinds(self) -> tuple[str, ...]:
        """Return action kinds belonging only to approved skills.

        The planner should consume this method rather than deriving actions
        directly from all registered skills.
        """

        action_kinds: list[str] = []

        for skill in self.planner_visible():
            info = getattr(skill, "info", None)

            if info is None:
                continue

            actions = getattr(info, "actions", ())

            for action in actions:
                kind = getattr(action, "kind", None)

                if isinstance(kind, str):
                    action_kinds.append(kind)

        return tuple(sorted(set(action_kinds)))

    def refresh_audit(
        self,
        name: str,
    ) -> SkillAuditReport:
        """Re-audit an already registered skill.

        Useful after a skill implementation changes or a previously restricted
        skill has been reviewed and modified.
        """

        skill = self._skills.get(name)

        if skill is None:
            raise KeyError(
                f"unknown skill: {name!r}"
            )

        report = audit_skill(skill)

        self._reports[name] = report

        return report

    def refresh_all_audits(
        self,
    ) -> tuple[SkillAuditReport, ...]:
        """Re-audit every registered skill."""

        reports: list[SkillAuditReport] = []

        for name in tuple(self._skills):
            reports.append(
                self.refresh_audit(name)
            )

        return tuple(reports)

    def audit_summary(self) -> dict[str, object]:
        """Return a serializable overview of registry security state."""

        statuses: dict[str, int] = {
            status.value: 0
            for status in SkillStatus
        }

        for report in self._reports.values():
            statuses[report.status.value] += 1

        return {
            "registered": len(self._skills),
            "planner_visible": len(
                self.planner_visible()
            ),
            "restricted": len(
                self.restricted()
            ),
            "statuses": statuses,
            "skills": {
                name: report.to_json()
                for name, report
                in self._reports.items()
            },
        }

    def __contains__(
        self,
        name: object,
    ) -> bool:
        """Support ``name in registry``."""

        return name in self._skills

    def __len__(self) -> int:
        """Return the number of registered skills."""

        return len(self._skills)

    def __iter__(self) -> Iterable[Skill]:
        """Iterate over registered skills."""

        return iter(
            self._skills.values()
        )