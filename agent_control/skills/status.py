"""Lifecycle and security status for DEIMOS skills."""

from __future__ import annotations

from enum import Enum


class SkillStatus(str, Enum):
    """Security/lifecycle state assigned to a skill.

    A skill is never planner-visible merely because it is registered.
    Registration and exposure are separate decisions.
    """

    DISCOVERED = "DISCOVERED"
    APPROVED = "APPROVED"
    RESTRICTED = "RESTRICTED"
    QUARANTINED = "QUARANTINED"
    BLOCKED = "BLOCKED"

    @property
    def planner_visible(self) -> bool:
        """Return whether this status permits planner exposure."""
        return self is SkillStatus.APPROVED

    @property
    def terminal(self) -> bool:
        """Return whether the skill requires no further automatic promotion."""
        return self in {
            SkillStatus.APPROVED,
            SkillStatus.RESTRICTED,
            SkillStatus.QUARANTINED,
            SkillStatus.BLOCKED,
        }