"""Skill security manifest.

A manifest is the skill's explicit declaration of the capabilities it expects
to use. The declaration is not trusted as proof: the audit layer compares it
with capabilities observed from the skill implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .security import Capability


@dataclass(frozen=True)
class SkillManifest:
    """Declared security requirements for a skill."""

    capabilities: frozenset[Capability] = field(
        default_factory=frozenset
    )

    #: Declares that the skill's actions, taken as a whole, can cause an
    #: external effect that a person would want to know happened (sending a
    #: message, posting content, uploading a file, deleting data) as
    #: distinct from pure observation/navigation (opening a page, reading
    #: text, listing files). This is informational only: the DEIMOS policy
    #: layer -- not this flag -- is what actually enforces execution
    #: permission for any given action.
    side_effecting: bool = False

    def declares(
        self,
        capability: Capability,
    ) -> bool:
        """Return whether the capability was explicitly declared."""
        return capability in self.capabilities

    def to_json(self) -> dict[str, object]:
        return {
            "capabilities": sorted(
                capability.value
                for capability in self.capabilities
            ),
            "side_effecting": self.side_effecting,
        }