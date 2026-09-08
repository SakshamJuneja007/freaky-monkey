"""Base interfaces for DEIMOS skills.

A skill is a reusable capability that contributes actions to the agent.

Skills do not bypass the normal DEIMOS safety pipeline:

    planner -> policy -> executor -> verifier

A skill describes supported actions and provides execution and verification
implementations. Policy remains the authority that decides whether an action
is allowed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

from .manifest import SkillManifest


@dataclass(frozen=True)
class SkillAction:
    """A skill-level action definition.

    This describes an action exposed by a skill without replacing the core
    ``agent_control.types.Action`` object used by the runner.
    """

    kind: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillInfo:
    """Immutable metadata describing a skill.

    ``manifest`` is the skill's own declaration of the capabilities it
    expects to use. It is never trusted as proof by itself: the audit layer
    (``agent_control.skills.audit``) inspects the skill's actual
    implementation and compares what it observes against this declaration.
    A skill's free-text ``description`` and its ``SkillAction.description``
    values are informational metadata only -- they are never treated as
    instructions by the planner or by the security layer.
    """

    name: str
    description: str
    actions: tuple[SkillAction, ...] = ()
    manifest: SkillManifest = field(
        default_factory=SkillManifest
    )


class SkillExecutor(Protocol):
    """Protocol implemented by skill execution backends."""

    def execute(self, action: Any) -> Any:
        """Execute a skill action."""
        ...


class SkillVerifier(Protocol):
    """Protocol implemented by skill verification backends."""

    def verify(self, action: Any, result: Any) -> Any:
        """Independently verify a skill action result."""
        ...


class Skill(ABC):
    """Base class for every DEIMOS skill.

    A skill is a capability provider. It does not make permission decisions
    and does not control the agent loop, and it does not decide whether it
    is safe to expose to the planner -- that decision belongs entirely to
    ``agent_control.skills.registry.SkillRegistry`` and the security audit
    it runs (``agent_control.skills.audit``, ``agent_control.skills.gate``).

    The core architecture remains:

        observe -> plan -> permit -> execute -> verify -> recover
    """

    @property
    @abstractmethod
    def info(self) -> SkillInfo:
        """Return immutable metadata describing this skill."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Return the stable registry name for this skill."""
        return self.info.name

    @property
    def description(self) -> str:
        """Return the human-readable description for this skill."""
        return self.info.description

    @property
    def manifest(self) -> SkillManifest:
        """Return the declared security manifest for this skill."""
        return self.info.manifest

    @abstractmethod
    def executor(self) -> SkillExecutor:
        """Return the executor responsible for this skill."""
        raise NotImplementedError

    @abstractmethod
    def verifier(self) -> SkillVerifier:
        """Return the verifier responsible for this skill."""
        raise NotImplementedError

    @abstractmethod
    def adapt_action(self, action: Any) -> Any:
        """Convert a core DEIMOS Action into this skill's action type."""
        raise NotImplementedError

    @property
    def action_kinds(self) -> tuple[str, ...]:
        """Return the action kinds exposed by this skill."""
        return tuple(
            action.kind
            for action in self.info.actions
        )

    def supports(self, kind: str) -> bool:
        """Return whether this skill supports an action kind."""
        return kind in self.action_kinds