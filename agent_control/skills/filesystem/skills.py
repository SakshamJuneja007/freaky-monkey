"""Built-in filesystem skill.

The filesystem skill is deliberately a thin adapter over the existing
``os_tools`` implementation. It owns capability metadata and dispatch
registration, but it does not reimplement filesystem semantics, policy, or
verification.
"""

from __future__ import annotations

from typing import Any

from ... import os_tools
from ...policy import Policy
from ...types import Action
from ..base import Skill, SkillAction, SkillExecutor, SkillInfo, SkillVerifier
from ..manifest import SkillManifest
from ..security import Capability


FILESYSTEM_ACTION_KINDS = (
    "create_dir",
    "write_file",
    "fetch_file",
    "open_file",
    "list_directory",
    "read_text_file",
    "search_files",
)


class FilesystemExecutor:
    """Adapter that preserves the existing ``os_tools`` execution semantics."""

    def __init__(self, policy: Policy) -> None:
        self._policy = policy

    def execute(self, action: Action):
        return os_tools.execute(self._policy, action)


class FilesystemVerifier:
    """Explicitly defer final effect verification to the existing Task verifier.

    ``GeneralTask`` and the registered tasks already provide independent
    verification of filesystem effects. This skill must not invent a second
    verifier that could disagree with them or turn executor output into proof.
    """

    def verify(self, action: Action, result: Any) -> Any:
        class DeferredVerification:
            ok = False
            detail = "filesystem verification is delegated to the task-level verifier"

        return DeferredVerification()


class FilesystemSkill(Skill):
    """Expose existing workspace filesystem actions through the skill registry."""

    _INFO = SkillInfo(
        name="filesystem",
        version="1.0.0",
        description=(
            "Create, read, search, fetch, and open files and directories "
            "using DEIMOS workspace-aware filesystem actions."
        ),
        actions=tuple(
            SkillAction(
                kind=kind,
                description=kind.replace("_", " "),
            )
            for kind in FILESYSTEM_ACTION_KINDS
        ),
        manifest=SkillManifest(
            capabilities=frozenset(
                {
                    Capability.FILESYSTEM_READ,
                    Capability.FILESYSTEM_WRITE,
                    
                }
            ),
            side_effecting=True,
        ),
    )

    def __init__(self, policy: Policy) -> None:
        self._executor = FilesystemExecutor(policy)
        self._verifier = FilesystemVerifier()

    @property
    def info(self) -> SkillInfo:
        return self._INFO

    def executor(self) -> SkillExecutor:
        return self._executor

    def verifier(self) -> SkillVerifier:
        return self._verifier

    def adapt_action(self, action: Action) -> Action:
        if not isinstance(action, Action):
            raise TypeError("filesystem skill expects a core Action")
        if not self.supports(action.kind):
            raise ValueError(f"filesystem skill does not support {action.kind!r}")
        # Keep the core Action intact. os_tools already consumes this exact
        # contract, so adaptation is an explicit no-op rather than a lossy copy.
        return action
