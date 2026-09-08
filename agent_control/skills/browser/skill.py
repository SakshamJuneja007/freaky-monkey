from __future__ import annotations

from typing import Any

from ..manifest import SkillManifest
from ..security import Capability
from ..base import (
    Skill,
    SkillAction,
    SkillExecutor,
    SkillInfo,
    SkillVerifier,
)
from .actions import (
    BROWSER_ACTION_KINDS,
    BrowserAction,
)
from .executor import (
    BrowserBackend,
    BrowserExecutionResult,
    BrowserExecutor,
)
from .verifier import (
    BrowserVerificationResult,
    BrowserVerifier,
)


class BrowserSkill(Skill):
    """Browser automation capability.

    BrowserSkill exposes semantic browser actions while keeping execution and
    verification separate.

    Policy is intentionally not implemented here. The central DEIMOS policy
    layer decides whether a browser action is permitted before execution.

    The skill also exposes ``adapt_action()`` so the central control loop can
    convert a core Action into the skill-specific BrowserAction without
    importing browser-specific action types.
    """

    _INFO = SkillInfo(
        name="browser",
        description=(
            "Navigate and interact with web browsers using semantic "
            "browser actions."
        ),
        manifest=SkillManifest(
            capabilities=frozenset(
                {
                    Capability.BROWSER,
                    Capability.NETWORK,
                }
            )
        ),
        actions=tuple(
            SkillAction(
                kind=kind,
                description=kind.replace("_", " "),
            )
            for kind in sorted(BROWSER_ACTION_KINDS)
        ),
    )

    def __init__(
        self,
        backend: BrowserBackend,
    ) -> None:
        """Initialize the browser skill."""
        self._backend = backend
        self._executor = BrowserExecutor(backend)

        # BrowserVerifier requires the backend to provide the read-only
        # browser inspection methods expected by the verifier.
        self._verifier = BrowserVerifier(
            backend
        )  # type: ignore[arg-type]

    @property
    def info(self) -> SkillInfo:
        """Return immutable metadata describing the browser skill."""
        return self._INFO

    def executor(self) -> SkillExecutor:
        """Return the browser execution backend."""
        return self._executor

    def verifier(self) -> SkillVerifier:
        """Return the browser verification backend."""
        return self._verifier

    def adapt_action(
        self,
        action: Any,
    ) -> BrowserAction:
        """Convert a core DEIMOS Action into a BrowserAction.

        This keeps browser-specific action conversion inside BrowserSkill,
        allowing the central control loop to remain skill-agnostic.
        """

        if action is None:
            raise TypeError(
                "browser skill cannot adapt a null action"
            )

        action_kind = getattr(
            action,
            "kind",
            None,
        )

        if not isinstance(action_kind, str):
            raise TypeError(
                "browser skill requires an action with "
                "a string 'kind'"
            )

        if not self.supports(action_kind):
            raise ValueError(
                "browser skill does not support "
                f"{action_kind!r}"
            )

        params = getattr(
            action,
            "params",
            {},
        )

        if params is None:
            params = {}

        if not isinstance(params, dict):
            try:
                params = dict(params)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "browser action params must be mapping-like"
                ) from exc

        rationale = getattr(
            action,
            "rationale",
            "",
        )

        return BrowserAction.from_json(
            {
                "kind": action_kind,
                "params": dict(params),
                "rationale": str(rationale),
            }
        )

    def execute(
        self,
        action: BrowserAction,
    ) -> BrowserExecutionResult:
        """Execute a browser action.

        This is a convenience API for direct skill use. The central runner
        normally routes execution through ``executor()`` after policy approval.
        """

        if not isinstance(
            action,
            BrowserAction,
        ):
            raise TypeError(
                "BrowserSkill.execute() expects "
                "a BrowserAction"
            )

        if not self.supports(
            action.kind.value
        ):
            raise ValueError(
                "browser skill does not support "
                f"{action.kind.value!r}"
            )

        return self._executor.execute(
            action
        )

    def verify_url(
        self,
        expected_url: str,
    ) -> BrowserVerificationResult:
        """Independently verify the current browser URL."""
        return self._verifier.verify_url(
            expected_url
        )

    def verify_url_contains(
        self,
        expected_fragment: str,
    ) -> BrowserVerificationResult:
        """Verify that the current browser URL contains a fragment."""
        return self._verifier.verify_url_contains(
            expected_fragment
        )

    def verify_title(
        self,
        expected_title: str,
    ) -> BrowserVerificationResult:
        """Verify the current page title."""
        return self._verifier.verify_title(
            expected_title
        )

    def verify_text(
        self,
        expected_text: str,
    ) -> BrowserVerificationResult:
        """Verify visible text on the current page."""
        return self._verifier.verify_text(
            expected_text
        )

    def describe(self) -> dict[str, Any]:
        """Return metadata useful for discovery and debugging."""
        return {
            "name": self.info.name,
            "description": self.info.description,
            "action_kinds": list(self.action_kinds),
        }