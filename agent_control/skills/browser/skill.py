from __future__ import annotations

from typing import Any

from ..base import (
    Skill,
    SkillAction,
    SkillExecutor,
    SkillInfo,
    SkillVerifier,
)
from ..manifest import SkillManifest
from ..security import Capability

from .actions import (
    BROWSER_ACTION_KINDS,
    BrowserAction,
)
from .backend import BrowserSkillAdapter
from .executor import (
    BrowserBackend,
    BrowserExecutionResult,
    BrowserExecutor,
)
from .browser_verifiers import (
    BrowserVerificationResult,
    BrowserVerifier,
)


class BrowserSkill(Skill):
    _INFO = SkillInfo(
        name="browser",
        version="2.0.0",
        description=(
            "Operate the user's real authenticated "
            "Chromium browser through Tencent "
            "BrowserSkill's Agent Window."
        ),
        actions=tuple(
            SkillAction(
                kind=k,
                description=k.replace(
                    "_",
                    " ",
                ),
            )
            for k in sorted(
                BROWSER_ACTION_KINDS
            )
        ),
        manifest=SkillManifest(
            capabilities=frozenset(
                {
                    Capability.BROWSER,
                    Capability.NETWORK,
                    Capability.SUBPROCESS,
                }
            ),
            side_effecting=True,
        ),
    )

    def __init__(
        self,
        backend: BrowserBackend,
    ) -> None:
        self._backend = backend
        self._executor = BrowserExecutor(
            backend
        )
        self._verifier = BrowserVerifier(
            backend
        )

    @property
    def info(self) -> SkillInfo:
        return self._INFO

    def supports(
        self,
        kind: str,
    ) -> bool:
        return (
            kind in BROWSER_ACTION_KINDS
            or kind == "open_url"
        )

    def executor(self) -> SkillExecutor:
        return self._executor

    def verifier(self) -> SkillVerifier:
        return self._verifier

    def adapt_action(
        self,
        action: Any,
    ) -> BrowserAction:
        kind = getattr(
            action,
            "kind",
            None,
        )

        if kind == "open_url":
            kind = "browser_open_url"

        if kind not in BROWSER_ACTION_KINDS:
            raise ValueError(
                f"browser skill does not support "
                f"{kind!r}"
            )

        params = dict(
            getattr(
                action,
                "params",
                {},
            )
            or {}
        )

        if (
            kind == "browser_open_url"
            and "expected_url" not in params
        ):
            params["expected_url"] = params.get(
                "url"
            )

        return BrowserAction.from_json(
            {
                "kind": kind,
                "params": params,
                "rationale": str(
                    getattr(
                        action,
                        "rationale",
                        "",
                    )
                ),
            }
        )

    def execute(
        self,
        action: BrowserAction,
    ) -> BrowserExecutionResult:
        return self._executor.execute(
            action
        )

    def close_session(self) -> None:
        close = getattr(
            self._backend,
            "close_session",
            None,
        )

        if callable(close):
            close()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "action_kinds": list(
                self.action_kinds
            ),
        }