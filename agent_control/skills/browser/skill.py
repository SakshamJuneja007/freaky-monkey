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
from .backend import BrowserSkillAdapter, BrowserTarget, BrowserSkillError
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

        # Browser application/process readiness is not BrowserSkill readiness.
        # Before resolving any semantic target, establish/reuse the real
        # BrowserSkill session and obtain a fresh observation.
        ensure_ready = getattr(self._backend, "ensure_ready", None)
        if callable(ensure_ready):
            ensure_ready()

        # Fast semantic actions may carry the existing BrowserTarget object
        # until this skill boundary. BrowserAction itself intentionally requires
        # plain JSON-compatible parameter values, so translate the target here
        # while preserving BrowserTarget's generation/staleness check.
        target = params.get("target")
        # Deterministic compound decomposition may preserve a semantic target
        # query (for example a textbox name) instead of inventing an @eN ref.
        # Resolve it against the current BrowserSkill observation at the skill
        # boundary; the resulting ref is still generation-bound.
        target_query = params.get("target_query")
        if target is None and isinstance(target_query, str) and target_query.strip():
            preferred_roles = ()
            semantic = params.get("target_semantic")
            if isinstance(semantic, dict) and isinstance(semantic.get("role"), str):
                preferred_roles = (semantic["role"],)
            target = self._backend.resolve_target(
                target_query,
                preferred_roles=preferred_roles,
            )
            params["target"] = target
            params.pop("target_query", None)
        if isinstance(target, BrowserTarget):
            params["target"] = target.ref
            if target.generation is not None:
                current_generation = getattr(self._backend, "_generation", None)
                if current_generation != target.generation:
                    raise BrowserSkillError(
                        f"stale BrowserSkill reference {target.ref!r}: observation generation "
                        f"{target.generation} is no longer current (current={current_generation})"
                    )

        # Normal YouTube playback must never execute a semantic click on a
        # Short. This is enforced at the BrowserSkill boundary as well as in
        # indexed/song selection so planner-generated click actions cannot bypass
        # the candidate-set exclusion. It uses the current cached observation;
        # it does not observe or mutate state between resolution and execution.
        if kind == "browser_click" and "youtube.com" in str(getattr(self._backend._last_observation, "url", "")).casefold():
            obs = getattr(self._backend, "_last_observation", None)
            if obs is not None:
                ref = params.get("target")
                ref = ref.ref if isinstance(ref, BrowserTarget) else ref
                if isinstance(ref, str):
                    element = next((e for e in obs.elements if e.ref == ref), None)
                    if element is not None and self._backend._song_result_is_short(element, observation=obs):
                        raise BrowserSkillError(
                            "YouTube Short is not a valid normal-playback target",
                            code="youtube_short_rejected",
                            data={"semantic_target_category": "youtube_short"},
                        )

        if kind == "browser_search" and not any(
            key in params
            for key in ("expected_url", "expected_url_contains", "expected_title", "expected_text")
        ):
            # Search is navigation-backed in BrowserSkill.  The query is a
            # generic, engine-independent postcondition candidate: the
            # resulting page should expose the submitted query in readable
            # browser state.  The verifier still requires fresh observation.
            query = params.get("query")
            if isinstance(query, str) and query.strip():
                params["expected_text"] = query.strip()

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