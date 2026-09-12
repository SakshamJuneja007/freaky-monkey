"""Construction of DEIMOS's built-in skill registry."""

from __future__ import annotations

from ..policy import Policy
from .browser import BrowserSkill, BrowserSkillAdapter
from .filesystem import FilesystemSkill
from .messaging import BrowserMessagingBackend, MessagingSkill
from .registry import SkillRegistry


# Legacy fallback for direct API callers that do not supply an application-owned
# browser backend. Conversational Session instances supply their own backend.
_BROWSER_BACKEND: BrowserSkillAdapter | None = None


def build_builtin_registry(
    policy: Policy,
    *,
    browser_backend: BrowserSkillAdapter | None = None,
) -> SkillRegistry:
    """Build the built-in skill registry with explicit browser ownership.

    A conversational Session supplies its long-lived BrowserSkillAdapter so the
    same browser session survives across turns. Direct API callers may omit it;
    those calls use the legacy module backend and the API cleans that backend up
    when the call finishes.
    """
    registry = SkillRegistry()
    registry.register(FilesystemSkill(policy))

    global _BROWSER_BACKEND

    backend = browser_backend
    if backend is None:
        if _BROWSER_BACKEND is None:
            _BROWSER_BACKEND = BrowserSkillAdapter()
        backend = _BROWSER_BACKEND

    registry.register(BrowserSkill(backend))
    registry.register(MessagingSkill(BrowserMessagingBackend(backend)))
    return registry


def close_builtin_browser_session() -> None:
    """Stop and release the legacy API-owned browser backend, if any."""
    global _BROWSER_BACKEND

    backend = _BROWSER_BACKEND
    _BROWSER_BACKEND = None

    if backend is not None:
        try:
            backend.close_session()
        except Exception:
            # Cleanup must never turn a completed API call into a failure when
            # the external browser has already stopped.
            pass
