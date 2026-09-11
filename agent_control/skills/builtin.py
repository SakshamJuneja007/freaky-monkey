"""Construction of DEIMOS's built-in skill registry."""

from __future__ import annotations

from ..policy import Policy
from .browser import BrowserSkill
from .browser.backend import PlaywrightChromeBackend
from .filesystem import FilesystemSkill
from .messaging import BrowserMessagingBackend, MessagingSkill
from .registry import SkillRegistry


def build_builtin_registry(policy: Policy) -> SkillRegistry:
    """Build the registry for one run.

    Skill instances are scoped to the run because the filesystem executor is
    bound to that run's Policy. This keeps policy state out of global skill
    objects and makes the registry safe to reuse as an ownership catalogue in
    future plugin work.
    """

    registry = SkillRegistry()
    registry.register(FilesystemSkill(policy))
    # The browser backend is lazy: constructing the registry does not launch
    # Chrome. A single backend is retained for the lifetime of the DEIMOS
    # process so persistent browser state (including user-authorized logins in
    # the dedicated DEIMOS profile) survives across conversational turns.
    global _BROWSER_BACKEND
    if _BROWSER_BACKEND is None:
        _BROWSER_BACKEND = PlaywrightChromeBackend()
    registry.register(BrowserSkill(_BROWSER_BACKEND))
    registry.register(MessagingSkill(BrowserMessagingBackend(_BROWSER_BACKEND)))
    return registry


_BROWSER_BACKEND: PlaywrightChromeBackend | None = None
