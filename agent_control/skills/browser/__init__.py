"""DEIMOS browser skill backed exclusively by Tencent BrowserSkill."""

from .backend import (
    BrowserElement,
    BrowserObservation,
    BrowserSkillAdapter,
    BrowserSkillCLI,
    BrowserSkillError,
    BrowserSkillProtocolError,
    BrowserSkillResult,
    BrowserSkillUnavailable,
    BrowserTarget,
    canonical_url,
)
from .skill import BrowserSkill
from .text_observer import BrowserTextObserver

__all__ = [
    "BrowserSkill",
    "BrowserTextObserver",
    "BrowserSkillAdapter",
    "BrowserSkillCLI",
    "BrowserSkillError",
    "BrowserSkillProtocolError",
    "BrowserSkillResult",
    "BrowserSkillUnavailable",
    "BrowserElement",
    "BrowserObservation",
    "BrowserTarget",
    "canonical_url",
]
