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

__all__ = [
    "BrowserSkill",
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
