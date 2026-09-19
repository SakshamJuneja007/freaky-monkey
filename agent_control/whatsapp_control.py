"""Explicit Human <-> DEIMOS ownership for the existing WhatsApp browser resource.

This is a control-plane lease, not a second browser/resource manager. The
BrowserSkillAdapter remains the only browser executor; the lease only decides
whether DEIMOS may use the existing WhatsApp adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class WhatsAppOwner(str, Enum):
    DEIMOS = "DEIMOS"
    HUMAN = "HUMAN"
    NONE = "NONE"


@dataclass
class WhatsAppControlLease:
    """Session-local ownership of the existing WhatsApp BrowserSkill resource."""

    owner: WhatsAppOwner = WhatsAppOwner.DEIMOS
    browser: Any | None = None
    active_task_id: str | None = None

    @property
    def human_owned(self) -> bool:
        return self.owner is WhatsAppOwner.HUMAN

    @property
    def deimos_owned(self) -> bool:
        return self.owner is WhatsAppOwner.DEIMOS

    def bind_browser(self, browser: Any) -> Any:
        if self.browser is None:
            self.browser = browser
        return self.browser

    def release_to_human(self) -> None:
        # Releasing the lease deliberately does not stop/close the BrowserSkill
        # session. The authenticated WhatsApp page remains available to the user.
        self.owner = WhatsAppOwner.HUMAN

    def release_resource(self) -> None:
        # "Release" means relinquish automation ownership, not browser teardown.
        self.owner = WhatsAppOwner.NONE

    def acquire_for_deimos(self, browser: Any) -> Any:
        self.bind_browser(browser)
        self.owner = WhatsAppOwner.DEIMOS
        return self.browser
