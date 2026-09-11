from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessagingActionKind(str, Enum):
    WHATSAPP_SEND = "whatsapp_send_message"
    WHATSAPP_SEARCH = "whatsapp_search_contact"
    GMAIL_SEND = "gmail_send_email"
    GMAIL_SEARCH = "gmail_search_mail"
    GMAIL_READ = "gmail_read_mail"


@dataclass(frozen=True)
class MessagingAction:
    kind: MessagingActionKind
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MessagingActionKind):
            raise TypeError("kind must be a MessagingActionKind")
        if not isinstance(self.params, dict):
            raise TypeError("params must be a dictionary")

    @classmethod
    def from_core(cls, action: Any) -> "MessagingAction":
        kind = MessagingActionKind(str(getattr(action, "kind", "")))
        params = getattr(action, "params", {})
        if not isinstance(params, dict):
            raise TypeError("messaging action params must be a dictionary")
        return cls(kind=kind, params=dict(params), rationale=str(getattr(action, "rationale", "")))


MESSAGING_ACTION_KINDS = frozenset(k.value for k in MessagingActionKind)
SEND_ACTION_KINDS = frozenset({
    MessagingActionKind.WHATSAPP_SEND.value,
    MessagingActionKind.GMAIL_SEND.value,
})
