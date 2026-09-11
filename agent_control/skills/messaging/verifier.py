from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import MessagingAction, MessagingActionKind


@dataclass(frozen=True)
class MessagingVerificationResult:
    ok: bool
    detail: str = ""


class MessagingVerifier:
    """Read-only postcondition checks for messaging operations."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def verify(self, action: MessagingAction, result: Any) -> MessagingVerificationResult:
        if not isinstance(action, MessagingAction):
            raise TypeError("MessagingVerifier expects MessagingAction")
        if not bool(getattr(result, "ok", False)):
            return MessagingVerificationResult(False, "messaging executor reported failure")

        try:
            url = self._backend.current_url().lower()
            if action.kind is MessagingActionKind.WHATSAPP_SEND:
                if "web.whatsapp.com" not in url:
                    return MessagingVerificationResult(False, "WhatsApp page is no longer active")
                return MessagingVerificationResult(
                    True,
                    "WhatsApp send completed on the DEIMOS messaging page",
                )
            if action.kind is MessagingActionKind.GMAIL_SEND:
                if "mail.google.com" not in url:
                    return MessagingVerificationResult(False, "Gmail page is no longer active")
                # Gmail's confirmation banner is the strongest lightweight
                # read-only evidence available without opening or changing mail.
                for text in ("Message sent", "Message Sent"):
                    if self._backend.page_contains_text(text):
                        return MessagingVerificationResult(True, "Gmail reported the message as sent")
                return MessagingVerificationResult(False, "Gmail send confirmation was not observed")

            return MessagingVerificationResult(True, "messaging read/search completed")
        except Exception as exc:
            return MessagingVerificationResult(False, f"messaging verification failed: {type(exc).__name__}: {exc}")
