from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import MessagingAction, MessagingActionKind


@dataclass(frozen=True)
class MessagingVerificationResult:
    ok: bool
    detail: str = ""
    status: str = "PASS"


class MessagingVerifier:
    """Independent read-only verification of messaging outcomes."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def verify(self, action: MessagingAction, result: Any) -> MessagingVerificationResult:
        if not isinstance(action, MessagingAction):
            raise TypeError("MessagingVerifier expects MessagingAction")

        # Verification intentionally ignores ``result.ok``. The executor's
        # belief that a click succeeded is not browser evidence.
        try:
            if action.kind is MessagingActionKind.WHATSAPP_SEND:
                recipient = str(action.params.get("recipient", ""))
                message = str(action.params.get("message", ""))
                browser = getattr(self._backend, "_browser", None)
                if browser is None:
                    return MessagingVerificationResult(
                        False,
                        "WhatsApp browser backend is unavailable",
                        "UNKNOWN",
                    )
                observed = browser.verify_whatsapp_message(recipient, message)
                status = str(observed.get("status", "UNKNOWN")).upper()
                return MessagingVerificationResult(
                    bool(observed.get("ok", False)) and status == "PASS",
                    str(observed.get("detail", observed.get("code", ""))),
                    status if status in {"PASS", "FAIL", "UNKNOWN"} else "UNKNOWN",
                )

            url = self._backend.current_url().lower()
            if action.kind is MessagingActionKind.GMAIL_SEND:
                if "mail.google.com" not in url:
                    return MessagingVerificationResult(False, "Gmail page is not active", "FAIL")
                if self._backend.page_contains_text("Message sent"):
                    return MessagingVerificationResult(True, "Gmail confirmation was independently observed")
                return MessagingVerificationResult(False, "Gmail send confirmation was not observed", "UNKNOWN")
            return MessagingVerificationResult(True, "read/search state observed")
        except Exception as exc:
            return MessagingVerificationResult(False, f"verification failed: {type(exc).__name__}: {exc}", "UNKNOWN")
