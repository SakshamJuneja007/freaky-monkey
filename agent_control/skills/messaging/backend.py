from __future__ import annotations

from typing import Any, Protocol


class MessagingBackend(Protocol):
    def send_whatsapp(self, recipient: str, message: str) -> dict[str, Any]: ...
    def search_whatsapp(self, query: str) -> dict[str, Any]: ...
    def send_gmail(self, recipient: str, subject: str, body: str, cc: str = "", bcc: str = "") -> dict[str, Any]: ...
    def search_gmail(self, query: str) -> dict[str, Any]: ...
    def read_gmail(self, query: str) -> dict[str, Any]: ...
    def current_url(self) -> str: ...
    def page_contains_text(self, text: str) -> bool: ...


class BrowserMessagingBackend:
    """Small messaging adapter built on DEIMOS's isolated Chrome profile.

    This deliberately uses the existing browser backend rather than creating a
    second browser or a second execution path. UI labels can change, so failures
    are reported instead of being treated as successful sends.
    """

    def __init__(self, browser: Any) -> None:
        self._browser = browser

    def _click_first(self, labels: tuple[str, ...]) -> str:
        last: Exception | None = None
        for label in labels:
            try:
                self._browser.click(label)
                return label
            except Exception as exc:
                last = exc
        raise RuntimeError(f"could not find messaging control {labels!r}: {last}")

    def send_whatsapp(self, recipient: str, message: str) -> dict[str, Any]:
        if not recipient.strip() or not message.strip():
            raise ValueError("recipient and message are required")
        self._browser.open_url("https://web.whatsapp.com/")
        self._click_first(("Search or start new chat", "Search"))
        self._browser.type_text(recipient)
        self._browser.wait(0.8)
        self._click_first((recipient,))
        self._browser.type_text(message)
        self._browser.press_key("Enter")
        self._browser.wait(0.8)
        return {"provider": "whatsapp", "recipient": recipient, "message_length": len(message)}

    def search_whatsapp(self, query: str) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("search query is required")
        self._browser.open_url("https://web.whatsapp.com/")
        self._click_first(("Search or start new chat", "Search"))
        self._browser.type_text(query)
        self._browser.wait(0.8)
        return {"provider": "whatsapp", "query": query}

    def send_gmail(self, recipient: str, subject: str, body: str, cc: str = "", bcc: str = "") -> dict[str, Any]:
        if not recipient.strip() or not subject.strip() or not body.strip():
            raise ValueError("recipient, subject, and body are required")
        self._browser.open_url("https://mail.google.com/mail/u/0/#inbox")
        self._click_first(("Compose",))
        self._click_first(("Recipients", "To recipients"))
        self._browser.type_text(recipient)
        self._browser.press_key("Enter")
        if cc.strip():
            self._click_first(("Cc",))
            self._browser.type_text(cc)
            self._browser.press_key("Enter")
        if bcc.strip():
            self._click_first(("Bcc",))
            self._browser.type_text(bcc)
            self._browser.press_key("Enter")
        self._browser.press_key("Tab")
        self._browser.type_text(subject)
        self._browser.press_key("Tab")
        self._browser.type_text(body)
        self._click_first(("Send",))
        self._browser.wait(0.8)
        return {"provider": "gmail", "recipient": recipient, "subject": subject}

    def search_gmail(self, query: str) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("search query is required")
        self._browser.open_url("https://mail.google.com/mail/u/0/#inbox")
        self._click_first(("Search mail",))
        self._browser.type_text(query)
        self._browser.press_key("Enter")
        self._browser.wait(0.8)
        return {"provider": "gmail", "query": query}

    def read_gmail(self, query: str) -> dict[str, Any]:
        result = self.search_gmail(query)
        return {**result, "read": True}

    def current_url(self) -> str:
        return self._browser.current_url()

    def page_contains_text(self, text: str) -> bool:
        return self._browser.page_contains_text(text)
