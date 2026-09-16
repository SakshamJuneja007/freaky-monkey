from __future__ import annotations

from typing import Any, Protocol


class MessagingBackend(Protocol):
    def open_whatsapp(self) -> dict[str, Any]: ...
    def send_whatsapp(self, recipient: str, message: str) -> dict[str, Any]: ...
    def search_whatsapp(self, query: str) -> dict[str, Any]: ...

    def send_gmail(self, recipient: str, subject: str, body: str, cc: str = "", bcc: str = "") -> dict[str, Any]: ...
    def search_gmail(self, query: str) -> dict[str, Any]: ...
    def read_gmail(self, query: str) -> dict[str, Any]: ...
    def current_url(self) -> str: ...
    def page_contains_text(self, text: str) -> bool: ...


class BrowserMessagingBackend:
    """Gmail/WhatsApp workflows implemented entirely with BrowserSkill refs.

    There is no Playwright, DOM scripting, coordinate click, or text-as-CSS
    selector shortcut here. Each interaction starts from a fresh semantic
    observation and acts on a current ``@eN`` ref.
    """

    def __init__(self, browser: Any) -> None:
        self._browser = browser

    def _ensure_browser_ready(self) -> None:
        ensure = getattr(self._browser, "ensure_ready", None)
        if callable(ensure):
            ensure()

    def _find(self, labels: tuple[str, ...], *, role: str | None = None) -> str:
        self._ensure_browser_ready()
        ref = self._browser.find_ref(labels)
        if ref:
            return ref
        refs = self._browser.find_refs(role=role, labels=labels)
        if refs:
            return refs[0]
        raise RuntimeError(f"could not find semantic browser control matching {labels!r}")

    def _find_textboxes(self, labels: tuple[str, ...] = ()) -> list[str]:
        return self._browser.find_refs(role="textbox", labels=labels)

    def send_whatsapp(self, recipient: str, message: str) -> dict[str, Any]:
        """Delegate WhatsApp execution to BrowserSkill's semantic workflow."""
        if not isinstance(recipient, str) or not recipient.strip():
            raise ValueError("recipient is required")
        if not isinstance(message, str) or not message:
            raise ValueError("message is required")
        return dict(self._browser.send_whatsapp_message(recipient.strip(), message))

    def search_whatsapp(self, query: str) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("search query is required")
        self._ensure_browser_ready()
        self._browser.open_url("https://web.whatsapp.com/")
        search = self._find(("Search or start new chat", "Search"), role="textbox")
        self._browser.type_text(search, query)
        self._browser.wait(0.8)
        return {"provider": "whatsapp", "query": query, "page": self._browser.page_text()}

    def send_gmail(self, recipient: str, subject: str, body: str, cc: str = "", bcc: str = "") -> dict[str, Any]:
        if not recipient.strip() or not subject.strip() or not body.strip():
            raise ValueError("recipient, subject, and body are required")
        self._ensure_browser_ready()
        self._browser.open_url("https://mail.google.com/mail/u/0/#inbox")
        compose = self._find(("Compose",))
        self._browser.click(compose)
        self._browser.wait(0.4)

        to_ref = self._find(("Recipients", "To recipients", "To"), role="textbox")
        self._browser.type_text(to_ref, recipient)
        self._browser.press_key("Enter", to_ref)

        if cc.strip():
            cc_ref = self._find(("Cc", "Carbon copy"))
            self._browser.click(cc_ref)
            self._browser.wait(0.2)
            boxes = self._find_textboxes(("Cc", "Carbon copy"))
            self._browser.type_text(boxes[-1], cc)
            self._browser.press_key("Enter", boxes[-1])

        if bcc.strip():
            bcc_ref = self._find(("Bcc", "Blind carbon copy"))
            self._browser.click(bcc_ref)
            self._browser.wait(0.2)
            boxes = self._find_textboxes(("Bcc", "Blind carbon copy"))
            self._browser.type_text(boxes[-1], bcc)
            self._browser.press_key("Enter", boxes[-1])

        subject_ref = self._find(("Subject",), role="textbox")
        self._browser.type_text(subject_ref, subject)
        body_boxes = self._find_textboxes(("Message Body", "Body"))
        if not body_boxes:
            body_boxes = self._find_textboxes()
        if not body_boxes:
            raise RuntimeError("Gmail message body was not found")
        body_ref = body_boxes[-1]
        if body_ref == subject_ref and len(body_boxes) > 1:
            body_ref = body_boxes[-2]
        self._browser.type_text(body_ref, body)
        send = self._find(("Send",), role="button")
        self._browser.click(send)
        self._browser.wait(0.8)
        return {"provider": "gmail", "recipient": recipient, "subject": subject, "body": body}

    def search_gmail(self, query: str) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("search query is required")
        self._browser.open_url("https://mail.google.com/mail/u/0/#inbox")
        search = self._find(("Search mail", "Search in mail"), role="textbox")
        self._browser.type_text(search, query)
        self._browser.press_key("Enter", search)
        self._browser.wait(0.8)
        return {"provider": "gmail", "query": query, "page": self._browser.page_text()}

    def read_gmail(self, query: str) -> dict[str, Any]:
        result = self.search_gmail(query)
        return {**result, "read": True}

    def current_url(self) -> str:
        return self._browser.current_url()

    def page_contains_text(self, text: str) -> bool:
        return self._browser.page_contains_text(text)
