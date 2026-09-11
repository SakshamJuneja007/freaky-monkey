from __future__ import annotations

from dataclasses import dataclass

from agent_control.skills.messaging import MessagingSkill
from agent_control.types import Action


@dataclass
class Backend:
    sent: list[tuple] = None

    def __post_init__(self):
        self.sent = []

    def send_whatsapp(self, recipient, message):
        self.sent.append(("whatsapp", recipient, message))
        return {"provider": "whatsapp"}

    def search_whatsapp(self, query):
        return {"query": query}

    def send_gmail(self, recipient, subject, body, cc="", bcc=""):
        self.sent.append(("gmail", recipient, subject, body, cc, bcc))
        return {"provider": "gmail"}

    def search_gmail(self, query):
        return {"query": query}

    def read_gmail(self, query):
        return {"query": query, "read": True}

    def current_url(self):
        return "https://web.whatsapp.com/"

    def page_contains_text(self, text):
        return text == "Message sent"


def test_messaging_skill_owns_p1_actions():
    skill = MessagingSkill(Backend())
    assert skill.supports("whatsapp_send_message")
    assert skill.supports("gmail_send_email")
    assert skill.supports("gmail_search_mail")


def test_whatsapp_executor_returns_core_action_result():
    backend = Backend()
    result = MessagingSkill(backend).executor().execute(
        MessagingSkill(backend).adapt_action(
            Action(kind="whatsapp_send_message", params={"recipient": "Alice", "message": "hello"})
        )
    )
    assert result.ok is True
    assert backend.sent == [("whatsapp", "Alice", "hello")]
