from __future__ import annotations

from types import SimpleNamespace

from agent_control.skills.browser.backend import BrowserElement, BrowserSkillAdapter
from agent_control.whatsapp_intelligence import extract_messages


def obs(*elements):
    return SimpleNamespace(elements=tuple(elements), text="")


def test_self_chat_semantics_mark_message_outgoing():
    value = obs(
        BrowserElement("@e1", role="heading", name="You"),
        BrowserElement("@e2", role="textbox", name="Type a message"),
        BrowserElement("@e3", role="text", name="Meeting at 6pm"),
        BrowserElement("@e4", role="text", name="11:00 PM"),
        BrowserElement("@e5", role="text", name="Sent"),
        BrowserElement("@e6", role="button", name="Saksham"),
    )
    messages = extract_messages(value, conversation_hint="Saksham", observed_at=1758190000)
    assert len(messages) == 1
    assert messages[0].text == "Meeting at 6pm"
    assert messages[0].metadata["direction"] == "outgoing"
    assert messages[0].metadata["is_outgoing"] is True


def test_self_chat_header_variants_are_verified():
    adapter = BrowserSkillAdapter.__new__(BrowserSkillAdapter)
    adapter.wait_ms = lambda _ms: None

    class Observation:
        text = ""
        elements = (
            BrowserElement("@e1", role="heading", name="You"),
            BrowserElement("@e2", role="textbox", name="Type a message"),
            BrowserElement("@e3", role="button", name="Saksham"),
        )

    adapter._whatsapp_observation = lambda: Observation()
    assert adapter._whatsapp_chat_header_matches("Saksham", timeout_s=0.2)

    class Observation2:
        text = ""
        elements = (
            BrowserElement("@e1", role="heading", name="Saksham (You)"),
            BrowserElement("@e2", role="textbox", name="Type a message"),
        )

    adapter._whatsapp_observation = lambda: Observation2()
    assert adapter._whatsapp_chat_header_matches("Saksham", timeout_s=0.2)


def test_home_candidate_prefers_target_named_clickable_child():
    # This is intentionally a minimal structural contract test. The selector
    # remains semantic: the clickable child must be the one naming the target.
    from agent_control.whatsapp_intelligence import WhatsAppIntelligence, WhatsAppIntelligenceStore
    import tempfile

    store = WhatsAppIntelligenceStore(tempfile.mktemp(suffix=".sqlite3"))
    try:
        store.authorize_target("Saksham")
        service = WhatsAppIntelligence(store)
        row = BrowserElement("@e10", role="button", name="Saksham")
        assert service._home_row_target(row, "Saksham", obs(row)) is row
    finally:
        store.close()
