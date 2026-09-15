from __future__ import annotations

from agent_control.planner.openai_compat import _parse_whatsapp_send_goal


def test_whatsapp_send_parser_accepts_common_typo_in_message_keyword():
    assert _parse_whatsapp_send_goal(
        "send messsgae to papa on whatsapp telling test message"
    ) == ("papa", "test message")


def test_whatsapp_send_parser_preserves_existing_supported_forms():
    commands = [
        "send whatsapp to papa saying test message",
        "send whatsapp to papa telling test message",
        "send message to papa on whatsapp saying test message",
        "send message to papa on whatsapp telling test message",
        "send a whatsapp message to papa saying test message",
        "send a whatsapp message to papa telling test message",
    ]

    for command in commands:
        assert _parse_whatsapp_send_goal(command) == ("papa", "test message")


def test_whatsapp_send_parser_is_case_insensitive():
    assert _parse_whatsapp_send_goal(
        "SEND MESSSGAE TO Papa ON WhatsApp TELLING test message"
    ) == ("Papa", "test message")


class _FakeClient:
    model = "test"


def test_whatsapp_typo_produces_existing_executable_action():
    from agent_control.planner.openai_compat import OpenAICompatPlanner

    step = OpenAICompatPlanner(_FakeClient()).plan(
        "send messsgae to papa on whatsapp telling test message",
        state={},
        history=[],
    )

    assert [action.kind for action in step.actions] == ["whatsapp_send_message"]
    assert step.actions[0].params == {
        "recipient": "papa",
        "message": "test message",
    }


def test_whatsapp_send_parser_accepts_natural_shorthand():
    assert _parse_whatsapp_send_goal("send hello to papa") == ("papa", "hello")
    assert _parse_whatsapp_send_goal("send hello to papa on whatsapp") == ("papa", "hello")


def test_whatsapp_shorthand_produces_existing_executable_action():
    from agent_control.planner.openai_compat import OpenAICompatPlanner

    class _FakeClient:
        model = "test"

    step = OpenAICompatPlanner(_FakeClient()).plan(
        "send hello to papa",
        state={},
        history=[],
    )

    assert [action.kind for action in step.actions] == ["whatsapp_send_message"]
    assert step.actions[0].params == {
        "recipient": "sir",
        "message": "hello",
    }
