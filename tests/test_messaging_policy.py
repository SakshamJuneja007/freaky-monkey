from pathlib import Path

from agent_control.policy import Decision, Policy
from agent_control.types import Action


def test_external_message_sends_require_confirmation(tmp_path: Path):
    policy = Policy(workspace=tmp_path, confirm_mode="ask", refuse_if_elevated=False)
    decision, _ = policy.check(Action(kind="whatsapp_send_message", params={"recipient": "Alice", "message": "hello"}))
    assert decision is Decision.CONFIRM
    decision, _ = policy.check(Action(kind="gmail_send_email", params={"recipient": "a@example.com", "subject": "Hi", "body": "hello"}))
    assert decision is Decision.CONFIRM


def test_exact_action_fingerprint_is_the_only_trusted_approval(tmp_path: Path):
    action = Action(
        kind="gmail_send_email",
        params={"recipient": "a@example.com", "subject": "Hi", "body": "hello"},
    )
    fingerprint = Policy.action_fingerprint(action)
    policy = Policy(
        workspace=tmp_path,
        confirm_mode="ask",
        approved_action_fingerprints=frozenset({fingerprint}),
        refuse_if_elevated=False,
    )
    decision, _ = policy.check(action)
    assert decision is Decision.ALLOW

    changed = Action(
        kind=action.kind,
        params={**action.params, "body": "different"},
    )
    decision, _ = policy.check(changed)
    assert decision is Decision.CONFIRM
