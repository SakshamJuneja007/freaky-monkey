from agent_control.runtime_control import RuntimeControlKind, classify_runtime_control


def test_grant_target_control_parses_explicit_contact():
    command = classify_runtime_control("allow WhatsApp intelligence for Mummy")
    assert command is not None
    assert command.kind is RuntimeControlKind.WHATSAPP_INTELLIGENCE_GRANT
    assert command.target == "mummy"


def test_grant_target_control_parses_group():
    command = classify_runtime_control("permit WhatsApp intelligence to monitor Family Group on WhatsApp")
    assert command is not None
    assert command.kind is RuntimeControlKind.WHATSAPP_INTELLIGENCE_GRANT
    assert command.target == "family group"


def test_revoke_target_control_parses_explicit_contact():
    command = classify_runtime_control("remove WhatsApp intelligence for Mummy")
    assert command is not None
    assert command.kind is RuntimeControlKind.WHATSAPP_INTELLIGENCE_REVOKE
    assert command.target == "mummy"


def test_list_target_control_parses():
    command = classify_runtime_control("list WhatsApp intelligence targets")
    assert command is not None
    assert command.kind is RuntimeControlKind.WHATSAPP_INTELLIGENCE_LIST
