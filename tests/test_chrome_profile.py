from __future__ import annotations

from agent_control.skills.browser.backend import BrowserSkillAdapter, BrowserSkillCLI


def test_browser_backend_is_browserskill_transport_only():
    adapter = BrowserSkillAdapter(BrowserSkillCLI(executable="bsk"))
    assert adapter.session_id is None
    source = open("agent_control/skills/browser/backend.py", encoding="utf-8").read().lower()
    assert "connect_over_cdp" not in source
    assert "--user-data-dir" not in source
    assert "playwright" not in source
