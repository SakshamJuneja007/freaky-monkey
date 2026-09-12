from __future__ import annotations

import shutil

import pytest

from agent_control.skills.browser.backend import BrowserSkillAdapter



def test_real_browserskill_session_smoke():
    """Exercise the real transport when bsk + extension are available.

    CI and unit-test environments normally do not have a connected browser,
    so this test skips rather than pretending a mock is a live integration.
    """
    if shutil.which("bsk") is None:
        pytest.skip("BrowserSkill CLI (bsk) is not installed")

    browser = BrowserSkillAdapter(timeout=45)
    browser.session_start()
    try:
        browser.navigate("https://example.com")
        observation = browser.observe()
        assert observation.ok
        target = browser.resolve_target("More information", preferred_roles=("link",), min_score=300)
        result = browser.click(target)
        assert result.ok
    finally:
        browser.close_session()
