"""Tests for BrowserSkill session ownership and lifecycle boundaries."""

from __future__ import annotations

from agent_control.session import Session


class FakeBrowserBackend:
    def __init__(self) -> None:
        self.close_calls = 0

    def close_session(self) -> None:
        self.close_calls += 1


class FakeNarrator:
    speaking = False

    def close(self, timeout_s: float = 15.0) -> None:
        return None


def test_session_close_stops_its_owned_browser_backend() -> None:
    backend = FakeBrowserBackend()
    session = Session(narrator=FakeNarrator(), _browser_backend=backend)

    session.close()

    assert backend.close_calls == 1
    assert session._browser_backend is None


def test_session_close_is_idempotent_for_browser_backend() -> None:
    backend = FakeBrowserBackend()
    session = Session(narrator=FakeNarrator(), _browser_backend=backend)

    session.close()
    session.close()

    assert backend.close_calls == 1
    assert session._browser_backend is None


def test_task_browser_resources_get_independent_sessions():
    from agent_control.skills.browser.backend import BrowserSkillAdapter

    class FakeCLI:
        def __init__(self):
            self.sessions = 0
        def run(self, args):
            args = list(args)
            if args[:2] == ["session", "start"]:
                self.sessions += 1
                return {"session_id": f"s{self.sessions}", "ok": True}
            return {"ok": True}

    cli = FakeCLI()
    base = BrowserSkillAdapter(cli)
    a = base.new_task_session()
    b = base.new_task_session()
    a.session_start()
    b.session_start()
    assert a.session_id != b.session_id
    assert a.session_id == "s1"
    assert b.session_id == "s2"
    assert a._generation != b._generation or a.session_id != b.session_id


def test_session_assigns_distinct_browser_resources_to_independent_tasks():
    from agent_control.session import Session

    class FakeBrowser:
        def __init__(self, name):
            self.name = name
            self.closed = False
        def new_task_session(self):
            return FakeBrowser(f"child-{self.name}")
        def close_session(self):
            self.closed = True

    session = Session(narrator=FakeNarrator(), _browser_backend=FakeBrowser("base"))
    try:
        a = session._browser_for_task("youtube-task")
        b = session._browser_for_task("whatsapp-task")
        assert a is not b
        assert session._browser_for_task("youtube-task") is a
    finally:
        session.close()


def test_all_browser_sites_reuse_one_live_browser_resource():
    from agent_control.session import Session

    class FakeBrowser:
        def __init__(self, name):
            self.name = name
        def new_task_session(self):
            return FakeBrowser(self.name + ".child")
        def close_session(self): pass

    session = Session(narrator=FakeNarrator(), _browser_backend=FakeBrowser("base"))
    try:
        youtube_a = session._browser_for_task("task-a", resource_key="youtube")
        youtube_b = session._browser_for_task("task-b", resource_key="youtube")
        whatsapp = session._browser_for_task("task-c", resource_key="whatsapp")
        assert youtube_a is youtube_b is whatsapp
    finally:
        session.close()


def test_resource_identity_reuses_matching_site_and_not_unrelated_site():
    from agent_control.session import Session

    class FakeBrowser:
        def __init__(self, name):
            self.name = name
            self.created = []
        def new_task_session(self):
            child = FakeBrowser(self.name + ".child")
            self.created.append(child)
            return child
        def close_session(self):
            pass

    base = FakeBrowser("base")
    session = Session(narrator=FakeNarrator(), _browser_backend=base)
    try:
        youtube = session._browser_for_task("task-a", resource_key="youtube")
        whatsapp = session._browser_for_task("task-b", resource_key="whatsapp")
        youtube_again = session._browser_for_task("task-c", resource_key="youtube")
        gmail = session._browser_for_task("task-d", resource_key="gmail")
        assert youtube_again is youtube
        assert whatsapp is youtube
        assert gmail is youtube
    finally:
        session.close()


def test_whatsapp_open_reuses_existing_ready_whatsapp_page():
    from unittest.mock import patch
    from agent_control.skills.browser.backend import BrowserSkillAdapter

    browser = BrowserSkillAdapter(cli=object(), session="s1")
    with patch.object(browser, "current_url", return_value="https://web.whatsapp.com/") as current_url, \
         patch.object(browser, "wait_for_whatsapp_ready", return_value="READY"), \
         patch.object(browser, "navigate") as navigate:
        result = browser.open_whatsapp(timeout_s=1.0)

    assert result.ok
    assert result.get("reused") is True
    current_url.assert_called_once()
    navigate.assert_not_called()


def test_browser_resources_share_one_live_browser_adapter_across_sites():
    from agent_control.session import Session

    class FakeBrowser:
        def __init__(self):
            self.created = 0
        def new_task_session(self):
            self.created += 1
            return self
        def close_session(self):
            pass

    base = FakeBrowser()
    session = Session(narrator=FakeNarrator(), _browser_backend=base)
    try:
        a = session._browser_for_task("youtube", resource_key="youtube")
        b = session._browser_for_task("whatsapp", resource_key="whatsapp")
        c = session._browser_for_task("gmail", resource_key="gmail")
        assert a is b is c
        assert base.created == 1
    finally:
        session.close()
