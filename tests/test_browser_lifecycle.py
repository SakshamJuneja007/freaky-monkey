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
