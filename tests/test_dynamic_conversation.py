from types import SimpleNamespace

from agent_control.api import TaskStatus
from agent_control.response import DeimosPresentation


def _result(request: str, *, app: str = "chrome", target: str = ""):
    return SimpleNamespace(
        ok=True,
        status=TaskStatus.SUCCESS,
        request=request,
        task_id="general-test",
        completed=["app_process", "app_window"],
        failed=[],
        unresolved=[],
        checks=[],
        false_success=False,
        target=target,
        app=app,
        verified="PASS",
    )


def test_youtube_success_gets_contextual_follow_up(monkeypatch):
    monkeypatch.setenv("DEIMOS_ADDRESS", "sir")
    reply = DeimosPresentation().result(_result("open youtube.com"))

    assert reply == (
        "YouTube is open, sir. Want to listen to some music, "
        "search for something, or just browse?"
    )


def test_music_request_gets_a_human_follow_up(monkeypatch):
    monkeypatch.setenv("DEIMOS_ADDRESS", "")
    reply = DeimosPresentation().result(_result("open spotify"))

    assert "Nice choice" in reply
    assert "music" in reply.lower()


def test_generic_app_gets_a_contextual_follow_up(monkeypatch):
    monkeypatch.setenv("DEIMOS_ADDRESS", "sir")
    reply = DeimosPresentation().result(_result("open calculator", app="calculator"))

    assert "Calculator is open" in reply
    assert "What would you like to do with it?" in reply
