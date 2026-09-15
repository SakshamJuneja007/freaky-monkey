from __future__ import annotations

import threading
import time

from agent_control.api import AgentResult, TaskStatus
from agent_control.response import Narrator
from agent_control.session import PendingApproval, Session, Turn, UserTask, parse_approval_response
from agent_control.skills.browser.backend import BrowserElement, BrowserSkillAdapter
from agent_control.fast_interaction import FastRoute
from agent_control.speech import mic
from agent_control.speech.tts import Speaker, TTSConfig, Utterance


def session() -> Session:
    return Session(narrator=Narrator(speaker=None, write=lambda _: None), planner="mock")


def el(ref: str, name: str, *, raw=None) -> BrowserElement:
    raw = dict(raw or {})
    raw.setdefault("duration", "3:45")
    return BrowserElement(ref, role="link", name=name, raw=raw)


def test_youtube_shorts_are_hard_excluded_before_ranking() -> None:
    candidates = [
        el("@short", "Song Name (Lyrics)", raw={"href": "https://www.youtube.com/shorts/abc"}),
        el("@normal", "Song Name" , raw={"href": "https://www.youtube.com/watch?v=xyz"}),
    ]
    ranked = BrowserSkillAdapter._rank_song_candidates("Song Name", candidates)
    assert [item[1].ref for item in ranked] == ["@normal"]


def test_metadata_marked_short_is_hard_excluded() -> None:
    candidates = [
        el("@short", "Song Name", raw={"type": "short", "url": "https://www.youtube.com/watch?v=abc"}),
        el("@normal", "Song Name (Official Video)", raw={"type": "video", "url": "https://www.youtube.com/watch?v=xyz"}),
    ]
    ranked = BrowserSkillAdapter._rank_song_candidates("Song Name", candidates)
    assert [item[1].ref for item in ranked] == ["@normal"]


def test_fast_indexed_video_selection_excludes_shorts() -> None:
    from agent_control.skills.browser.backend import BrowserObservation

    browser = BrowserSkillAdapter(cli=type("CLI", (), {})())
    observation = BrowserObservation(
        generation=7,
        session="s1",
        tab_id=1,
        url="https://www.youtube.com/results",
        text="",
        elements=(
            el("@short", "First result", raw={"href": "https://www.youtube.com/shorts/a"}),
            el("@normal", "Second result", raw={"href": "https://www.youtube.com/watch?v=b"}),
        ),
        raw={},
    )
    route = FastRoute("browser_click", {}, target_index=1, target_roles=("link",))
    action = route.resolve(browser, observation)
    assert action.params["target"].ref == "@normal"
    assert action.params["target"].generation == 7


def test_lyrics_short_never_outranks_normal_video() -> None:
    candidates = [
        el("@short", "Song Name (Official Lyrics)", raw={"href": "https://www.youtube.com/shorts/a"}),
        el("@normal", "Song Name (Official Video)", raw={"href": "https://www.youtube.com/watch?v=b"}),
    ]
    ranked = BrowserSkillAdapter._rank_song_candidates("Song Name", candidates)
    assert ranked[0][1].ref == "@normal"


def test_approval_parser_is_small_and_explicit() -> None:
    assert parse_approval_response("yes") == "APPROVE"
    assert parse_approval_response("yeah") == "APPROVE"
    assert parse_approval_response("go ahead") == "APPROVE"
    assert parse_approval_response("no") == "REJECT"
    assert parse_approval_response("don't send") == "REJECT"
    assert parse_approval_response("maybe") == "AMBIGUOUS"


def _approval(task_id="task-123", background_task_id=None):
    return PendingApproval(
        request="send message to papa saying hello",
        action={"kind": "whatsapp_send_message", "params": {"recipient": "papa", "message": "hello"}},
        asked_at=time.time(),
        task_id=task_id,
        goal="send message to papa saying hello",
        background_task_id=background_task_id,
    )


def test_approval_yes_resumes_same_task_without_normal_router(monkeypatch) -> None:
    s = session()
    s.pending_approval = _approval()
    seen = []

    def fake_run(prepared):
        seen.append(prepared)
        return Turn(prepared.task, "sent", AgentResult(request=prepared.task.text, task_id=prepared.task.task_id, status=TaskStatus.SUCCESS))

    monkeypatch.setattr(s, "_run", fake_run)
    monkeypatch.setattr(s, "_converse", lambda *_: (_ for _ in ()).throw(AssertionError("normal router used")))
    turn = s.submit("yes")
    assert turn.result is not None and turn.result.ok
    assert seen[0].task.task_id == "task-123"
    assert seen[0].approved_action["params"]["message"] == "hello"
    assert s.pending_approval is None


def test_approval_yes_does_not_create_background_task(monkeypatch) -> None:
    s = session()
    bg_id = s.submit_background("open youtube")
    # Replace the background task with a pending approval belonging to it.
    s._background[bg_id].state = "WAITING_FOR_APPROVAL"
    s.pending_approval = _approval(task_id="task-123", background_task_id=bg_id)
    submitted = []
    monkeypatch.setattr(s, "_resume_approved_background", lambda task_id, prepared: submitted.append((task_id, prepared.task.task_id)))
    before = len(s.background_tasks())
    returned = s.submit_background("yes")
    assert returned == "task-123"
    assert len(s.background_tasks()) == before
    deadline = time.time() + 1
    while time.time() < deadline and not submitted:
        time.sleep(0.01)
    assert submitted == [(bg_id, "task-123")]
    s.close()


def test_approval_no_and_ambiguous_keep_normal_router_out(monkeypatch) -> None:
    s = session()
    s.pending_approval = _approval()
    monkeypatch.setattr(s, "_converse", lambda *_: (_ for _ in ()).throw(AssertionError("normal router used")))
    ambiguous = s.submit("maybe")
    assert ambiguous.result is None
    assert s.pending_approval is not None
    rejected = s.submit("no")
    assert rejected.result is None
    assert s.pending_approval is None


def test_voice_approval_uses_approval_owner(monkeypatch) -> None:
    s = session()
    s.pending_approval = _approval()
    seen = []
    monkeypatch.setattr(s, "_run", lambda prepared: seen.append(prepared) or Turn(prepared.task, "sent", AgentResult(request=prepared.task.text, task_id=prepared.task.task_id, status=TaskStatus.SUCCESS)))
    s.submit_capture(type("Capture", (), {"ok": True, "text": "yes"})())
    assert len(seen) == 1
    assert seen[0].task.task_id == "task-123"


def test_normal_yes_does_not_enter_approval_resolver(monkeypatch) -> None:
    s = session()
    seen = []
    monkeypatch.setattr("agent_control.session.api.resolve_task", lambda _: None)
    monkeypatch.setattr(s, "_converse", lambda task: seen.append(task) or Turn(task, "hello", None))
    turn = s.submit("yes")
    assert seen and turn.task.text == "yes"


def test_only_one_microphone_capture_can_own_listener() -> None:
    assert mic._CAPTURE_LOCK.acquire(blocking=False)
    try:
        result = mic.record_utterance(max_seconds=0.1)
        assert result.stopped_by == "listener_busy"
        assert not result.ok
    finally:
        mic._CAPTURE_LOCK.release()


def test_tts_can_finish_and_next_listener_can_start(monkeypatch) -> None:
    calls = []
    started = threading.Event()

    def backend(text, *, config):
        calls.append(text)
        started.set()
        time.sleep(0.02)
        return Utterance(text=text, spoken=True)

    speaker = Speaker(TTSConfig(enabled=True), backend=backend)
    speaker.say("hello")
    assert started.wait(1)
    assert speaker.wait_until_idle(1)
    assert speaker._worker is not None and speaker._worker.is_alive()
    speaker.say("hey")
    assert speaker.wait_until_idle(1)
    assert calls == ["hello", "hey"]
    speaker.close()


def test_same_site_resource_key_is_reused_and_sites_are_distinct() -> None:
    s = session()
    youtube = object()
    whatsapp = object()
    s._browser_tasks["youtube"] = youtube
    s._browser_tasks["whatsapp"] = whatsapp
    assert s._browser_for_task("task-a", resource_key="youtube") is youtube
    assert s._browser_for_task("task-b", resource_key="whatsapp") is whatsapp
    assert youtube is not whatsapp
