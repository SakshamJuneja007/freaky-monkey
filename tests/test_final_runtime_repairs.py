from __future__ import annotations

import threading
import time

from agent_control.fast_interaction import FastRoute, _indexed_candidates, classify_fast
from agent_control.response import Narrator
from agent_control.session import Capture, PendingApproval, Session
from agent_control.skills.browser.backend import BrowserElement, BrowserObservation, BrowserSkillAdapter
from agent_control.skills.browser.skill import BrowserSkill
from agent_control.skills.browser.actions import BrowserAction, BrowserActionKind


def el(ref, name, raw=None):
    raw = dict(raw or {})
    raw.setdefault("duration", "3:45")
    return BrowserElement(ref, role="link", name=name, raw=raw)


def youtube_obs(elements, raw=None):
    return BrowserObservation(
        generation=7, session="s1", tab_id=1,
        url="https://www.youtube.com/results?search_query=song",
        text="", elements=tuple(elements), raw=raw or {},
    )


def test_youtube_short_is_hard_excluded_even_when_ranked_first():
    elements = (
        el("@short", "Song", {"href": "https://www.youtube.com/shorts/x"}),
        el("@normal", "Song Official Video", {"href": "https://www.youtube.com/watch?v=y"}),
    )
    candidates = _indexed_candidates(elements, ("link",), youtube_only=True, observation=youtube_obs(elements))
    assert [e.ref for e in candidates] == ["@normal"]


def test_youtube_short_metadata_is_hard_excluded():
    elements = (el("@short", "Song"), el("@normal", "Song Official Video"))
    obs = youtube_obs(elements, {"elements": [
        {"ref": "@short", "type": "SHORTS", "url": "https://www.youtube.com/watch?v=x"},
        {"ref": "@normal", "type": "video", "url": "https://www.youtube.com/watch?v=y"},
    ]})
    assert [e.ref for e in _indexed_candidates(elements, ("link",), youtube_only=True, observation=obs)] == ["@normal"]


def test_lyrics_short_is_rejected_before_ranking():
    elements = (
        el("@short", "Song Official Lyrics", {"href": "https://www.youtube.com/shorts/x"}),
        el("@normal", "Song Official Video", {"href": "https://www.youtube.com/watch?v=y"}),
    )
    ranked = BrowserSkillAdapter._rank_song_candidates("Song", elements, observation=youtube_obs(elements))
    assert ranked and ranked[0][1].ref == "@normal"


def test_play_song_phrase_routes_to_dedicated_browser_workflow():
    assert classify_fast("play do i wanna know").route.action_kind == "browser_play_song"
    assert classify_fast("open youtube and play do i wanna know").route.params["query"] == "do i wanna know"


def test_browser_skill_rejects_planner_click_on_youtube_short_before_execution():
    elements = (el("@short", "Song", {"href": "https://www.youtube.com/shorts/x"}),)
    browser = BrowserSkillAdapter()
    browser.session = "s1"
    browser._last_observation = youtube_obs(elements)
    action = BrowserAction(BrowserActionKind.CLICK, {"target": "@short"})
    try:
        BrowserSkill(browser).adapt_action(action)
    except Exception as exc:
        assert getattr(exc, "code", "") == "youtube_short_rejected"
    else:
        raise AssertionError("planner click on YouTube Short was not rejected")


def test_approval_yes_resumes_same_task_and_never_routes_normally():
    output = []
    s = Session(narrator=Narrator.build(enabled=False, write=output.append))
    s.pending_approval = PendingApproval(
        request="send message to papa saying hello",
        action={"kind": "whatsapp_send_message", "params": {"recipient": "papa", "message": "hello"}},
        asked_at=time.time(), task_id="task-42", goal="send message to papa saying hello",
    )
    seen = []
    def fake_run(prepared):
        seen.append(prepared.task.task_id)
        from agent_control.session import Turn
        from agent_control.api import AgentResult, TaskStatus
        return Turn(prepared.task, "sent", AgentResult(request=prepared.task.text, task_id=prepared.task.task_id, status=TaskStatus.SUCCESS))
    s._run = fake_run
    turn = s.submit("yes")
    assert seen == ["task-42"]
    assert turn.task.task_id == "task-42"
    assert s.pending_approval is None
    s.close()


def test_approval_ambiguous_does_not_create_task():
    s = Session(narrator=Narrator.build(enabled=False))
    s.pending_approval = PendingApproval("send message to papa saying hello", {"kind": "whatsapp_send_message", "params": {"recipient": "papa", "message": "hello"}}, time.time(), task_id="task-42")
    before = len(s.background_tasks())
    turn = s.submit_background("maybe")
    assert turn == "task-42"
    assert len(s.background_tasks()) == before
    assert s.pending_approval is not None
    s.close()


def test_voice_capture_can_queue_background_task_without_blocking(monkeypatch):
    s = Session(narrator=Narrator.build(enabled=False))
    gate = threading.Event()
    started = threading.Event()
    def fake_submit(raw, *, source="text"):
        started.set()
        gate.wait(1)
        from agent_control.session import Turn, UserTask
        return Turn(UserTask(raw=raw, text=raw, source=source, task_id="x", status="accepted"), "ok", None)
    monkeypatch.setattr(s, "submit", fake_submit)
    began = time.time()
    task_id = s.submit_capture(Capture(text="hey", ok=True, seconds=1), background=True)
    elapsed = time.time() - began
    assert isinstance(task_id, str)
    assert elapsed < 0.2
    assert started.wait(1)
    gate.set()
    s.close()


def test_obvious_conversation_never_creates_background_task(monkeypatch):
    s = Session(narrator=Narrator.build(enabled=False))
    monkeypatch.setattr(s, "_conversation_background_run", lambda event: None)
    for text in ("hey", "what's up", "what are you doing"):
        before = len(s.background_tasks())
        returned = s.submit_background(text)
        assert returned.startswith("input-")
        assert len(s.background_tasks()) == before
    s.close()


def test_normal_yes_without_approval_is_conversation(monkeypatch):
    s = Session(narrator=Narrator.build(enabled=False))
    seen = []
    monkeypatch.setattr(s, "_conversation_background_run", lambda event: seen.append(event.text))
    returned = s.submit_background("yes")
    assert returned.startswith("input-")
    deadline = time.time() + 1
    while time.time() < deadline and not seen:
        time.sleep(0.01)
    assert seen == ["yes"]
    assert s.background_tasks() == []
    s.close()


def test_tts_worker_has_single_lifecycle():
    from agent_control.speech.tts import Speaker, TTSConfig, Utterance
    calls = []
    def backend(text, *, config):
        calls.append(text)
        return Utterance(text=text, spoken=True)
    speaker = Speaker(TTSConfig(enabled=True), backend=backend)
    speaker.say("hello")
    assert speaker.wait_until_idle(1)
    first = speaker._worker
    speaker.say("hey")
    assert speaker.wait_until_idle(1)
    assert speaker._worker is first
    assert calls == ["hello", "hey"]
    speaker.close()


def test_voice_approval_is_owned_by_pending_task(monkeypatch):
    s = Session(narrator=Narrator.build(enabled=False))
    from agent_control.api import AgentResult, TaskStatus
    s.pending_approval = PendingApproval(
        request="send message to papa saying hello",
        action={"kind": "whatsapp_send_message", "params": {"recipient": "papa", "message": "hello"}},
        asked_at=time.time(), task_id="task-voice", goal="send message to papa saying hello",
        background_task_id="bg-1",
    )
    s._background["bg-1"] = type("R", (), {"state": "WAITING_FOR_APPROVAL", "future": None, "result": None, "failure": "", "started_at": None, "finished_at": None, "kind": "background"})()
    captured = []
    def fake_resume(bg_id, prepared):
        captured.append((bg_id, prepared.task.task_id, prepared.approved_action["params"]["message"]))
    monkeypatch.setattr(s, "_resume_approved_background", fake_resume)
    turn = s.submit_capture(Capture(text="yes", ok=True), background=True)
    assert turn.task.task_id == "task-voice"
    assert captured == [("bg-1", "task-voice", "hello")]
    assert s.pending_approval is None
    s.close()


def test_chat_defaults_to_speech_and_can_be_explicitly_disabled():
    import main
    parser = main.build_parser()
    assert parser.parse_args(["chat"]).speak is True
    assert parser.parse_args(["chat", "--no-speak"]).speak is False
