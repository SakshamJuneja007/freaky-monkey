from __future__ import annotations

from agent_control.skills.browser.actions import BROWSER_ACTION_KINDS
from agent_control.skills.browser.skill import BrowserSkill
from agent_control.types import Action


class Backend:
    def __init__(self):
        self.url = "about:blank"
        self.title = ""
        self.text = ""
        self.calls = []

    def open_url(self, url): self.url = url; self.calls.append(("open", url)); return {}
    def search(self, q): self.calls.append(("search", q)); return {}
    def current_url(self): return self.url
    def page_title(self): return self.title
    def page_text(self): return self.text
    def snapshot(self): return {"text": self.text}
    def observe(self): return {"text": self.text}
    def click(self, t): self.calls.append(("click", t)); return {}
    def type_text(self, t, x): self.calls.append(("type", t, x)); return {}
    def press_key(self, k, t=None): self.calls.append(("press", k, t)); return {}
    def scroll(self, a): return {}
    def scroll_to(self, t): return {}
    def select(self, t, v): return {}
    def upload_file(self, t, f, mode=None): return {}
    def download(self, t, p, overwrite=False): return {}
    def wait(self, s): return {}
    def close_tab(self, tab_id=None): return {}
    def list_tabs(self, scope="all"): return {}
    def create_tab(self, url=None): return {}
    def select_tab(self, tab_id): return {}
    def borrow_tab(self, tab_id): return {}
    def return_tab(self, tab_id): return {}
    def play_song(self, q): self.calls.append(("play", q)); return {"query": q}
    def apply_job(self, job_url, resume_path, answers=None, *, submit=True):
        self.calls.append(("apply", job_url, resume_path, answers, submit))
        return {"submitted": submit}
    def close_session(self): pass


def test_browser_skill_exposes_real_browserskill_actions():
    skill = BrowserSkill(Backend())
    assert "browser_play_song" in BROWSER_ACTION_KINDS
    assert "browser_upload_file" in BROWSER_ACTION_KINDS
    assert skill.supports("browser_play_song")
    assert skill.supports("browser_upload_file")
    assert skill.supports("browser_apply_job")


def test_open_url_compatibility_is_preserved():
    skill = BrowserSkill(Backend())
    adapted = skill.adapt_action(Action(kind="open_url", params={"url": "https://example.com"}))
    assert adapted.kind.value == "browser_open_url"


def test_browser_job_application_is_semantic_action():
    skill = BrowserSkill(Backend())
    action = skill.adapt_action(Action(kind="browser_apply_job", params={
        "job_url": "https://example.com/jobs/1",
        "resume_path": "/tmp/resume.pdf",
        "answers": {"Full name": "Raju"},
        "submit": True,
    }))
    result = skill.executor().execute(action)
    assert result.ok is True
    assert result.value["submitted"] is True
