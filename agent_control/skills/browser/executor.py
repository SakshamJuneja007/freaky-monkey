from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .actions import BrowserAction, BrowserActionKind


class BrowserBackend(Protocol):
    def open_url(self, url: str) -> Any: ...
    def search(self, query: str) -> Any: ...
    def current_url(self) -> str: ...
    def page_title(self) -> str: ...
    def page_text(self) -> str: ...
    def snapshot(self) -> Any: ...
    def observe(self) -> Any: ...
    def click(self, target: str) -> Any: ...
    def type_text(self, target: str, text: str) -> Any: ...
    def press_key(self, key: str, target: str | None = None) -> Any: ...
    def scroll(self, amount: int) -> Any: ...
    def scroll_to(self, target: str) -> Any: ...
    def select(self, target: str, value: str) -> Any: ...
    def upload_file(self, target: str, file_path: str, *, mode: str | None = None) -> Any: ...
    def download(self, target: str, output_path: str, *, overwrite: bool = False) -> Any: ...
    def wait(self, seconds: float) -> Any: ...
    def close_tab(self, tab_id: int | None = None) -> Any: ...
    def list_tabs(self, scope: str = "all") -> Any: ...
    def create_tab(self, url: str | None = None) -> Any: ...
    def select_tab(self, tab_id: int) -> Any: ...
    def borrow_tab(self, tab_id: int) -> Any: ...
    def return_tab(self, tab_id: int) -> Any: ...
    def play_song(self, query: str) -> Any: ...
    def apply_job(self, job_url: str, resume_path: str, answers: dict[str, str] | None = None, *, submit: bool = True) -> Any: ...


@dataclass(frozen=True)
class BrowserExecutionResult:
    action: BrowserAction
    ok: bool
    detail: str = ""
    value: Any = None


class BrowserExecutor:
    def __init__(self, backend: BrowserBackend) -> None:
        self._backend = backend

    def execute(self, action: BrowserAction) -> BrowserExecutionResult:
        try:
            value = self._dispatch(action)
            return BrowserExecutionResult(action, True, f"executed {action.kind.value}", value)
        except Exception as exc:
            return BrowserExecutionResult(action, False, f"{type(exc).__name__}: {exc}")

    def _dispatch(self, action: BrowserAction) -> Any:
        p = action.params
        k = action.kind
        if k is BrowserActionKind.OPEN_URL:
            return self._backend.open_url(self._require_str(p, "url"))
        if k is BrowserActionKind.SEARCH:
            return self._backend.search(self._require_str(p, "query"))
        if k is BrowserActionKind.GET_CURRENT_PAGE:
            return {"url": self._backend.current_url(), "title": self._backend.page_title()}
        if k is BrowserActionKind.LIST_TABS:
            return self._backend.list_tabs(str(p.get("scope", "all")))
        if k is BrowserActionKind.OPEN_NEW_TAB:
            return self._backend.create_tab(p.get("url"))
        if k is BrowserActionKind.SWITCH_TAB:
            return self._backend.select_tab(self._require_int(p, "tab_id"))
        if k is BrowserActionKind.CLOSE_TAB:
            tab = p.get("tab_id")
            return self._backend.close_tab(int(tab) if tab is not None else None)
        if k is BrowserActionKind.GO_BACK:
            return self._backend.go_back()
        if k is BrowserActionKind.GO_FORWARD:
            return self._backend.go_forward()
        if k is BrowserActionKind.REFRESH:
            return self._backend.refresh()
        if k is BrowserActionKind.PAGE_STATE:
            return self._backend.observe()
        if k is BrowserActionKind.EXTRACT_TEXT:
            return {"text": self._backend.page_text()}
        if k is BrowserActionKind.CLICK:
            return self._backend.click(self._require_str(p, "target"))
        if k is BrowserActionKind.TYPE:
            return self._backend.type_text(self._require_str(p, "target"), self._require_str(p, "text", allow_empty=True))
        if k is BrowserActionKind.PRESS_KEY:
            return self._backend.press_key(self._require_str(p, "key"), p.get("target"))
        if k is BrowserActionKind.SCROLL:
            return self._backend.scroll(self._require_int(p, "amount"))
        if k is BrowserActionKind.SCROLL_TO:
            return self._backend.scroll_to(self._require_str(p, "target"))
        if k is BrowserActionKind.SELECT:
            return self._backend.select(self._require_str(p, "target"), self._require_str(p, "value"))
        if k is BrowserActionKind.UPLOAD_FILE:
            return self._backend.upload_file(self._require_str(p, "target"), self._require_str(p, "file_path"), mode=p.get("mode"))
        if k is BrowserActionKind.DOWNLOAD_FILE:
            return self._backend.download(self._require_str(p, "target"), self._require_str(p, "output_path"), overwrite=bool(p.get("overwrite", False)))
        if k is BrowserActionKind.WAIT:
            seconds = p.get("seconds")
            if not isinstance(seconds, (int, float)) or seconds < 0:
                raise ValueError("browser_wait requires non-negative numeric seconds")
            return self._backend.wait(float(seconds))
        if k is BrowserActionKind.BORROW_TAB:
            return self._backend.borrow_tab(self._require_int(p, "tab_id"))
        if k is BrowserActionKind.RETURN_TAB:
            return self._backend.return_tab(self._require_int(p, "tab_id"))
        if k is BrowserActionKind.PLAY_SONG:
            return self._backend.play_song(self._require_str(p, "query"))
        if k is BrowserActionKind.APPLY_JOB:
            answers = p.get("answers", {})
            if not isinstance(answers, dict):
                raise TypeError("browser_apply_job 'answers' must be a dictionary")
            return self._backend.apply_job(
                self._require_str(p, "job_url"),
                self._require_str(p, "resume_path"),
                {str(k): str(v) for k, v in answers.items()},
                submit=bool(p.get("submit", True)),
            )
        raise ValueError(f"unsupported browser action: {k.value}")

    @staticmethod
    def _require_str(params: dict[str, Any], name: str, *, allow_empty: bool = False) -> str:
        value = params.get(name)
        if not isinstance(value, str):
            raise TypeError(f"browser action requires string parameter {name!r}")
        if not allow_empty and not value.strip():
            raise ValueError(f"browser action parameter {name!r} must not be empty")
        return value

    @staticmethod
    def _require_int(params: dict[str, Any], name: str) -> int:
        value = params.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"browser action requires integer parameter {name!r}")
        return value
