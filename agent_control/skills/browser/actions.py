from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BrowserActionKind(str, Enum):
    OPEN_URL = "browser_open_url"
    SEARCH = "browser_search"
    GET_CURRENT_PAGE = "browser_get_current_page"
    LIST_TABS = "browser_list_tabs"
    OPEN_NEW_TAB = "browser_open_new_tab"
    SWITCH_TAB = "browser_switch_tab"
    CLOSE_TAB = "browser_close_tab"
    GO_BACK = "browser_go_back"
    GO_FORWARD = "browser_go_forward"
    REFRESH = "browser_refresh"
    PAGE_STATE = "browser_page_state"
    EXTRACT_TEXT = "browser_extract_text"
    CLICK = "browser_click"
    TYPE = "browser_type"
    PRESS_KEY = "browser_press_key"
    SCROLL = "browser_scroll"
    SCROLL_TO = "browser_scroll_to"
    SELECT = "browser_select"
    UPLOAD_FILE = "browser_upload_file"
    DOWNLOAD_FILE = "browser_download_file"
    WAIT = "browser_wait"
    BORROW_TAB = "browser_borrow_tab"
    RETURN_TAB = "browser_return_tab"
    PLAY_SONG = "browser_play_song"
    APPLY_JOB = "browser_apply_job"


@dataclass(frozen=True)
class BrowserAction:
    kind: BrowserActionKind
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BrowserActionKind):
            raise TypeError("kind must be a BrowserActionKind")
        if not isinstance(self.params, dict):
            raise TypeError("browser action params must be a dictionary")

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "BrowserAction":
        if not isinstance(payload, dict):
            raise TypeError("browser action payload must be a dictionary")
        kind = BrowserActionKind(payload.get("kind"))
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise TypeError("browser action params must be a dictionary")
        return cls(kind, dict(params), str(payload.get("rationale", "")))

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "params": dict(self.params), "rationale": self.rationale}


BROWSER_ACTION_KINDS = frozenset(k.value for k in BrowserActionKind)


def is_browser_action_kind(kind: str) -> bool:
    return kind in BROWSER_ACTION_KINDS
