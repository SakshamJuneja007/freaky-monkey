from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BrowserActionKind(str, Enum):
    """Supported semantic actions for the browser skill."""

    OPEN_URL = "browser_open_url"
    SEARCH = "browser_search"
    CLICK = "browser_click"
    TYPE = "browser_type"
    PRESS_KEY = "browser_press_key"
    SCROLL = "browser_scroll"
    SELECT = "browser_select"
    WAIT = "browser_wait"
    CLOSE_TAB = "browser_close_tab"


@dataclass(frozen=True)
class BrowserAction:
    """A semantic browser action.

    This object intentionally describes intent rather than implementation.

    Examples:

        BrowserAction(
            kind=BrowserActionKind.OPEN_URL,
            params={"url": "https://www.youtube.com"},
        )

        BrowserAction(
            kind=BrowserActionKind.SEARCH,
            params={"query": "Killshot"},
        )

    The executor is responsible for translating this semantic action into
    concrete browser automation.
    """

    kind: BrowserActionKind
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BrowserActionKind):
            raise TypeError(
                "kind must be an instance of BrowserActionKind"
            )

        if not isinstance(self.params, dict):
            raise TypeError("params must be a dictionary")

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "kind": self.kind.value,
            "params": dict(self.params),
            "rationale": self.rationale,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "BrowserAction":
        """Create a BrowserAction from JSON-compatible data."""

        if not isinstance(payload, dict):
            raise TypeError("browser action payload must be a dictionary")

        raw_kind = payload.get("kind")

        try:
            kind = BrowserActionKind(raw_kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported browser action kind: {raw_kind!r}"
            ) from exc

        params = payload.get("params", {})

        if not isinstance(params, dict):
            raise TypeError("browser action params must be a dictionary")

        rationale = payload.get("rationale", "")

        return cls(
            kind=kind,
            params=params,
            rationale=str(rationale),
        )


def is_browser_action_kind(kind: str) -> bool:
    """Return True if *kind* is a supported browser action."""

    try:
        BrowserActionKind(kind)
        return True
    except (TypeError, ValueError):
        return False


BROWSER_ACTION_KINDS = frozenset(
    action_kind.value
    for action_kind in BrowserActionKind
)