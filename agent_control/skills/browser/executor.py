from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .actions import BrowserAction, BrowserActionKind


class BrowserBackend(Protocol):
    """Interface implemented by a concrete browser automation backend."""

    def open_url(self, url: str) -> Any:
        ...

    def search(self, query: str) -> Any:
        ...

    def click(self, target: str) -> Any:
        ...

    def type_text(self, text: str) -> Any:
        ...

    def press_key(self, key: str) -> Any:
        ...

    def scroll(self, amount: int) -> Any:
        ...

    def select(self, target: str, value: str) -> Any:
        ...

    def wait(self, seconds: float) -> Any:
        ...

    def close_tab(self) -> Any:
        ...


@dataclass(frozen=True)
class BrowserExecutionResult:
    """Result returned after executing one browser action."""

    action: BrowserAction
    ok: bool
    detail: str = ""
    value: Any = None


class BrowserExecutor:
    """Execute semantic browser actions through a browser backend.

    This class does not decide whether an action is safe.
    Policy checks must happen before an action reaches the executor.
    """

    def __init__(self, backend: BrowserBackend) -> None:
        self._backend = backend

    def execute(self, action: BrowserAction) -> BrowserExecutionResult:
        """Execute one browser action."""

        try:
            value = self._dispatch(action)

            return BrowserExecutionResult(
                action=action,
                ok=True,
                detail=f"executed {action.kind.value}",
                value=value,
            )

        except Exception as exc:
            return BrowserExecutionResult(
                action=action,
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )

    def _dispatch(self, action: BrowserAction) -> Any:
        """Dispatch a semantic action to the concrete backend."""

        params = action.params
        kind = action.kind

        if kind is BrowserActionKind.OPEN_URL:
            return self._backend.open_url(
                self._require_str(params, "url")
            )

        if kind is BrowserActionKind.SEARCH:
            return self._backend.search(
                self._require_str(params, "query")
            )

        if kind is BrowserActionKind.CLICK:
            return self._backend.click(
                self._require_str(params, "target")
            )

        if kind is BrowserActionKind.TYPE:
            return self._backend.type_text(
                self._require_str(params, "text")
            )

        if kind is BrowserActionKind.PRESS_KEY:
            return self._backend.press_key(
                self._require_str(params, "key")
            )

        if kind is BrowserActionKind.SCROLL:
            amount = params.get("amount")

            if not isinstance(amount, int):
                raise TypeError(
                    "browser_scroll requires integer parameter 'amount'"
                )

            return self._backend.scroll(amount)

        if kind is BrowserActionKind.SELECT:
            return self._backend.select(
                self._require_str(params, "target"),
                self._require_str(params, "value"),
            )

        if kind is BrowserActionKind.WAIT:
            seconds = params.get("seconds")

            if not isinstance(seconds, (int, float)):
                raise TypeError(
                    "browser_wait requires numeric parameter 'seconds'"
                )

            if seconds < 0:
                raise ValueError(
                    "browser_wait parameter 'seconds' must not be negative"
                )

            return self._backend.wait(float(seconds))

        if kind is BrowserActionKind.CLOSE_TAB:
            return self._backend.close_tab()

        raise ValueError(
            f"unsupported browser action: {kind.value}"
        )

    @staticmethod
    def _require_str(
        params: dict[str, Any],
        name: str,
    ) -> str:
        """Return a required non-empty string parameter."""

        value = params.get(name)

        if not isinstance(value, str):
            raise TypeError(
                f"browser action requires string parameter {name!r}"
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"browser action parameter {name!r} must not be empty"
            )

        return value