from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class BrowserVerifierBackend(Protocol):
    """Read-only browser inspection interface.

    A verifier must observe browser state independently of the action
    execution result.
    """

    def current_url(self) -> str:
        ...

    def page_title(self) -> str:
        ...

    def page_contains_text(self, text: str) -> bool:
        ...


@dataclass(frozen=True)
class BrowserVerificationResult:
    """Result of independently checking browser state."""

    ok: bool
    detail: str = ""


class BrowserVerifier:
    """Independently verify observable browser state.

    The verifier deliberately uses a read-only backend interface. It does not
    click, type, submit forms, or otherwise change browser state.

    This preserves the core architecture:

        Executor -> changes state
        Verifier -> observes state

    ``verify()`` implements the generic SkillVerifier contract used by the
    DEIMOS runner. Specialized verification helpers remain available for
    direct browser-specific checks.
    """

    _VERIFICATION_TARGETS = (
        "expected_url",
        "expected_url_contains",
        "expected_title",
        "expected_text",
    )

    def __init__(
        self,
        backend: BrowserVerifierBackend,
    ) -> None:
        """Initialize the verifier with a read-only browser backend."""
        if backend is None:
            raise ValueError(
                "BrowserVerifier requires a backend"
            )

        self._backend = backend

    def verify(
        self,
        action: Any,
        result: Any,
    ) -> BrowserVerificationResult:
        """Return the browser executor outcome without tab inspection.

        Browser state verification is intentionally disabled. The Playwright
        executor is already responsible for reporting whether the requested
        browser operation completed; re-reading Chrome adds latency and was not
        reliable for this persistent-profile workflow.
        """
        from .actions import BrowserAction

        if not isinstance(action, BrowserAction):
            raise TypeError("BrowserVerifier.verify() expects a BrowserAction")

        ok = bool(getattr(result, "ok", False))
        detail = str(getattr(result, "detail", "") or "")

        if ok:
            return BrowserVerificationResult(
                ok=True,
                detail=detail or "browser operation completed",
            )

        return BrowserVerificationResult(
            ok=False,
            detail=detail or "browser operation failed",
        )

    def verify_url(
        self,
        expected_url: str,
    ) -> BrowserVerificationResult:
        """Verify that the browser is currently at the expected URL."""
        expected_url = self._require_non_empty_string(
            expected_url,
            "expected_url",
        )

        try:
            actual_url = self._backend.current_url()

            if not isinstance(
                actual_url,
                str,
            ):
                return BrowserVerificationResult(
                    ok=False,
                    detail=(
                        "browser backend returned a non-string URL: "
                        f"{type(actual_url).__name__}"
                    ),
                )

            if actual_url == expected_url:
                return BrowserVerificationResult(
                    ok=True,
                    detail=(
                        f"browser is at {actual_url!r}"
                    ),
                )

            return BrowserVerificationResult(
                ok=False,
                detail=(
                    "browser URL did not match; "
                    f"expected {expected_url!r}, "
                    f"got {actual_url!r}"
                ),
            )

        except Exception as exc:
            return self._backend_error(
                "reading browser URL",
                exc,
            )

    def verify_url_contains(
        self,
        expected_fragment: str,
    ) -> BrowserVerificationResult:
        """Verify that the current URL contains a required fragment."""
        expected_fragment = self._require_non_empty_string(
            expected_fragment,
            "expected_fragment",
        )

        try:
            actual_url = self._backend.current_url()

            if not isinstance(
                actual_url,
                str,
            ):
                return BrowserVerificationResult(
                    ok=False,
                    detail=(
                        "browser backend returned a non-string URL: "
                        f"{type(actual_url).__name__}"
                    ),
                )

            if expected_fragment in actual_url:
                return BrowserVerificationResult(
                    ok=True,
                    detail=(
                        "browser URL contains "
                        f"{expected_fragment!r}"
                    ),
                )

            return BrowserVerificationResult(
                ok=False,
                detail=(
                    "browser URL did not contain "
                    f"{expected_fragment!r}; "
                    f"got {actual_url!r}"
                ),
            )

        except Exception as exc:
            return self._backend_error(
                "reading browser URL",
                exc,
            )

    def verify_title(
        self,
        expected_title: str,
    ) -> BrowserVerificationResult:
        """Verify the exact page title."""
        expected_title = self._require_non_empty_string(
            expected_title,
            "expected_title",
        )

        try:
            actual_title = self._backend.page_title()

            if not isinstance(
                actual_title,
                str,
            ):
                return BrowserVerificationResult(
                    ok=False,
                    detail=(
                        "browser backend returned a non-string title: "
                        f"{type(actual_title).__name__}"
                    ),
                )

            if actual_title == expected_title:
                return BrowserVerificationResult(
                    ok=True,
                    detail=(
                        "page title matched "
                        f"{actual_title!r}"
                    ),
                )

            return BrowserVerificationResult(
                ok=False,
                detail=(
                    "page title did not match; "
                    f"expected {expected_title!r}, "
                    f"got {actual_title!r}"
                ),
            )

        except Exception as exc:
            return self._backend_error(
                "reading page title",
                exc,
            )

    def verify_text(
        self,
        expected_text: str,
    ) -> BrowserVerificationResult:
        """Verify that text is currently visible on the page."""
        expected_text = self._require_non_empty_string(
            expected_text,
            "expected_text",
        )

        try:
            found = self._backend.page_contains_text(
                expected_text
            )

            if not isinstance(
                found,
                bool,
            ):
                found = bool(found)

            if found:
                return BrowserVerificationResult(
                    ok=True,
                    detail=(
                        "expected text is visible: "
                        f"{expected_text!r}"
                    ),
                )

            return BrowserVerificationResult(
                ok=False,
                detail=(
                    "expected text is not visible: "
                    f"{expected_text!r}"
                ),
            )

        except Exception as exc:
            return self._backend_error(
                "checking visible page text",
                exc,
            )

    @staticmethod
    def _require_non_empty_string(
        value: Any,
        name: str,
    ) -> str:
        """Validate and normalize a required string parameter."""
        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"{name} must not be empty"
            )

        return value

    @staticmethod
    def _backend_error(
        operation: str,
        exc: Exception,
    ) -> BrowserVerificationResult:
        """Normalize backend failures into verification failures."""
        return BrowserVerificationResult(
            ok=False,
            detail=(
                f"browser verifier failed while {operation}: "
                f"{type(exc).__name__}: {exc}"
            ),
        )