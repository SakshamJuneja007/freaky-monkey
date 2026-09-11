from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from .backend import canonical_url


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
        """Verify a browser action using independently observable state.

        The execution result is accepted because all skill verifiers share the
        same generic contract. It is not treated as independent evidence that
        the action succeeded.

        Browser actions may optionally provide one of these verification
        targets in their parameters:

        * ``expected_url``
        * ``expected_url_contains``
        * ``expected_title``
        * ``expected_text``

        If no skill-level verification target is provided, verification is
        deferred to the task-level checkpoint verifier.

        Returns:
            BrowserVerificationResult describing the independent observation.

        Raises:
            TypeError: If the supplied action is not a BrowserAction.
        """
        from .actions import BrowserAction

        if not isinstance(
            action,
            BrowserAction,
        ):
            raise TypeError(
                "BrowserVerifier.verify() expects "
                "a BrowserAction"
            )

        if not bool(
            getattr(
                result,
                "ok",
                False,
            )
        ):
            detail = getattr(
                result,
                "detail",
                "",
            )

            return BrowserVerificationResult(
                ok=False,
                detail=(
                    "browser execution did not succeed"
                    + (
                        f": {detail}"
                        if detail
                        else ""
                    )
                ),
            )

        params = getattr(
            action,
            "params",
            {},
        )

        if params is None:
            params = {}

        if not isinstance(
            params,
            dict,
        ):
            try:
                params = dict(params)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "browser action params must be mapping-like"
                ) from exc

        provided = [
            name
            for name in self._VERIFICATION_TARGETS
            if name in params
        ]

        if len(provided) > 1:
            return BrowserVerificationResult(
                ok=False,
                detail=(
                    "browser action provided multiple skill-level "
                    "verification targets: "
                    + ", ".join(provided)
                ),
            )

        if "expected_url" in params:
            return self.verify_url(
                params["expected_url"]
            )

        if "expected_url_contains" in params:
            return self.verify_url_contains(
                params["expected_url_contains"]
            )

        if "expected_title" in params:
            return self.verify_title(
                params["expected_title"]
            )

        if "expected_text" in params:
            return self.verify_text(
                params["expected_text"]
            )

        # Browser actions without an explicit target are deliberately not
        # subjected to exact URL verification.  We only perform a fresh,
        # read-only liveness check: there must be an active non-blank page.
        # This avoids making redirects, YouTube landing pages, or repeated
        # navigation depend on a brittle URL-equality check.
        try:
            actual_url = self._backend.current_url()
        except Exception as exc:
            return BrowserVerificationResult(
                ok=False,
                detail=f"could not observe active browser page: {exc}",
            )

        if isinstance(actual_url, str) and actual_url.strip() and actual_url != "about:blank":
            return BrowserVerificationResult(
                ok=True,
                detail=f"active browser page observed at {actual_url!r}",
            )

        return BrowserVerificationResult(
            ok=False,
            detail="browser has no active non-blank page",
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

            try:
                expected_canonical = canonical_url(expected_url)
                actual_canonical = canonical_url(actual_url)
            except (TypeError, ValueError) as exc:
                return BrowserVerificationResult(
                    ok=False,
                    detail=f"could not canonicalize browser URL: {exc}",
                )

            if actual_canonical == expected_canonical:
                return BrowserVerificationResult(
                    ok=True,
                    detail=(
                        f"browser is at {actual_url!r} "
                        f"(canonical match: {expected_canonical!r})"
                    ),
                )

            # A bare host request such as ``youtube.com`` legitimately redirects
            # from the root to a site-specific landing path. In that case the
            # browser has still reached the requested host. Path/query targets
            # remain exact.
            expected_parts = urlsplit(expected_canonical)
            actual_parts = urlsplit(actual_canonical)
            if (
                expected_parts.path == "/"
                and not expected_parts.query
                and not expected_parts.fragment
                and expected_parts.netloc == actual_parts.netloc
            ):
                return BrowserVerificationResult(
                    ok=True,
                    detail=(
                        f"browser reached requested host {actual_parts.netloc!r}; "
                        f"landing path is {actual_parts.path!r}"
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