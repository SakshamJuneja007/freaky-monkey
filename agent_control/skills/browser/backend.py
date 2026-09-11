"""Real Chrome backend for the DEIMOS Browser Skill.

The backend owns an isolated, persistent DEIMOS Chrome profile and exposes a
small synchronous Playwright adapter to the skill layer. It deliberately uses
Chrome DevTools Protocol over loopback rather than attaching to the user's
normal Chrome profile. That gives DEIMOS persistent logins without silently
reading the user's personal browser profile.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_DEFAULT_PORT = 9222
_CONNECT_TIMEOUT_S = 15.0
_NAVIGATION_TIMEOUT_MS = 30_000


def normalize_url(url: str) -> str:
    """Normalize a user URL without changing its meaning."""
    value = str(url or "").strip()
    if not value:
        raise ValueError("browser URL must not be empty")
    if "://" not in value:
        value = "https://" + value
    return value


def canonical_url(url: str) -> str:
    """Canonicalize URLs for navigation/verification comparisons.

    ``www.example.com`` and ``example.com`` are treated as the same host, and
    an empty root path is treated as ``/``. Query parameters and non-root paths
    remain significant.
    """
    value = normalize_url(url)
    parts = urlsplit(value)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    port = parts.port
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    netloc = host
    if port and not default_port:
        netloc = f"{host}:{port}"

    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"

    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def _chrome_executable() -> str:
    configured = os.environ.get("DEIMOS_CHROME_EXECUTABLE", "").strip()
    if configured and Path(configured).is_file():
        return configured

    candidates: list[Path] = []
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "")
        program = os.environ.get("PROGRAMFILES", "")
        program86 = os.environ.get("PROGRAMFILES(X86)", "")
        candidates.extend(
            [
                Path(local) / "Google/Chrome/Application/chrome.exe",
                Path(program) / "Google/Chrome/Application/chrome.exe",
                Path(program86) / "Google/Chrome/Application/chrome.exe",
            ]
        )
    elif os.name == "darwin":
        candidates.append(
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        )
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            found = shutil.which(name)
            if found:
                return found

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    found = shutil.which("chrome") or shutil.which("google-chrome")
    if found:
        return found

    raise RuntimeError(
        "Google Chrome executable was not found; set DEIMOS_CHROME_EXECUTABLE"
    )


def _profile_dir() -> Path:
    configured = os.environ.get("DEIMOS_CHROME_PROFILE", "").strip()
    if configured:
        path = Path(configured).expanduser()
    elif os.name == "nt":
        path = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "DEIMOS" / "ChromeProfile"
    else:
        path = Path.home() / ".deimos" / "chrome-profile"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _port() -> int:
    raw = os.environ.get("DEIMOS_CHROME_DEBUG_PORT", str(_DEFAULT_PORT)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("DEIMOS_CHROME_DEBUG_PORT must be an integer") from exc
    if not 1024 <= value <= 65535:
        raise ValueError("DEIMOS_CHROME_DEBUG_PORT must be between 1024 and 65535")
    return value


class PlaywrightChromeBackend:
    """Persistent Chrome backend used by the DEIMOS Browser Skill."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._process: subprocess.Popen | None = None

    def _endpoint(self) -> str:
        return f"http://127.0.0.1:{_port()}"

    def _debug_ready(self) -> bool:
        try:
            with urllib.request.urlopen(
                self._endpoint() + "/json/version", timeout=0.75
            ) as response:
                return response.status == 200
        except Exception:
            return False

    def _launch_chrome(self) -> None:
        executable = _chrome_executable()
        profile = _profile_dir()
        port = _port()

        argv = [
            executable,
            f"--remote-debugging-address=127.0.0.1",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-session-crashed-bubble",
        ]

        self._process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )

    def _connect(self) -> None:
        if self._context is not None and self._page is not None:
            return

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for Phase 3 browser control. "
                "Install it with 'pip install playwright'."
            ) from exc

        if not self._debug_ready():
            self._launch_chrome()

        deadline = time.monotonic() + _CONNECT_TIMEOUT_S
        while not self._debug_ready():
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"Chrome exited while starting (code {self._process.returncode})"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Chrome DevTools endpoint did not become ready at {self._endpoint()}"
                )
            time.sleep(0.2)

        if self._playwright is None:
            self._playwright = sync_playwright().start()

        self._browser = self._playwright.chromium.connect_over_cdp(self._endpoint())
        contexts = self._browser.contexts
        if not contexts:
            raise RuntimeError("Chrome exposed no browser context")
        self._context = contexts[0]

        pages = self._context.pages
        self._page = pages[-1] if pages else self._context.new_page()

    def _ensure_page(self) -> Any:
        with self._lock:
            self._connect()
            if self._page is None or self._page.is_closed():
                pages = [p for p in self._context.pages if not p.is_closed()]
                self._page = pages[-1] if pages else self._context.new_page()
            return self._page

    def _select_target_page(self) -> Any:
        page = self._ensure_page()
        pages = [p for p in self._context.pages if not p.is_closed()]
        for candidate in reversed(pages):
            if candidate.url and candidate.url != "about:blank":
                self._page = candidate
                break
        return self._page

    def open_url(self, url: str) -> dict[str, Any]:
        target = normalize_url(url)
        with self._lock:
            page = self._select_target_page()
            current = page.url or ""
            if current and canonical_url(current) == canonical_url(target):
                return {"url": current, "already_open": True}

            try:
                page.goto(
                    target,
                    wait_until="domcontentloaded",
                    timeout=_NAVIGATION_TIMEOUT_MS,
                )
            except Exception as exc:
                # A navigation timeout can still leave the requested page active.
                actual = page.url or ""
                if actual and canonical_url(actual) == canonical_url(target):
                    return {
                        "url": actual,
                        "navigation_timeout": True,
                        "detail": str(exc),
                    }
                raise

            result: dict[str, Any] = {"url": page.url, "title": page.title()}

            # A YouTube search URL represents a request to find something, not
            # a request to merely leave the user on the results grid.  For the
            # high-level conversational "play X on YouTube" flow, select the
            # first normal video result after the search page is loaded.
            parts = urlsplit(page.url or target)
            host = (parts.hostname or "").lower().removeprefix("www.")
            query_values = dict(parse_qsl(parts.query, keep_blank_values=True))
            if host in {"youtube.com", "m.youtube.com"} and parts.path.rstrip("/") == "/results" and query_values.get("search_query"):
                try:
                    locator = page.locator("ytd-video-renderer a#video-title").first
                    locator.wait_for(state="visible", timeout=8_000)
                    title = locator.get_attribute("title") or (locator.inner_text(timeout=2_000) or "").strip()
                    locator.click(timeout=8_000)
                    result.update({
                        "played_first_result": True,
                        "video_title": title,
                        "url": page.url,
                    })
                except Exception as exc:
                    # Keep navigation itself successful. The caller can still
                    # see the search page instead of turning a missing result
                    # into a browser crash.
                    result.update({
                        "played_first_result": False,
                        "video_selection_error": f"{type(exc).__name__}: {exc}",
                    })

            return result

    def search(self, query: str) -> dict[str, Any]:
        from urllib.parse import quote_plus
        value = str(query or "").strip()
        if not value:
            raise ValueError("search query must not be empty")
        return self.open_url("https://www.google.com/search?q=" + quote_plus(value))

    def click(self, target: str) -> Any:
        page = self._ensure_page()
        value = str(target or "").strip()
        if not value:
            raise ValueError("click target must not be empty")

        selectors = [
            page.get_by_role("button", name=value, exact=True),
            page.get_by_role("link", name=value, exact=True),
            page.get_by_text(value, exact=True),
            page.locator(value),
        ]
        last_error: Exception | None = None
        for locator in selectors:
            try:
                locator.first.click(timeout=5_000)
                return {"target": value}
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"could not click {value!r}: {last_error}")

    def type_text(self, text: str) -> Any:
        page = self._ensure_page()
        value = str(text or "")
        page.keyboard.type(value)
        return {"text_length": len(value)}

    def press_key(self, key: str) -> Any:
        value = str(key or "").strip()
        if not value:
            raise ValueError("key must not be empty")
        self._ensure_page().keyboard.press(value)
        return {"key": value}

    def scroll(self, amount: int) -> Any:
        self._ensure_page().mouse.wheel(0, int(amount))
        return {"amount": int(amount)}

    def select(self, target: str, value: str) -> Any:
        self._ensure_page().locator(str(target)).first.select_option(str(value))
        return {"target": str(target), "value": str(value)}

    def wait(self, seconds: float) -> Any:
        self._ensure_page().wait_for_timeout(int(float(seconds) * 1000))
        return {"seconds": float(seconds)}

    def close_tab(self) -> Any:
        page = self._ensure_page()
        page.close()
        self._page = None
        self._ensure_page()
        return {"closed": True}

    # Read-only observation API used by BrowserVerifier.
    def current_url(self) -> str:
        return str(self._select_target_page().url or "")

    def page_title(self) -> str:
        return str(self._select_target_page().title() or "")

    def page_contains_text(self, text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            raise ValueError("text must not be empty")
        body = self._select_target_page().locator("body").inner_text(timeout=5_000)
        return value.casefold() in body.casefold()

    def tabs(self) -> list[dict[str, Any]]:
        self._ensure_page()
        result = []
        for index, page in enumerate(self._context.pages):
            if page.is_closed():
                continue
            result.append(
                {
                    "index": index,
                    "active": page is self._page,
                    "url": page.url,
                    "title": page.title(),
                }
            )
        return result

    def close_connection(self) -> None:
        """Detach Playwright without closing the user's DEIMOS Chrome."""
        with self._lock:
            if self._browser is not None:
                try:
                    self._browser.close()
                except Exception:
                    pass
            self._browser = None
            self._context = None
            self._page = None
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
            self._playwright = None
