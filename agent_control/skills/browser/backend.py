"""Synchronous DEIMOS adapter for Tencent BrowserSkill.

Architecture
------------

DEIMOS
    -> BrowserSkillAdapter
    -> ``bsk`` CLI
    -> BrowserSkill daemon
    -> extension / Agent Window

The adapter deliberately does not hard-code browser element references such
as ``@eN``. References are obtained dynamically from ``observe`` and are
valid only for the current observed page state.

The planner decides WHAT to do. This adapter provides generic browser
primitives and semantic target resolution for deciding WHICH live element
reference to operate on.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Sequence
from urllib.parse import quote_plus, urlsplit, urlunsplit


_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_BSK_COMMAND = "bsk"
_DEFAULT_OBSERVE_MAX_DEPTH = 32
_DEFAULT_OBSERVE_MAX_TOKENS = 12000
_DEFAULT_READY_TIMEOUT_S = 12.0
_DEFAULT_POLL_MS = 200
_WHATSAPP_READY_TIMEOUT_S = 45.0
_WHATSAPP_SEND_TIMEOUT_S = 60.0

# WhatsApp Web's composer is a contenteditable textbox that does not reliably
# accept BrowserSkill's generic ``fill`` command. ``_whatsapp_type_text``
# instead drives it character-by-character through ``press``. This map
# translates characters that BrowserSkill's key-press command needs spelled
# out as named keys rather than the literal character.
_WHATSAPP_KEY_MAP = {
    " ": "Space",
    "\n": "Shift+Enter",
    "\r": "Shift+Enter",
}



class BrowserSkillError(RuntimeError):
    """Base error raised by the BrowserSkill adapter.

    BrowserSkill errors carry structured diagnostics in addition to their
    human-readable message.  Keeping these fields on the exception lets the
    skill/executor layer preserve the failure code without having to parse
    exception text.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "unknown",
        exit_code: int | None = None,
        data: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code
        self.data = data


class BrowserSkillProtocolError(BrowserSkillError):
    """Raised when BrowserSkill returns malformed or unexpected data."""


class BrowserSkillUnavailable(BrowserSkillError):
    """Raised when the BrowserSkill CLI cannot be started."""


class BrowserSkillResult(dict):
    """Dictionary result returned by BrowserSkill commands."""

    @property
    def ok(self) -> bool:
        value = self.get("ok")

        if isinstance(value, bool):
            return value

        status = self.get("status")

        if isinstance(status, str):
            return status.lower() in {
                "ok",
                "success",
                "succeeded",
                "pass",
                "passed",
            }

        return True


@dataclass(frozen=True)
class BrowserObservation:
    """Fresh semantic observation and the generation of its ref store.

    BrowserSkill refs are observation-scoped capabilities.  A generation is
    advanced after every operation that can invalidate the current ref store.
    Keeping the generation with the observation makes accidental reuse of a
    stale ``@eN`` ref detectable when callers use :class:`BrowserTarget`.
    """

    generation: int
    session: str
    tab_id: int | None
    url: str
    text: str
    elements: tuple["BrowserElement", ...]
    raw: Any


class BrowserTarget:
    """Resolved browser target.

    ``ref`` is a live BrowserSkill semantic reference such as ``@eN``.
    ``role`` and ``name`` are retained for diagnostics.
    """

    def __init__(
        self,
        ref: str,
        *,
        role: str = "",
        name: str = "",
        raw: Any = None,
        generation: int | None = None,
    ) -> None:
        self.ref = ref
        self.role = role
        self.name = name
        self.raw = raw
        self.generation = generation

    def __repr__(self) -> str:
        return (
            f"BrowserTarget("
            f"ref={self.ref!r}, "
            f"role={self.role!r}, "
            f"name={self.name!r})"
        )


class BrowserElement:
    """Normalized semantic browser element."""

    def __init__(
        self,
        ref: str,
        *,
        role: str = "",
        name: str = "",
        value: str = "",
        raw: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.ref = ref
        self.role = role
        self.name = name
        self.value = value
        self.raw = raw
        self.attributes = dict(attributes or {})

    def __repr__(self) -> str:
        return (
            f"BrowserElement("
            f"ref={self.ref!r}, "
            f"role={self.role!r}, "
            f"name={self.name!r}, "
            f"value={self.value!r})"
        )


class BrowserSkillCLI:
    """Small synchronous wrapper around the ``bsk`` executable."""

    def __init__(
        self,
        *,
        command: str | Iterable[str] = _DEFAULT_BSK_COMMAND,
        timeout: float = _DEFAULT_TIMEOUT_S,
        executable: str | None = None,
        working_directory: str | os.PathLike[str] | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        if executable is not None:
            if command != _DEFAULT_BSK_COMMAND:
                raise ValueError("pass either command or executable, not both")
            command = executable

        if isinstance(command, str):
            self.command = (command,)
        else:
            self.command = tuple(command)

        if not self.command:
            raise ValueError("BrowserSkill CLI command cannot be empty")

        self.timeout = float(timeout)

        self.working_directory = (
            str(working_directory)
            if working_directory is not None
            else None
        )

        self.environment = dict(environment or {})
        self._lock = RLock()

    def run(self, arguments: Iterable[str]) -> BrowserSkillResult:
        """Run ``bsk --json ...`` and parse its response."""

        args = [str(value) for value in arguments]

        command = [
            *self.command,
            "--json",
            *args,
        ]

        env = os.environ.copy()
        env.update(self.environment)

        with self._lock:
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.working_directory,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    check=False,
                )

            except FileNotFoundError as exc:
                raise BrowserSkillUnavailable(
                    "BrowserSkill CLI executable was not found: "
                    f"{self.command[0]!r}"
                ) from exc

            except subprocess.TimeoutExpired as exc:
                raise BrowserSkillError(
                    "BrowserSkill command timed out after "
                    f"{self.timeout:.1f}s: {' '.join(command)}"
                ) from exc

            except OSError as exc:
                raise BrowserSkillUnavailable(
                    f"Unable to start BrowserSkill CLI: {exc}"
                ) from exc

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()

        if completed.returncode != 0:
            message = (
                self._extract_error_message(stdout)
                or self._extract_error_message(stderr)
                or stderr
                or stdout
                or f"exit code {completed.returncode}"
            )

            lowered = message.casefold()
            if any(token in lowered for token in ("session not found", "unknown session", "session expired", "no such session")):
                code = "browser_session_expired"
            elif any(token in lowered for token in ("extension", "agent window", "not connected")):
                code = "browser_extension_not_connected"
            elif any(token in lowered for token in ("daemon", "service unavailable", "connection refused")):
                code = "browser_daemon_unavailable"
            else:
                code = "browser_action_failed"
            raise BrowserSkillError(
                "BrowserSkill command failed "
                f"(exit code {completed.returncode}): {message}",
                code=code,
                exit_code=completed.returncode,
            )

        if not stdout:
            result = BrowserSkillResult()
        else:
            result = self._parse_json_output(stdout)
        if stderr:
            result.setdefault("_stderr", stderr)
        return result

    @staticmethod
    def _parse_json_output(output: str) -> BrowserSkillResult:
        """Parse normal JSON and JSON-lines output."""

        try:
            value = json.loads(output)

        except json.JSONDecodeError:
            values: list[Any] = []

            for line in output.splitlines():
                line = line.strip()

                if not line:
                    continue

                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            if not values:
                raise BrowserSkillProtocolError(
                    "BrowserSkill returned non-JSON output: "
                    f"{output[:500]!r}"
                )

            value = values[-1]

        if isinstance(value, dict):
            return BrowserSkillResult(value)

        return BrowserSkillResult({"result": value})

    @staticmethod
    def _extract_error_message(value: str) -> str:
        """Extract a useful error string from JSON-ish output."""

        if not value:
            return ""

        try:
            parsed = json.loads(value)

        except json.JSONDecodeError:
            return value.strip()

        if isinstance(parsed, dict):
            for key in (
                "error",
                "message",
                "detail",
                "reason",
            ):
                item = parsed.get(key)

                if isinstance(item, str) and item.strip():
                    return item.strip()

                if isinstance(item, dict):
                    nested = item.get("message")

                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()

            return json.dumps(
                parsed,
                ensure_ascii=False,
            )

        return str(parsed)


class BrowserSkillAdapter:
    """Generic synchronous BrowserSkill adapter.

    This class intentionally exposes browser primitives instead of encoding
    site-specific workflows.

    Example
    -------

    ``resolve_target("Killshot")`` may return ``@eN`` today and a different
    reference after the page changes. The reference is always obtained from
    the current ``observe`` result.
    """

    _SESSION_COMMANDS = {
        "observe",
        "snapshot",
        "screenshot",
        "console",
        "network",
        "get-html",
        "navigate",
        "navigate-back",
        "navigate-forward",
        "reload",
        "click",
        "hover",
        "wheel",
        "scroll-to",
        "focus",
        "blur",
        "fill",
        "press",
        "select",
        "upload",
        "download",
        "evaluate",
        "wait-for-navigation",
        "request-help",
    }

    _NON_SESSION_COMMANDS = {
        "status",
        "wait-ms",
        "browsers",
        "doctor",
        "session",
    }

    def __init__(
        self,
        cli: Any | None = None,
        *,
        session: str | None = None,
        bsk_command: str | Iterable[str] = _DEFAULT_BSK_COMMAND,
        timeout: float = _DEFAULT_TIMEOUT_S,
        working_directory: str | os.PathLike[str] | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.session = session

        if cli is not None:
            self.cli = cli
        else:
            self.cli = BrowserSkillCLI(
                command=bsk_command,
                timeout=timeout,
                working_directory=working_directory,
                environment=environment,
            )

        self._lock = RLock()
        self.debug = False
        self._generation = 0
        self._last_observation: BrowserObservation | None = None

    def new_task_session(self) -> "BrowserSkillAdapter":
        """Create an independent BrowserSkill resource for one task.

        The new adapter shares only the CLI transport configuration. It does
        not share a BrowserSkill session, active tab, observation cache, or
        semantic-ref generation with this adapter. BrowserSkill assigns the
        actual session id when the child first executes a command.
        """
        return BrowserSkillAdapter(
            cli=self.cli,
        )

    @property
    def session_id(self) -> str | None:
        """Compatibility alias for the active session id."""
        return self.session

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self.session = value

    def close_session(self) -> None:
        """Best-effort session teardown used by the builtin registry."""
        if self.session:
            try:
                self.session_stop()
            finally:
                self._invalidate_refs()

    def _invalidate_refs(self) -> None:
        self._generation += 1
        self._last_observation = None

    # ------------------------------------------------------------------
    # Low-level execution
    # ------------------------------------------------------------------

    def _run(
        self,
        arguments: Iterable[str],
    ) -> BrowserSkillResult:
        result = self.cli.run(arguments)
        if isinstance(result, BrowserSkillResult):
            return result
        if isinstance(result, dict):
            return BrowserSkillResult(result)
        return BrowserSkillResult({"result": result})

    def ensure_ready(self) -> BrowserObservation:
        """Establish/reuse BrowserSkill and prove it can observe the browser.

        Process/window existence is deliberately not treated as browser
        readiness. A successful fresh BrowserSkill observation is the readiness
        proof used by semantic browser actions.
        """
        if not self.session:
            try:
                self.session_start()
            except BrowserSkillError:
                raise
            except Exception as exc:
                raise BrowserSkillError(
                    f"BrowserSkill session start failed: {type(exc).__name__}: {exc}",
                    code="browser_session_start_failed",
                ) from exc
        try:
            self.observe()
        except BrowserSkillError as exc:
            # A dead session may be recreated once, but only before an action is
            # attempted. Never replay an unknown browser mutation here.
            if exc.code in {"browser_session_expired", "browser_session_unhealthy", "session_not_found", "session_expired"} or not self.session:
                self.session = None
                self._invalidate_refs()
                self.session_start()
                self.observe()
            else:
                raise
        except Exception as exc:
            raise BrowserSkillError(
                f"BrowserSkill observation failed: {type(exc).__name__}: {exc}",
                code="browser_observation_failed",
            ) from exc
        if self._last_observation is None:
            raise BrowserSkillError(
                "BrowserSkill did not produce a usable browser observation",
                code="browser_observation_failed",
            )
        return self._last_observation

    def _require_session(self) -> str:
        if not self.session:
            self.session_start()
        if not self.session:
            raise BrowserSkillError("BrowserSkill session could not be established", code="browser_session_start_failed")
        return self.session

    def _session_command(
        self,
        command: str,
        *arguments: str,
    ) -> BrowserSkillResult:
        if command not in self._SESSION_COMMANDS:
            raise BrowserSkillError(
                f"Command {command!r} is not configured "
                "as session-scoped"
            )

        session = self._require_session()

        result = self._run(
            [
                command,
                "--session",
                session,
                *arguments,
            ]
        )
        if command not in {"observe", "snapshot", "screenshot", "console", "network", "get-html", "wait-for-navigation", "request-help"}:
            self._invalidate_refs()
        return result

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _browser_profile_score(item: dict[str, Any]) -> tuple[int, int, int]:
        """Rank connected Chromium profiles for the normal user browser.

        BrowserSkill can see more than one connected Chrome profile.  DEIMOS
        should prefer the user's normal ``Default`` profile, while still
        falling back safely when an older BrowserSkill build does not expose
        profile metadata.
        """
        values = []
        for key in ("profile", "profile_name", "profileName", "name", "label", "title"):
            value = item.get(key)
            if isinstance(value, str):
                values.append(value.strip().casefold())
        text = " ".join(values)
        browser_text = " ".join(
            str(item.get(key) or "").strip().casefold()
            for key in ("browser", "browser_name", "browserName", "type", "name", "label")
        )
        is_chrome = int("chrome" in browser_text and "edge" not in browser_text)
        is_default = int(any(v == "default" or "default" in v for v in values))
        looks_user_profile = int(any(
            marker in text for marker in ("default", "personal", "main")
        ))
        return (is_default, is_chrome, looks_user_profile)

    @classmethod
    def _select_default_chrome_browser(cls, result: BrowserSkillResult) -> str | None:
        """Return the connected BrowserSkill browser id for Chrome Default."""
        candidates: list[dict[str, Any]] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                if any(key in value for key in ("id", "browser_id", "browserId")):
                    candidates.append(value)
                for child in value.values():
                    collect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child)

        collect(result)
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in candidates:
            browser_id = next((
                item.get(key) for key in ("id", "browser_id", "browserId")
                if isinstance(item.get(key), str) and item.get(key).strip()
            ), None)
            if not browser_id or browser_id in seen:
                continue
            seen.add(browser_id)
            unique.append(item)

        chrome = [
            item for item in unique
            if "chrome" in " ".join(
                str(item.get(key) or "").casefold()
                for key in ("browser", "browser_name", "browserName", "type", "name", "label", "profile", "profile_name")
            )
            and "edge" not in " ".join(
                str(item.get(key) or "").casefold()
                for key in ("browser", "browser_name", "browserName", "type")
            )
        ]
        if not chrome:
            return None

        selected = max(chrome, key=cls._browser_profile_score)
        return next((
            selected.get(key) for key in ("id", "browser_id", "browserId")
            if isinstance(selected.get(key), str) and selected.get(key).strip()
        ), None)

    @staticmethod
    def _configured_chrome_profile() -> dict[str, str]:
        """Return the existing Chrome deployment configuration from env."""
        return {
            key: value
            for key, value in {
                "user_data": os.getenv("DEIMOS_CHROME_USER_DATA", "").strip(),
                "profile": os.getenv("DEIMOS_CHROME_PROFILE", "").strip(),
                "debug_port": os.getenv("DEBUG_PORT", "").strip(),
            }.items()
            if value
        }

    @staticmethod
    def _browser_record_matches_config(item: dict[str, Any], config: dict[str, str]) -> bool:
        """Match configured Chrome identity using metadata exposed by BrowserSkill."""
        raw = json.dumps(item, ensure_ascii=False)
        folded = raw.casefold()
        if "chrome" not in folded or "edge" in folded:
            return False
        profile = config.get("profile", "").casefold()
        user_data = config.get("user_data", "")
        if profile:
            values = [
                str(item.get(key) or "").strip().casefold()
                for key in ("profile", "profile_name", "profileName", "profile_directory", "profileDirectory")
            ]
            if profile not in values:
                return False
        if user_data:
            normalized = os.path.normcase(os.path.normpath(user_data)).casefold()
            metadata_values = [
                str(item.get(key) or "").strip()
                for key in ("user_data", "user_data_dir", "userDataDir", "user-data-dir", "userData", "userDataPath")
            ]
            if not any(normalized == os.path.normcase(os.path.normpath(v)).casefold() for v in metadata_values if v):
                # Some BrowserSkill versions only expose the path nested in a
                # browser/session object. Search serialized metadata rather than
                # constructing or probing profile paths ourselves.
                if normalized.replace("\\", "/") not in folded.replace("\\", "/"):
                    return False
        return True

    @classmethod
    def _select_configured_chrome_browser(cls, result: BrowserSkillResult, config: dict[str, str]) -> str | None:
        candidates: list[dict[str, Any]] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                if any(key in value for key in ("id", "browser_id", "browserId")):
                    candidates.append(value)
                for child in value.values():
                    collect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child)

        collect(result)
        seen: set[str] = set()
        unique: list[tuple[str, dict[str, Any]]] = []
        explicit_chrome: list[tuple[str, dict[str, Any]]] = []
        for item in candidates:
            browser_id = next((item.get(key) for key in ("id", "browser_id", "browserId") if str(item.get(key) or "").strip()), None)
            if not browser_id:
                continue
            browser_id = str(browser_id).strip()
            if browser_id in seen:
                continue
            seen.add(browser_id)
            unique.append((browser_id, item))
            raw = json.dumps(item, ensure_ascii=False).casefold()
            if "chrome" in raw and "edge" not in raw:
                explicit_chrome.append((browser_id, item))
            if cls._browser_record_matches_config(item, config):
                return browser_id

        # Some BrowserSkill builds expose a generic id such as ``discovered``
        # and omit browser/profile metadata entirely. If exactly one browser is
        # exposed, the local Chrome process is the only safe external identity
        # source available to DEIMOS. Accept that browser only when the process
        # matches the configured Chrome profile. Multiple candidates remain
        # fail-closed because choosing one would be an arbitrary profile switch.
        candidates_for_process_match = explicit_chrome if explicit_chrome else unique
        if len(candidates_for_process_match) == 1:
            browser_id, item = candidates_for_process_match[0]
            raw = json.dumps(item, ensure_ascii=False).casefold()
            if "edge" not in raw:
                try:
                    from ... import os_tools
                    if os_tools._chrome_process_matches_configuration(config):
                        return browser_id
                except Exception:
                    pass
        return None

    def session_start(self) -> BrowserSkillResult:
        """Start a BrowserSkill session on the user's normal Chrome profile.

        BrowserSkill, rather than ``subprocess.Popen(chrome.exe)``, owns the
        browser lifecycle.  This preserves the user's real Chrome cookies,
        logins and tabs while keeping DEIMOS inside BrowserSkill's Agent Window
        boundary.
        """
        arguments = ["session", "start"]
        config = self._configured_chrome_profile()
        try:
            # Preserve the existing BrowserSkill default path when no explicit
            # Chrome configuration exists. Configured deployments require an
            # explicit metadata match so they can never silently attach to an
            # unrelated profile.
            if config:
                browsers = self.browsers()
                browser_id = self._select_configured_chrome_browser(browsers, config)
                if browser_id is None:
                    raise BrowserSkillError(
                        "configured Chrome profile/session is unavailable to BrowserSkill",
                        code="configured_profile_unavailable",
                        data={"configured_profile": config},
                    )
            else:
                try:
                    browsers = self.browsers()
                    browser_id = self._select_default_chrome_browser(browsers)
                except Exception:
                    # Older BrowserSkill installations may not expose browser
                    # discovery; retain the existing default session behavior.
                    browser_id = None
        except BrowserSkillError:
            raise
        except Exception as exc:
            if config:
                raise BrowserSkillError(
                    "could not inspect BrowserSkill browser sessions for the configured Chrome profile",
                    code="configured_profile_unavailable",
                    data={"configured_profile": config, "error": str(exc)},
                ) from exc
            browser_id = None

        if browser_id:
            arguments.extend(["--browser", browser_id])
        arguments.append("--no-focus")

        result = self._run(arguments)
        if config and not self._extract_session_id(result):
            raise BrowserSkillError(
                "BrowserSkill did not establish a session for the configured Chrome profile",
                code="browser_session_not_ready",
                data={"configured_profile": config},
            )

        session_id = self._extract_session_id(result)

        if session_id:
            self.session = session_id

        return result

    def session_stop(
        self,
        session_id: str | None = None,
    ) -> BrowserSkillResult:
        """Stop a BrowserSkill session."""

        target = session_id or self.session

        if not target:
            raise BrowserSkillError(
                "No BrowserSkill session ID was provided"
            )

        result = self._run(
            [
                "session",
                "stop",
                target,
            ]
        )

        if target == self.session:
            self.session = None
            self._invalidate_refs()

        return result

    def session_list(self) -> BrowserSkillResult:
        """List BrowserSkill sessions."""

        return self._run(
            [
                "session",
                "list",
            ]
        )

    def _extract_session_id(
        self,
        result: BrowserSkillResult,
    ) -> str | None:
        value = self._find_first_value(
            result,
            (
                "session_id",
                "sessionId",
            ),
        )

        if isinstance(value, str) and value.strip():
            return value.strip()

        return None

    # ------------------------------------------------------------------
    # Global commands
    # ------------------------------------------------------------------

    def status(self) -> BrowserSkillResult:
        """Return global BrowserSkill status.

        The installed BrowserSkill CLI exposes ``status`` as a
        non-session-scoped command.
        """

        return self._run(["status"])

    def wait_ms(
        self,
        milliseconds: int,
    ) -> BrowserSkillResult:
        """Wait for a specified number of milliseconds."""

        milliseconds = int(milliseconds)

        if milliseconds < 0:
            raise ValueError(
                "milliseconds cannot be negative"
            )

        return self._run(
            [
                "wait-ms",
                str(milliseconds),
            ]
        )

    def browsers(self) -> BrowserSkillResult:
        """List available browser instances."""

        return self._run(["browsers"])

    def doctor(self) -> BrowserSkillResult:
        """Run BrowserSkill diagnostics."""

        return self._run(["doctor"])

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate(
        self,
        url: str,
    ) -> BrowserSkillResult:
        """Navigate the Agent Window to ``url``."""

        if not isinstance(url, str) or not url.strip():
            raise ValueError(
                "url must be a non-empty string"
            )

        return self._session_command(
            "navigate",
            url.strip(),
            "--wait-until",
            "domcontentloaded",
            "--timeout",
            str(int(getattr(self.cli, "timeout", _DEFAULT_TIMEOUT_S) * 1000)),
        )

    def navigate_back(self) -> BrowserSkillResult:
        return self._session_command(
            "navigate-back"
        )

    def navigate_forward(self) -> BrowserSkillResult:
        return self._session_command(
            "navigate-forward"
        )

    def reload(self) -> BrowserSkillResult:
        return self._session_command("reload")

    def open_url(
        self,
        url: str,
    ) -> BrowserSkillResult:
        """Compatibility alias for navigation."""

        return self.navigate(url)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self) -> BrowserSkillResult:
        """Get a fresh semantic observation and cache its ref generation."""
        result = self._session_command(
            "observe",
            "--max-depth",
            str(_DEFAULT_OBSERVE_MAX_DEPTH),
            "--max-tokens",
            str(_DEFAULT_OBSERVE_MAX_TOKENS),
        )
        # Every observe allocates a fresh BrowserSkill ref-store.  Advance the
        # local generation before exposing any of its refs.
        self._generation += 1
        elements = tuple(self._extract_elements(result))
        # BrowserSkill can expose the semantic tree either as structured
        # data or as a serialized/escaped semantic string nested inside the
        # JSON response.  The old implementation kept only the first
        # ``text`` field, which could be unrelated to the visible page.
        # Preserve that field when present, but also expose the complete
        # semantic tree reconstructed from the elements we just extracted.
        raw_text = self._find_first_value(result, ("text",))
        text_parts = [str(raw_text)] if isinstance(raw_text, str) and raw_text else []
        for element in elements:
            line = f"{element.ref} {element.role}"
            if element.name:
                line += f' "{element.name}"'
            if element.value:
                line += f' ="{element.value}"'
            text_parts.append(line)
        text = "\n".join(text_parts)
        tab_value = self._find_first_value(result, ("tab_id", "tabId"))
        tab_id = int(tab_value) if isinstance(tab_value, (int, str)) and str(tab_value).isdigit() else None
        self._last_observation = BrowserObservation(
            generation=self._generation,
            session=self._require_session(),
            tab_id=tab_id,
            url=self._find_first_string(result, ("url", "final_url", "finalUrl")),
            text=text,
            elements=elements,
            raw=result,
        )
        return result

    def snapshot(self) -> BrowserSkillResult:
        """Get browser accessibility/page snapshot."""

        return self._session_command("snapshot")

    def screenshot(self) -> BrowserSkillResult:
        """Capture a screenshot."""

        return self._session_command("screenshot")

    def get_html(self) -> BrowserSkillResult:
        """Retrieve page HTML."""

        return self._session_command("get-html")

    def console(self) -> BrowserSkillResult:
        """Retrieve browser console information."""

        return self._session_command("console")

    def network(self) -> BrowserSkillResult:
        """Retrieve browser network information."""

        return self._session_command("network")

    # ------------------------------------------------------------------
    # Independent browser state
    # ------------------------------------------------------------------

    def current_url(self) -> str:
        """Return the active session tab URL, falling back to observation."""
        try:
            tabs = self.list_tabs("agent")
            for tab in tabs:
                if tab.get("active") or tab.get("is_active") or tab.get("selected"):
                    url = tab.get("url")
                    if isinstance(url, str) and url.strip():
                        return url.strip()
            if tabs:
                url = tabs[0].get("url")
                if isinstance(url, str) and url.strip():
                    return url.strip()
        except Exception:
            pass
        try:
            self.observe()
            if self._last_observation and self._last_observation.url:
                return self._last_observation.url
        except Exception:
            pass
        return ""

    def page_title(self) -> str:
        """Return the active session tab title, falling back to observation."""
        try:
            tabs = self.list_tabs("agent")
            for tab in tabs:
                if tab.get("active") or tab.get("is_active") or tab.get("selected"):
                    title = tab.get("title")
                    if isinstance(title, str) and title.strip():
                        return title.strip()
            if tabs:
                title = tabs[0].get("title")
                if isinstance(title, str) and title.strip():
                    return title.strip()
        except Exception:
            pass
        try:
            self.observe()
            if self._last_observation:
                value = self._find_first_string(self._last_observation.raw, ("title", "page_title", "pageTitle"))
                if value:
                    return value
        except Exception:
            pass
        return ""

    def page_text(self) -> str:
        """Extract visible text from the current observation."""

        observation = self.observe()

        values: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, str):
                values.append(value)
                return

            if isinstance(value, dict):
                for key, item in value.items():
                    if key.lower() in {
                        "text",
                        "name",
                        "label",
                        "description",
                        "value",
                        "title",
                    }:
                        if isinstance(item, str):
                            values.append(item)

                    walk(item)

                return

            if isinstance(value, list):
                for item in value:
                    walk(item)

        walk(observation)

        seen: set[str] = set()
        output: list[str] = []

        for value in values:
            value = value.strip()

            if not value or value in seen:
                continue

            seen.add(value)
            output.append(value)

        return "\n".join(output)

    # ------------------------------------------------------------------
    # Generic semantic resolution
    # ------------------------------------------------------------------

    def elements(self) -> list[BrowserElement]:
        """Return normalized semantic elements from ``observe``."""

        observation = self.observe()

        return self._extract_elements(observation)

    def resolve_target(
        self,
        query: str,
        *,
        preferred_roles: Iterable[str] = (),
        role: str | None = None,
        observation: BrowserSkillResult | BrowserObservation | None = None,
        min_score: int = 0,
        reject_ambiguous: bool = False,
    ) -> BrowserTarget:
        """Resolve a semantic request against one fresh observation.

        Matching is deterministic and role-aware.  Ambiguous close matches
        can be rejected instead of clicking the first textual substring.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        query = query.strip()
        if observation is None and self._last_observation is None:
            self.ensure_ready()
        if isinstance(observation, BrowserObservation):
            obs = observation
        else:
            if observation is None:
                self.observe()
                obs = self._last_observation
            else:
                elements = tuple(self._extract_elements(observation))
                text = str(self._find_first_value(observation, ("text",)) or "")
                obs = BrowserObservation(self._generation, self._require_session(), None, "", text, elements, observation)
        if obs is None or not obs.elements:
            raise BrowserSkillError(f"No semantic browser elements were found while resolving {query!r}")
        preferred = {str(r).strip().casefold() for r in preferred_roles if str(r).strip()}
        requested_role = role.casefold().strip() if role else None
        normalized = self._normalize_text(query)
        candidates: list[tuple[int, BrowserElement]] = []
        for element in obs.elements:
            if not element.ref:
                continue
            score = self._target_score(normalized, name=self._normalize_text(element.name), value=self._normalize_text(element.value), role=element.role.casefold().strip(), preferred_roles=preferred)
            if requested_role and element.role.casefold().strip() == requested_role:
                score += 80
            elif requested_role and element.role:
                score -= 40
            if score >= min_score and score > 0:
                candidates.append((score, element))
        if not candidates:
            raise BrowserSkillError(f"Could not resolve browser target {query!r}; candidates=0")
        candidates.sort(key=lambda item: (-item[0], item[1].ref))
        best_score, best = candidates[0]
        if reject_ambiguous and len(candidates) > 1:
            second_score = candidates[1][0]
            if best_score - second_score < 40:
                detail = ", ".join(f"{e.ref}:{e.name!r}({s})" for s, e in candidates[:5])
                raise BrowserSkillError(f"Ambiguous browser target {query!r}: {detail}")
        return BrowserTarget(best.ref, role=best.role, name=best.name, raw=best.raw, generation=obs.generation)

    @staticmethod
    def _target_score(query: str, *, name: str, value: str, role: str, preferred_roles: set[str]) -> int:
        if not query:
            return 0
        score = 0
        if role in preferred_roles:
            score += 120
        if name == query:
            score += 1000
        elif value == query:
            score += 900
        elif name.startswith(query):
            score += 700
        elif query in name:
            score += 520
        elif value.startswith(query):
            score += 450
        elif query in value:
            score += 320
        qwords = set(query.split())
        if qwords:
            nwords = set(name.split())
            vwords = set(value.split())
            score += 70 * len(qwords & nwords)
            score += 25 * len(qwords & vwords)
            if qwords and qwords <= nwords:
                score += 100
        return score

    # ------------------------------------------------------------------
    # Browser interaction primitives
    # ------------------------------------------------------------------

    def type_text(self, target: str | BrowserTarget, text: str) -> BrowserSkillResult:
        return self.fill(target, text)

    def press_key(self, key: str, target: str | BrowserTarget | None = None) -> BrowserSkillResult:
        if target is None:
            return self._session_command("press", str(key))
        return self.press(target, str(key))

    def scroll(self, amount: int) -> BrowserSkillResult:
        """Scroll the active page using BrowserSkill's supported key primitive.

        Some installed BrowserSkill CLI builds do not expose the newer ``wheel``
        subcommand even though the browser adapter historically called it.  The
        portable BrowserSkill contract already exposes global ``press`` and
        Chromium treats PageDown/PageUp as real viewport scrolling.  Keep the
        capability at this adapter boundary rather than guessing a CLI command
        or bypassing BrowserSkill.
        """
        delta = int(amount)
        if delta == 0:
            return BrowserSkillResult({"ok": True, "amount": 0, "key": None})

        # Newer BrowserSkill builds expose native wheel input.  Some installed
        # Windows builds in the field do not; they return an explicit
        # ``unrecognized subcommand 'wheel'`` error.  Try the native primitive
        # first, then use the already-supported semantic keyboard primitive as a
        # compatibility path.  Both stay inside BrowserSkill and both cause a
        # real browser mutation; there is no fake success path.
        try:
            return self.wheel(delta)
        except BrowserSkillError as exc:
            message = str(exc).casefold()
            if "unrecognized subcommand 'wheel'" not in message:
                raise

        key = "PageDown" if delta > 0 else "PageUp"
        result = self._session_command("press", key)
        if isinstance(result, BrowserSkillResult):
            result.setdefault("scroll_amount", delta)
            result.setdefault("scroll_key", key)
            result.setdefault("scroll_fallback", "keyboard")
        return result

    def upload_file(self, target: str | BrowserTarget, file_path: str, *, mode: str | None = None) -> BrowserSkillResult:
        return self.upload(target, file_path, mode=mode)

    def close_tab(self, tab_id: int | None = None) -> BrowserSkillResult:
        args = ["tab", "close"]
        if tab_id is None:
            raise ValueError("tab_id is required for close_tab")
        return self._session_command("tab-close", str(tab_id)) if False else self._run(["tab", "close", str(int(tab_id)), "--session", self._require_session()])

    def create_tab(self, url: str | None = None) -> BrowserSkillResult:
        args = ["tab", "create", "--session", self._require_session()]
        if url:
            args += ["--url", str(url)]
        result = self._run(args)
        self._invalidate_refs()
        return result

    def select_tab(self, tab_id: int) -> BrowserSkillResult:
        result = self._run(["tab", "select", str(int(tab_id)), "--session", self._require_session()])
        self._invalidate_refs()
        return result

    def borrow_tab(self, tab_id: int) -> BrowserSkillResult:
        result = self._run(["tab", "borrow", str(int(tab_id)), "--session", self._require_session()])
        self._invalidate_refs()
        return result

    def return_tab(self, tab_id: int) -> BrowserSkillResult:
        result = self._run(["tab", "return", str(int(tab_id)), "--session", self._require_session()])
        self._invalidate_refs()
        return result

    def go_back(self) -> BrowserSkillResult:
        return self.navigate_back()

    def go_forward(self) -> BrowserSkillResult:
        return self.navigate_forward()

    def refresh(self) -> BrowserSkillResult:
        return self.reload()

    def page_contains_text(self, text: str) -> bool:
        return self._normalize_text(text) in self._normalize_text(self.page_text())

    def click(
        self,
        target: str | BrowserTarget,
    ) -> BrowserSkillResult:
        """Click a live BrowserSkill semantic reference."""

        ref = self._coerce_ref(target)

        return self._session_command(
            "click",
            "--ref",
            ref,
        )

    def hover(
        self,
        target: str | BrowserTarget,
    ) -> BrowserSkillResult:
        """Hover a live BrowserSkill semantic reference."""

        ref = self._coerce_ref(target)

        return self._session_command(
            "hover",
            "--ref",
            ref,
        )

    def focus(
        self,
        target: str | BrowserTarget,
    ) -> BrowserSkillResult:
        ref = self._coerce_ref(target)

        return self._session_command(
            "focus",
            "--ref",
            ref,
        )

    def blur(
        self,
        target: str | BrowserTarget,
    ) -> BrowserSkillResult:
        ref = self._coerce_ref(target)

        return self._session_command(
            "blur",
            "--ref",
            ref,
        )

    def fill(
        self,
        target: str | BrowserTarget,
        text: str,
    ) -> BrowserSkillResult:
        """Fill a browser input."""

        ref = self._coerce_ref(target)

        return self._session_command(
            "fill",
            "--ref",
            ref,
            "--value",
            str(text),
        )

    def press(
        self,
        target: str | BrowserTarget,
        key: str,
    ) -> BrowserSkillResult:
        """Press a key on a browser target."""

        ref = self._coerce_ref(target)

        return self._session_command(
            "press",
            str(key),
            "--ref",
            ref,
        )

    def select(
        self,
        target: str | BrowserTarget,
        value: str,
    ) -> BrowserSkillResult:
        """Select an option from a browser control."""

        ref = self._coerce_ref(target)

        return self._session_command(
            "select",
            "--ref",
            ref,
            "--value",
            str(value),
        )

    def wheel(
        self,
        delta_y: int,
    ) -> BrowserSkillResult:
        """Scroll using the browser wheel."""

        return self._session_command(
            "wheel",
            "--delta-y",
            str(int(delta_y)),
        )

    def scroll_to(
        self,
        target: str | BrowserTarget,
    ) -> BrowserSkillResult:
        """Scroll a target into view."""

        ref = self._coerce_ref(target)

        return self._session_command(
            "scroll-to",
            "--ref",
            ref,
        )

    def upload(
        self,
        target: str | BrowserTarget,
        path: str | os.PathLike[str],
        *,
        mode: str | None = None,
    ) -> BrowserSkillResult:
        """Upload a file through BrowserSkill."""

        ref = self._coerce_ref(target)
        file_path = str(Path(path))

        args = ["upload", "--ref", ref, "--file", file_path]
        if mode:
            args += ["--mode", str(mode)]
        return self._session_command(*args)

    def download(
        self,
        target: str | BrowserTarget,
        output_path: str | os.PathLike[str] | None = None,
        *,
        overwrite: bool = False,
    ) -> BrowserSkillResult:
        ref = self._coerce_ref(target)
        if output_path is None:
            raise ValueError("output_path is required for download")
        args = ["download", "--ref", ref, "--out", str(Path(output_path))]
        if overwrite:
            args.append("--overwrite")
        return self._session_command(*args)

    # ------------------------------------------------------------------
    # Waiting
    # ------------------------------------------------------------------

    def wait_for_navigation(
        self,
        timeout_ms: int | None = None,
    ) -> BrowserSkillResult:
        """Wait for navigation to complete."""

        arguments: list[str] = []

        if timeout_ms is not None:
            arguments.extend(["--timeout", f"{int(timeout_ms)}ms"])

        return self._session_command(
            "wait-for-navigation",
            *arguments,
        )

    # ------------------------------------------------------------------
    # Tabs / windows
    # ------------------------------------------------------------------

    def list_tabs(
        self,
        scope: str = "all",
    ) -> list[dict[str, Any]]:
        """Return normalized tab information."""

        result = self._run_tabs_command(scope)

        candidates = self._find_first_list(
            result,
            (
                "tabs",
                "windows",
                "pages",
                "items",
                "results",
            ),
        )

        if candidates is None:
            return []

        output: list[dict[str, Any]] = []

        for item in candidates:
            if isinstance(item, dict):
                output.append(item)

        return output

    def _run_tabs_command(
        self,
        scope: str,
    ) -> BrowserSkillResult:
        """Run the current BrowserSkill ``tab list`` command."""
        if scope not in {"all", "agent", "user"}:
            raise ValueError("scope must be one of: all, agent, user")
        return self._run([
            "tab",
            "list",
            "--session",
            self._require_session(),
            "--scope",
            scope,
        ])

    # ------------------------------------------------------------------
    # Advanced / human-in-the-loop operations
    # ------------------------------------------------------------------

    def evaluate(
        self,
        expression: str,
    ) -> BrowserSkillResult:
        """Evaluate browser JavaScript.

        This is an escape hatch. Normal DEIMOS browser actions should use
        semantic observation and primitives first.
        """

        if (
            not isinstance(expression, str)
            or not expression.strip()
        ):
            raise ValueError(
                "expression must be non-empty"
            )

        return self._session_command(
            "evaluate",
            expression,
        )

    def request_help(
        self,
        reason: str,
    ) -> BrowserSkillResult:
        """Request human assistance."""

        if (
            not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError(
                "reason must be non-empty"
            )

        return self._session_command(
            "request-help",
            reason,
        )

    def record(
        self,
        enabled: bool = True,
    ) -> BrowserSkillResult:
        """Start/stop BrowserSkill recording using its dedicated lifecycle.

        ``record start`` owns its own session and blocks until the human ends
        the recording, so it is intentionally not treated as a normal
        session-scoped primitive.
        """
        result = self._run(["record", "start" if enabled else "stop"])
        if not enabled:
            self._invalidate_refs()
        return result

    # ------------------------------------------------------------------
    # Readiness / observation synchronization
    # ------------------------------------------------------------------

    def wait_until_observation_contains(
        self,
        query: str,
        *,
        timeout_s: float = _DEFAULT_READY_TIMEOUT_S,
        poll_ms: int = _DEFAULT_POLL_MS,
        preferred_roles: Iterable[str] = (),
        role: str | None = None,
        min_score: int = 0,
        dismiss_blocking_layer: bool = False,
    ) -> BrowserTarget:
        """Wait for a fresh semantic observation to resolve ``query``.

        Dynamic pages may report ``load`` before their actionable semantic
        elements exist.  Every poll therefore observes again and resolves the
        target against that observation, so returned refs always belong to the
        latest ref generation.  Optional blocking-layer dismissal is kept
        generic and uses only the semantic observation signal plus Escape.
        """
        import time

        deadline = time.monotonic() + max(0.1, float(timeout_s))
        last_count = 0
        dismiss_attempts = 0

        while True:
            self.observe()
            observation = self._last_observation
            elements = [] if observation is None else observation.elements
            last_count = len(elements)

            if (
                dismiss_blocking_layer
                and self._observation_has_blocking_layer(
                    observation.text if observation is not None else ""
                )
                and dismiss_attempts < 3
                and time.monotonic() < deadline
            ):
                dismiss_attempts += 1
                try:
                    self.press_key("Escape")
                finally:
                    self._invalidate_refs()
                self.wait_ms(min(max(1, int(poll_ms)), max(
                    1, int((deadline - time.monotonic()) * 1000)
                )))
                continue

            try:
                return self.resolve_target(
                    query,
                    preferred_roles=preferred_roles,
                    role=role,
                    observation=observation,
                    min_score=min_score,
                )
            except BrowserSkillError as exc:
                if time.monotonic() >= deadline:
                    raise BrowserSkillError(
                        f"Timed out after {timeout_s:.1f}s waiting for semantic target "
                        f"{query!r}; last_observation_elements={last_count}; "
                        f"url={self.current_url()!r}",
                        code="semantic_target_not_ready",
                        data={"query": query, "elements": last_count},
                    ) from exc

            remaining = max(1, int((deadline - time.monotonic()) * 1000))
            self.wait_ms(min(max(1, int(poll_ms)), remaining))

    def wait_for_url(
        self,
        expected: str,
        *,
        timeout_s: float = _DEFAULT_READY_TIMEOUT_S,
        poll_ms: int = _DEFAULT_POLL_MS,
    ) -> str:
        expected_c = canonical_url(expected)
        deadline = __import__("time").monotonic() + max(0.1, float(timeout_s))
        while True:
            actual = canonical_url(self.current_url())
            if actual == expected_c or expected_c in actual:
                return actual
            if __import__("time").monotonic() >= deadline:
                raise BrowserSkillError(f"Timed out waiting for URL {expected!r}; observed={actual!r}")
            self.wait_ms(min(int(poll_ms), max(1, int((deadline-__import__('time').monotonic())*1000))))

    # ------------------------------------------------------------------
    # Higher-level generic helpers
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        engine: str = "google",
    ) -> BrowserSkillResult:
        """Perform a generic web search by navigation."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "query must be non-empty"
            )

        encoded = quote_plus(query.strip())
        engine = engine.lower().strip()

        if engine == "google":
            url = (
                "https://www.google.com/search"
                f"?q={encoded}"
            )

        elif engine == "bing":
            url = (
                "https://www.bing.com/search"
                f"?q={encoded}"
            )

        elif engine == "duckduckgo":
            url = (
                "https://duckduckgo.com/"
                f"?q={encoded}"
            )

        else:
            raise ValueError(
                f"Unsupported search engine: {engine!r}"
            )

        return self.navigate(url)


    # ------------------------------------------------------------------
    # WhatsApp Web semantic workflow
    # ------------------------------------------------------------------

    @staticmethod
    def _whatsapp_normalize(value: str) -> str:
        return " ".join(str(value or "").casefold().split())

    @classmethod
    def _whatsapp_has_any(cls, text: str, phrases: Iterable[str]) -> bool:
        normalized = cls._whatsapp_normalize(text)
        return any(cls._whatsapp_normalize(p) in normalized for p in phrases)

    _WHATSAPP_SEARCH_NAMES = {
        "search or start a new chat",
        "search or start new chat",
        "search",
    }

    @classmethod
    def _whatsapp_is_search_element(cls, e: BrowserElement) -> bool:
        """Identify the WhatsApp search textbox by semantic signal.

        WhatsApp normally exposes the search control with a fixed
        placeholder name such as "Search or start a new chat". Once text has
        been typed into it, some WhatsApp Web builds instead expose the
        control's *current query* as its accessible name (e.g. a search for
        "Papa" can surface as ``textbox "Papa"``), alongside a bracketed
        ``[ctx: ic-search]`` annotation that ``_parse_observation_text``
        captures into ``attributes["annotations"]``. Treat that annotation as
        an equally valid signal so the control is still recognized as the
        search box in that state, rather than only matching a literal
        placeholder string.
        """
        if e.role.casefold() not in {"textbox", "combobox", "searchbox"}:
            return False

        if cls._whatsapp_normalize(e.name) in cls._WHATSAPP_SEARCH_NAMES:
            return True

        annotations = (e.attributes or {}).get("annotations") or []
        return any(
            "search" in cls._whatsapp_normalize(str(a))
            for a in annotations
        )

    def _whatsapp_observation(self) -> BrowserObservation:
        self.observe()
        if self._last_observation is None:
            raise BrowserSkillError(
                "WhatsApp semantic observation was unavailable",
                code="whatsapp_state_unknown",
            )
        return self._last_observation

    def whatsapp_state(self) -> str:
        """Classify the current WhatsApp Web state from a fresh observation.

        Returns ``READY``, ``LOGIN_REQUIRED``, ``LOADING`` or ``UNKNOWN``.
        This is intentionally semantic: no WhatsApp DOM selectors are used.
        """
        try:
            obs = self._whatsapp_observation()
        except Exception as exc:
            if isinstance(exc, BrowserSkillError):
                raise
            raise BrowserSkillError(
                f"could not observe WhatsApp state: {exc}",
                code="whatsapp_state_unknown",
            ) from exc

        url = (obs.url or self.current_url()).casefold()
        text = obs.text
        elements = self._whatsapp_semantic_elements(obs)
        semantic = "\n".join(
            f"{e.role} {e.name} {e.value}" for e in elements
        )
        haystack = f"{text}\n{semantic}"

        if "web.whatsapp.com" not in url:
            return "UNKNOWN"

        # An actionable WhatsApp semantic control is the readiness signal.
        # The startup/download banner may remain in the observation after the
        # usable UI is already live, so it must not override these controls.
        ready_names = {
            "search or start a new chat",
            "search or start new chat",
            "search",
            "voice call",
            "video call",
            "voice call button",
            "video call button",
        }
        if any(
            (
                self._whatsapp_normalize(e.name) in ready_names
                or self._whatsapp_is_search_element(e)
            )
            and e.role.casefold() in {
                "textbox", "combobox", "searchbox", "button", "link", "menuitem",
            }
            for e in elements
        ):
            return "READY"

        if self._whatsapp_has_any(
            haystack,
            (
                "Loading",
                "Loading WhatsApp",
                "Connecting",
                "Connecting to WhatsApp",
                "Your messages are downloading",
                "messages are downloading",
                "Don't close this window",
            ),
        ):
            return "LOADING"

        return "UNKNOWN"

    def wait_for_whatsapp_ready(
        self,
        *,
        timeout_s: float = _WHATSAPP_READY_TIMEOUT_S,
        poll_ms: int = 200,
    ) -> str:
        """Wait for a bounded WhatsApp readiness state using fresh observes.

        WhatsApp Web can expose its normal semantic controls while still
        performing an initial message/database sync. Keep polling fresh
        observations long enough for that startup phase instead of treating
        a slow authenticated session as a permanent failure.
        """
        import time

        deadline = time.monotonic() + max(0.1, float(timeout_s))
        last_state = "UNKNOWN"
        while time.monotonic() < deadline:
            last_state = self.whatsapp_state()
            if last_state in {"READY", "LOGIN_REQUIRED"}:
                return last_state
            remaining_ms = max(
                1,
                int((deadline - time.monotonic()) * 1000),
            )
            self.wait_ms(min(max(1, int(poll_ms)), remaining_ms))
        return last_state

    def open_whatsapp(
        self,
        *,
        timeout_s: float = _WHATSAPP_READY_TIMEOUT_S,
    ) -> BrowserSkillResult:
        """Ensure WhatsApp Web is ready without reopening an already-open page.

        The BrowserSkill session is the reusable resource boundary. If this
        resource is already on WhatsApp Web and authenticated, preserve that
        live page/current chat and continue from it. Only navigate when the
        existing resource is not already on WhatsApp Web.
        """
        if self.session:
            try:
                current_url = self.current_url()
            except BrowserSkillError:
                current_url = ""
            if "web.whatsapp.com" in str(current_url).casefold():
                state = self.wait_for_whatsapp_ready(timeout_s=timeout_s)
                if state == "READY":
                    return BrowserSkillResult({
                        "ok": True,
                        "provider": "whatsapp",
                        "state": state,
                        "url": current_url,
                        "reused": True,
                    })
                if state == "LOGIN_REQUIRED":
                    raise BrowserSkillError(
                        "WhatsApp Web requires login/QR authentication",
                        code="whatsapp_login_required",
                        data={"state": state},
                    )
                if state == "LOADING":
                    raise BrowserSkillError(
                        "WhatsApp Web did not become ready before the bounded timeout",
                        code="whatsapp_not_ready",
                        data={"state": state},
                    )
                raise BrowserSkillError(
                    "WhatsApp Web state could not be safely determined",
                    code="whatsapp_state_unknown",
                    data={"state": state},
                )

        navigation = self.navigate("https://web.whatsapp.com/")
        if not navigation.ok:
            raise BrowserSkillError(
                "WhatsApp Web navigation was rejected",
                code="whatsapp_navigation_failed",
                data={"result": navigation},
            )
        state = self.wait_for_whatsapp_ready(timeout_s=timeout_s)
        if state == "READY":
            return BrowserSkillResult({
                "ok": True,
                "provider": "whatsapp",
                "state": state,
                "url": self.current_url(),
            })
        if state == "LOGIN_REQUIRED":
            raise BrowserSkillError(
                "WhatsApp Web requires login/QR authentication",
                code="whatsapp_login_required",
                data={"state": state},
            )
        if state == "LOADING":
            raise BrowserSkillError(
                "WhatsApp Web did not become ready before the bounded timeout",
                code="whatsapp_not_ready",
                data={"state": state},
            )
        raise BrowserSkillError(
            "WhatsApp Web state could not be safely determined",
            code="whatsapp_state_unknown",
            data={"state": state},
        )

    def _whatsapp_current_elements(self) -> tuple[BrowserElement, ...]:
        obs = self._whatsapp_observation()
        return self._whatsapp_semantic_elements(obs)

    @classmethod
    def _whatsapp_semantic_elements(
        cls,
        observation: BrowserObservation,
    ) -> tuple[BrowserElement, ...]:
        """Return the complete live WhatsApp semantic element set.

        BrowserSkill normally exposes semantic refs both through structured
        data and through its rendered semantic text tree.  Some observations
        have shown a ref such as the WhatsApp search textbox in the text tree
        while that same ref is missing from ``observation.elements``.  Because
        the text tree itself contains the live ``@eN`` capability, merge both
        sources before resolving WhatsApp controls.

        No selectors, coordinates, DOM queries, or persisted refs are used.
        """
        structured = list(observation.elements)
        textual = cls._parse_observation_text(observation.text)

        merged: dict[str, BrowserElement] = {}
        for element in (*structured, *textual):
            if not element.ref:
                continue
            existing = merged.get(element.ref)
            if existing is None:
                merged[element.ref] = element
                continue

            # Annotations (e.g. the ``ctx: ic-search`` context marker) can be
            # present on one source's copy of an element and absent on the
            # other's. Union them regardless of which side "wins" the score
            # comparison below, so a richer structured match never silently
            # discards a semantic signal that only the textual parse saw.
            existing_annotations = list(
                (existing.attributes or {}).get("annotations") or []
            )
            new_annotations = list(
                (element.attributes or {}).get("annotations") or []
            )
            annotations = list(existing_annotations)
            for annotation in new_annotations:
                if annotation not in annotations:
                    annotations.append(annotation)

            existing_score = (
                bool(existing.role) * 2
                + bool(existing.name) * 2
                + bool(existing.value)
                + len(existing.name)
                + len(existing.value)
                + len(existing.raw or "")
            )
            new_score = (
                bool(element.role) * 2
                + bool(element.name) * 2
                + bool(element.value)
                + len(element.name)
                + len(element.value)
                + len(element.raw or "")
            )
            chosen = element if new_score > existing_score else existing

            merged[element.ref] = BrowserElement(
                element.ref,
                role=chosen.role or existing.role or element.role,
                name=chosen.name or existing.name or element.name,
                value=chosen.value or existing.value or element.value,
                raw=chosen.raw or existing.raw or element.raw,
                attributes=(
                    {"annotations": annotations}
                    if annotations
                    else dict(chosen.attributes or {})
                ),
            )

        return tuple(merged.values())

    def _whatsapp_search_ref(
        self,
        *,
        timeout_s: float = 10.0,
        poll_ms: int = 200,
    ) -> BrowserTarget:
        """Wait for and return the live WhatsApp search control.

        WhatsApp can briefly expose a READY-looking semantic tree while the
        search textbox is being rebuilt. Resolve the search target from a
        fresh observation on every poll instead of requiring it to exist on
        the first observation.

        The returned ref always belongs to the observation generation from
        which it was resolved.
        """
        import time

        deadline = time.monotonic() + max(0.1, float(timeout_s))
        last_candidates: list[BrowserElement] = []

        while time.monotonic() < deadline:
            obs = self._whatsapp_observation()
            elements = self._whatsapp_semantic_elements(obs)

            candidates = [
                e
                for e in elements
                if self._whatsapp_is_search_element(e)
                and "message" not in self._whatsapp_normalize(e.name)
            ]
            last_candidates = candidates

            if len(candidates) == 1:
                e = candidates[0]
                return BrowserTarget(
                    e.ref,
                    role=e.role,
                    name=e.name,
                    raw=e.raw,
                    generation=obs.generation,
                )

            if len(candidates) > 1:
                # Prefer the exact WhatsApp home-screen search control name
                # over a control only identified via its annotation context
                # (e.g. a search box whose name is now the typed query).
                exact = [
                    e
                    for e in candidates
                    if self._whatsapp_normalize(e.name)
                    in self._WHATSAPP_SEARCH_NAMES
                ]
                if len(exact) == 1:
                    e = exact[0]
                    return BrowserTarget(
                        e.ref,
                        role=e.role,
                        name=e.name,
                        raw=e.raw,
                        generation=obs.generation,
                    )

            remaining_ms = max(
                1,
                int((deadline - time.monotonic()) * 1000),
            )
            self.wait_ms(min(max(1, int(poll_ms)), remaining_ms))

        raise BrowserSkillError(
            "WhatsApp search control was not uniquely resolved; "
            f"candidates={len(last_candidates)}",
            code="whatsapp_search_not_ready",
            data={"candidates": [e.ref for e in last_candidates]},
        )

    def _whatsapp_first_search_result(
        self,
        query: str,
        *,
        timeout_s: float = 5.0,
        poll_ms: int = 200,
    ) -> BrowserTarget:
        """Resolve the first semantic result produced by WhatsApp search.

        WhatsApp search results are intentionally treated as an ordered UI
        result set rather than requiring an exact contact-name match.  The
        search query is only used to prefer visible result names containing
        the query; otherwise the first clickable result is selected.  No DOM
        selector or coordinate is used.
        """
        import time

        wanted = self._whatsapp_normalize(query)
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        result_roles = {
            "button", "link", "listitem", "menuitem", "option", "text",
        }
        # Chrome/list labels that are not real search results and must
        # never be mistaken for the first contact match. This includes the
        # sidebar tabs that sit directly next to search results (e.g. the
        # "Channels" control that can appear immediately adjacent to a
        # contact result) so a broad or misordered element never gets
        # clicked instead of the actual contact.
        chrome_names = {
            "search", "search or start a new chat",
            "search or start new chat", "new chat",
            "communities", "status", "calls",
            "chats", "contacts", "messages",
            "channels", "channel", "explore channels", "updates",
        }

        # WhatsApp debounces its search-as-you-type results by a short
        # interval. The very first observation right after fill() can still
        # show the previous, unfiltered chat list (e.g. an existing "Mummy"
        # chat), which looks like a perfectly valid clickable result even
        # though it has nothing to do with the query just entered. Give the
        # UI a brief moment to react before trusting any candidate.
        self.wait_ms(min(250, max(1, int((deadline - time.monotonic()) * 1000))))

        while time.monotonic() < deadline:
            obs = self._whatsapp_observation()
            elements = self._whatsapp_semantic_elements(obs)

            # When the search textbox itself is present in this observation,
            # confirm it actually holds the query we just typed before
            # trusting the result list below it. This is what stops a
            # leftover/default chat entry from being clicked while WhatsApp
            # is still re-rendering search results for the new query.
            search_value_seen = False
            search_value_matches = True
            for element in elements:
                if self._whatsapp_is_search_element(element):
                    if element.value:
                        search_value_seen = True
                        search_value_matches = (
                            self._whatsapp_normalize(element.value) == wanted
                        )
                    break

            if search_value_seen and not search_value_matches:
                remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
                self.wait_ms(min(max(1, int(poll_ms)), remaining_ms))
                continue

            candidates = []
            for element in elements:
                role = element.role.casefold()
                name = self._whatsapp_normalize(element.name)
                if role not in result_roles or not name:
                    continue
                if name in chrome_names:
                    continue
                candidates.append(element)

            if candidates:
                # The required WhatsApp workflow is literal: click the FIRST
                # visible selectable search result. The query is diagnostic
                # context only and is used only to break ties when several
                # candidates are visible in the same observation: a result
                # whose own name actually relates to what was searched (e.g.
                # contains "papa") is preferred over an unrelated neighboring
                # control that merely happens to share a role/list position,
                # such as a "Channels" entry rendered next to the contact.
                # When nothing relates to the query, the first candidate in
                # observed order is used, unchanged from before.
                # Prefer the contact whose OWN semantic name exactly equals
                # the search query.  WhatsApp can expose a broad parent result
                # before the real selectable child, for example:
                #
                #   @e17 button "Papa Yesterday ... Voice call"
                #   @e18 button "Papa"
                #
                # @e17 is not the contact target we want.  Exact own-name
                # matching must therefore happen before substring matching.
                exact = [
                    c for c in candidates
                    if wanted and self._whatsapp_normalize(c.name) == wanted
                ]
                if exact:
                    chosen = exact[0]
                else:
                    related = [
                        c for c in candidates
                        if wanted and (
                            wanted in self._whatsapp_normalize(c.name)
                            or self._whatsapp_normalize(c.name) in wanted
                        )
                    ]
                    chosen = related[0] if related else candidates[0]
                return BrowserTarget(
                    chosen.ref, role=chosen.role, name=chosen.name,
                    raw=chosen.raw, generation=obs.generation,
                )

            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            self.wait_ms(min(max(1, int(poll_ms)), remaining_ms))

        raise BrowserSkillError(
            f"WhatsApp search produced no selectable result for {query!r}",
            code="whatsapp_contact_not_found",
            data={"query": query},
        )

    def _whatsapp_contact_target(
        self,
        contact: str,
        *,
        timeout_s: float = 5.0,
        poll_ms: int = 200,
    ) -> BrowserTarget:
        import time

        wanted = self._whatsapp_normalize(contact)
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        last_count = 0

        while time.monotonic() < deadline:
            obs = self._whatsapp_observation()
            elements = self._whatsapp_semantic_elements(obs)
            candidates: list[BrowserElement] = []
            for element in elements:
                name = self._whatsapp_normalize(element.name)
                if name != wanted:
                    continue
                # Search results are normally links/buttons/list items. Do
                # not treat the search textbox or arbitrary generic text as a
                # contact target.
                if element.role.casefold() in {
                    "textbox", "combobox", "heading", "document", "generic",
                }:
                    continue
                candidates.append(element)

            last_count = len(candidates)
            if len(candidates) == 1:
                e = candidates[0]
                return BrowserTarget(
                    e.ref,
                    role=e.role,
                    name=e.name,
                    raw=e.raw,
                    generation=obs.generation,
                )
            if len(candidates) > 1:
                raise BrowserSkillError(
                    f"Multiple exact WhatsApp contacts matched {contact!r}",
                    code="whatsapp_contact_ambiguous",
                    data={"matches": [e.name for e in candidates]},
                )

            remaining_ms = max(
                1,
                int((deadline - time.monotonic()) * 1000),
            )
            self.wait_ms(min(max(1, int(poll_ms)), remaining_ms))

        # Fuzzy matches are deliberately not selected. Safety requires an
        # exact visible match or an explicit ambiguity/not-found outcome.
        if last_count == 0:
            raise BrowserSkillError(
                f"No exact WhatsApp contact match for {contact!r}",
                code="whatsapp_contact_not_found",
            )
        raise BrowserSkillError(
            f"WhatsApp contact {contact!r} could not be safely resolved",
            code="whatsapp_contact_ambiguous",
        )

    def _whatsapp_chat_header_matches(
        self,
        contact: str,
        *,
        timeout_s: float = 3.0,
        poll_ms: int = 200,
    ) -> bool:
        """Confirm that a WhatsApp Home-row click opened the intended chat.

        Normal chats expose the contact/group name in the conversation header.
        WhatsApp self-chat can expose ``You`` as the active header even though
        the Home row is named with the user's account display name.  In that
        case, require both the ``You`` header and the original target name to
        remain visible alongside the message composer.
        """
        import time

        wanted = self._whatsapp_normalize(contact)
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while time.monotonic() < deadline:
            obs = self._whatsapp_observation()
            elements = self._whatsapp_semantic_elements(obs)
            names = [self._whatsapp_normalize(e.name) for e in elements if e.name]
            header_matches = any(
                e.role.casefold() in {"heading", "button", "link", "text"}
                and (
                    self._whatsapp_normalize(e.name) == wanted
                    or re.fullmatch(
                        rf"{re.escape(wanted)}\s*\(you\)",
                        self._whatsapp_normalize(e.name),
                    ) is not None
                )
                for e in elements
            )
            composer_visible = any(
                (e.role.casefold() in {"textbox", "combobox", "searchbox"})
                or "type a message" in name
                or "message input" in name
                for e, name in ((e, self._whatsapp_normalize(e.name)) for e in elements)
            )
            # Self-chat may label the active header as ``You`` rather than the
            # account name, or as ``<account> (You)``. The target must still be
            # present in the same fresh observation and the composer must exist.
            self_header = any(
                e.role.casefold() in {"heading", "button", "link", "text"}
                and self._whatsapp_normalize(e.name) == "you"
                for e in elements
            )
            target_visible = any(
                wanted and wanted in self._whatsapp_normalize(e.name)
                for e in elements
            )
            if header_matches or (self_header and composer_visible and target_visible):
                return True

            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            self.wait_ms(min(max(1, int(poll_ms)), remaining_ms))

        return False

    def open_whatsapp_chat_row(
        self,
        target: BrowserTarget,
        *,
        timeout_s: float = 8.0,
    ) -> BrowserSkillResult:
        """Open an already-observed WhatsApp Home chat row.

        This is deliberately separate from ``open_whatsapp_chat`` because the
        latter is the normal explicit-recipient path and resolves a recipient
        through WhatsApp search. Intelligence discovery must never use that
        path: authorization only permits inspection of a target already found
        on the Home chat list. The supplied ``BrowserTarget`` therefore comes
        directly from the current Home observation and is consumed by the
        existing semantic click primitive.
        """
        if not isinstance(target, BrowserTarget):
            raise ValueError("target must be a BrowserTarget from the current observation")

        click_result = self.click(target)
        if not click_result.ok:
            raise BrowserSkillError(
                "WhatsApp Home chat row click was rejected",
                code="whatsapp_home_row_click_failed",
                data={"target": target.name, "ref": target.ref},
            )

        if not self._whatsapp_chat_header_matches(
            target.name,
            timeout_s=max(1.0, min(3.0, timeout_s / 2)),
        ):
            raise BrowserSkillError(
                f"WhatsApp did not confirm that the {target.name!r} chat was open after selecting the Home row",
                code="whatsapp_chat_not_verified",
                data={"target": target.name, "ref": target.ref},
            )

        return BrowserSkillResult({
            "ok": True,
            "provider": "whatsapp",
            "contact": target.name,
            "state": "CHAT_OPEN",
            "verification": "passed",
            "telemetry": {"semantic_target_category": "home_chat_row", "ref": target.ref},
        })

    def open_whatsapp_chat(
        self,
        contact: str,
        *,
        timeout_s: float = 8.0,
    ) -> BrowserSkillResult:
        """Open the first contact returned by WhatsApp's semantic search."""
        if not isinstance(contact, str) or not contact.strip():
            raise ValueError("contact must be non-empty")

        state = self.wait_for_whatsapp_ready(timeout_s=min(3.0, timeout_s))
        if state != "READY":
            if state == "LOGIN_REQUIRED":
                raise BrowserSkillError(
                    "WhatsApp Web requires login/QR authentication",
                    code="whatsapp_login_required",
                )
            raise BrowserSkillError(
                f"WhatsApp Web is not ready: {state}",
                code="whatsapp_not_ready" if state == "LOADING" else "whatsapp_state_unknown",
            )

        # Resolve the actionable search control from a fresh observation.
        # _whatsapp_search_ref() itself waits for the control if WhatsApp is
        # still rebuilding its semantic tree.
        search = self._whatsapp_search_ref(
            timeout_s=max(1.0, timeout_s),
        )
        click_search = self.click(search)
        if not click_search.ok:
            raise BrowserSkillError(
                "WhatsApp search control could not be clicked",
                code="whatsapp_chat_search_failed",
            )

        search = self._whatsapp_search_ref(
            timeout_s=max(1.0, timeout_s),
        )
        try:
            self._whatsapp_type_text(search, contact.strip())
        except BrowserSkillError as exc:
            raise BrowserSkillError(
                "WhatsApp search query could not be entered",
                code="whatsapp_search_not_ready",
                data={"contact": contact, "error": str(exc)},
            ) from exc

        # Search results are now resolved from a fresh observation.
        #
        # IMPORTANT: BrowserSkill semantic refs are observation-scoped and
        # WhatsApp may rebuild the result list while its search UI settles.
        # Never reuse an old ref.  Each click attempt performs:
        #
        #   fresh observe -> resolve contact -> click
        #
        # The resolver itself prefers an exact own-name match before related
        # names, so a broad parent such as "Papa Yesterday ... Voice call"
        # cannot win over the actual "Papa" result.
        import time
        click_result = None
        target = None
        last_click_error = None
        click_deadline = time.monotonic() + max(1.0, float(timeout_s))

        while time.monotonic() < click_deadline:
            remaining_s = max(0.5, click_deadline - time.monotonic())

            try:
                target = self._whatsapp_first_search_result(
                    contact.strip(),
                    timeout_s=min(2.5, remaining_s),
                    poll_ms=150,
                )
                try:
                    click_result = self.click(target)
                except BrowserSkillError as exc:
                    if "stale" in str(exc).casefold():
                        last_click_error = BrowserSkillError(
                            "WhatsApp chat-result semantic target became stale before click",
                            code="whatsapp_target_stale",
                            data={"semantic_target_category": "chat_result"},
                        )
                        self.wait_ms(50)
                        continue
                    raise
                if click_result.ok:
                    break
            except BrowserSkillError as exc:
                last_click_error = exc
                error_text = str(exc).casefold()

                # Retry only transient ref/visibility failures. Other
                # BrowserSkill errors should surface immediately.
                if not any(
                    marker in error_text
                    for marker in (
                        "not visible",
                        "no content quads",
                        "box model",
                        "visible descendant bounds",
                        "stale",
                        "not found",
                    )
                ):
                    raise

            if time.monotonic() >= click_deadline:
                break

            # Let WhatsApp finish the result-list re-render before taking the
            # next fresh observation.
            self.wait_ms(150)

        if click_result is None or not click_result.ok:
            raise BrowserSkillError(
                "WhatsApp search result click was rejected",
                code="whatsapp_chat_search_failed",
                data={
                    "contact": contact,
                    "result": target.name if target is not None else "",
                    "ref": target.ref if target is not None else "",
                    "error": str(last_click_error) if last_click_error else None,
                },
            )

        # The contact click above is the operation that opens the chat, but a
        # single observation taken immediately after the click is prone to
        # false negatives while WhatsApp rebuilds its semantic tree. Poll
        # several fresh observations for the requested contact's own chat
        # header before trusting that the correct chat is open. This is what
        # stops send_whatsapp_message() from typing into whatever chat
        # happened to be open previously.
        chat_confirmed = self._whatsapp_chat_header_matches(
            contact.strip(),
            timeout_s=max(1.0, min(3.0, timeout_s / 2)),
        )
        if not chat_confirmed:
            raise BrowserSkillError(
                f"WhatsApp did not confirm that the {contact!r} chat was open "
                "after selecting the search result",
                code="whatsapp_chat_not_verified",
                data={
                    "contact": contact,
                    "selected_result": target.name if target is not None else "",
                },
            )

        return BrowserSkillResult({
            "ok": True,
            "provider": "whatsapp",
            "contact": contact,
            "selected_result": target.name if target is not None else "",
            "state": "CHAT_OPEN",
            "verification": "passed",
            "telemetry": {"semantic_target_category": "chat_result"},
        })

    def _whatsapp_type_text(
        self,
        target: str | BrowserTarget,
        text: str,
    ) -> BrowserSkillResult:
        """Type exact text into WhatsApp using BrowserSkill keyboard events.

        WhatsApp Web's composer is a contenteditable textbox that does not
        reliably accept the generic ``fill`` command. This helper instead
        clicks the composer and drives it character-by-character through the
        existing ``press`` primitive, which has been confirmed to work
        reliably against WhatsApp Web's composer.

        The target is resolved to a plain ref ONCE up front. All subsequent
        keyboard operations reuse that ref string rather than the original
        ``BrowserTarget``, since ``BrowserTarget`` generation checks are tied
        to observation generation while the repeated keyboard events mutate
        the live browser command state.
        """

        if not isinstance(text, str) or not text:
            raise ValueError("text must be non-empty")

        ref = self._coerce_ref(target)


        last_result: BrowserSkillResult | None = None

        for char in text:
            key = _WHATSAPP_KEY_MAP.get(char, char)
            last_result = self.press(ref, key)

        return last_result or BrowserSkillResult({"ok": True})

    def _whatsapp_composer_target(
        self,
        *,
        timeout_s: float = 5.0,
    ) -> BrowserTarget:
        """Resolve the CURRENT WhatsApp message composer from a fresh observe.

        Composer refs are observation-scoped. Opening the chat and filling the
        composer can invalidate the previous ref-store, so this helper never
        persists or guesses an ``@eN`` reference.
        """
        import time

        deadline = time.monotonic() + max(0.1, float(timeout_s))
        last_candidates: list[BrowserElement] = []

        while time.monotonic() < deadline:
            obs = self._whatsapp_observation()
            elements = self._whatsapp_semantic_elements(obs)

            # Prefer the exact semantic composer label from THIS observation.
            exact = [
                e for e in elements
                if e.role.casefold() in {"textbox", "combobox"}
                and self._whatsapp_normalize(e.name) == "type a message"
            ]
            if len(exact) == 1:
                e = exact[0]
                return BrowserTarget(
                    e.ref,
                    role=e.role,
                    name=e.name,
                    raw=e.raw,
                    generation=obs.generation,
                )

            # Compatibility with builds that expose the composer as
            # "message", "enter message", etc.  WhatsApp Web can also expose
            # the live composer as a generic node instead of a textbox.
            message_names = {
                "message",
                "enter message",
                "write a message",
                "type message",
                "type a message",
            }
            message_candidates = [
                e for e in elements
                if e.role.casefold() in {"textbox", "combobox", "generic"}
                and (
                    self._whatsapp_normalize(e.name) in message_names
                    or "message" in self._whatsapp_normalize(e.name)
                )
            ]
            last_candidates = [*exact, *message_candidates]
            if len(message_candidates) == 1 and not exact:
                e = message_candidates[0]
                return BrowserTarget(
                    e.ref,
                    role=e.role,
                    name=e.name,
                    raw=e.raw,
                    generation=obs.generation,
                )

            # Last semantic fallback: after the chat-opening click, WhatsApp
            # normally leaves the search box plus exactly one other editable
            # control.  If the composer has no accessible name, select the
            # non-search textbox/combobox from THIS fresh observation.  The
            # search box's accessible name can be its own fixed placeholder
            # OR the currently typed query (e.g. "Papa"), so exclusion uses
            # the same semantic signal as the search resolver rather than a
            # name substring check that only catches the placeholder case.
            unnamed_editables = [
                e for e in elements
                if e.role.casefold() in {"textbox", "combobox"}
                and not self._whatsapp_is_search_element(e)
            ]
            if len(unnamed_editables) == 1:
                e = unnamed_editables[0]
                return BrowserTarget(
                    e.ref,
                    role=e.role,
                    name=e.name,
                    raw=e.raw,
                    generation=obs.generation,
                )

            remaining_ms = max(
                1,
                int((deadline - time.monotonic()) * 1000),
            )
            self.wait_ms(min(200, remaining_ms))

        raise BrowserSkillError(
            "WhatsApp message composer was not ready",
            code="whatsapp_composer_not_found",
            data={"candidates": [e.ref for e in last_candidates]},
        )

    def _whatsapp_wait_for_composer_text(
        self,
        expected: str,
        *,
        timeout_s: float = 3.0,
        poll_ms: int = 150,
    ) -> bool:
        """Poll fresh observations to confirm the composer holds ``expected``.

        Uses the same semantic identification as ``_whatsapp_composer_target``
        (a fresh observation every iteration; no persisted ref) so this
        checks the live composer's current value rather than trusting that an
        earlier ``press()`` sequence landed correctly. Called before Send is
        clicked so a message is never submitted without first confirming it
        is actually present in the composer.
        """
        import time

        wanted = self._whatsapp_normalize(expected)
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        message_names = {
            "message",
            "enter message",
            "write a message",
            "type message",
            "type a message",
        }

        while time.monotonic() < deadline:
            obs = self._whatsapp_observation()
            elements = self._whatsapp_semantic_elements(obs)

            for e in elements:
                role = e.role.casefold()
                if role not in {"textbox", "combobox", "generic"}:
                    continue
                name = self._whatsapp_normalize(e.name)
                if (
                    name not in message_names
                    and "message" not in name
                ):
                    continue
                if self._whatsapp_normalize(e.value) == wanted:
                    return True

            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            self.wait_ms(min(max(1, int(poll_ms)), remaining_ms))

        return False

    def _whatsapp_send_target(
        self,
        *,
        timeout_s: float = 5.0,
    ) -> BrowserTarget:
        import time

        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while time.monotonic() < deadline:
            obs = self._whatsapp_observation()
            elements = self._whatsapp_semantic_elements(obs)
            candidates = [
                e for e in elements
                if e.role.casefold() == "button"
                and self._whatsapp_normalize(e.name) in {
                    "send", "send message",
                }
            ]
            if len(candidates) == 1:
                e = candidates[0]
                return BrowserTarget(
                    e.ref, role=e.role, name=e.name, raw=e.raw,
                    generation=obs.generation,
                )
            if len(candidates) > 1:
                raise BrowserSkillError(
                    "WhatsApp send control was ambiguous",
                    code="whatsapp_send_control_not_found",
                    data={"matches": [e.name for e in candidates]},
                )
            self.wait_ms(min(200, max(1, int((deadline - time.monotonic()) * 1000))))
        raise BrowserSkillError(
            "WhatsApp send control was not ready",
            code="whatsapp_send_control_not_found",
        )

    def send_whatsapp_message(
        self,
        contact: str,
        message: str,
        *,
        timeout_s: float = _WHATSAPP_SEND_TIMEOUT_S,
    ) -> BrowserSkillResult:
        """Send an exact message through WhatsApp Web and return only execution state.

        Independent verification is performed by ``verify_whatsapp_message``;
        this method never treats a click result as proof of delivery.
        """
        if not isinstance(contact, str) or not contact.strip():
            raise ValueError("contact must be non-empty")
        if not isinstance(message, str) or not message:
            raise ValueError("message must be non-empty")

        self.open_whatsapp(timeout_s=min(_WHATSAPP_READY_TIMEOUT_S, timeout_s))
        self.open_whatsapp_chat(
            contact.strip(),
            timeout_s=max(8.0, timeout_s / 2),
        )

        # The chat-opening click invalidates the search/result ref-store.
        # Resolve the composer from a NEW observation after the chat is open.
        # WhatsApp can rebuild the composer immediately after the chat-opening
        # click.  A target can therefore be valid when observed but lose focus
        # during the fill command.  Resolve -> focus -> fill, and retry once
        # from a completely fresh observation if the browser reports that the
        # target changed/lost focus.
        import time
        fill_deadline = time.monotonic() + max(2.0, timeout_s / 2)
        fill_result = None
        last_fill_error = None

        while time.monotonic() < fill_deadline:
            composer = self._whatsapp_composer_target(
                timeout_s=min(2.0, max(0.5, fill_deadline - time.monotonic())),
            )
            try:
                # Never focus the BrowserTarget here: focus() invalidates its
                # observation generation. The typing helper resolves once and
                # sends direct keyboard events using the plain live ref.
                fill_result = self._whatsapp_type_text(composer, message)
                if fill_result.ok:
                    break
                last_fill_error = str(fill_result)
            except BrowserSkillError as exc:
                last_fill_error = str(exc)

            # The failed typing attempt may have invalidated the ref.  Never
            # reuse it; wait briefly and resolve the composer again from a
            # fresh observation.
            if time.monotonic() < fill_deadline:
                self.wait_ms(150)

        if fill_result is None or not fill_result.ok:
            fill_error_text = str(last_fill_error or "").casefold()
            fill_code = "whatsapp_target_stale" if "stale" in fill_error_text else "whatsapp_composer_not_found"
            raise BrowserSkillError(
                "WhatsApp message composer rejected the message",
                code=fill_code,
                data={
                    "contact": contact,
                    "stage": "message_composer",
                    "message_body_length": len(message),
                    "error": last_fill_error,
                    "semantic_target_category": "message_composer",
                },
            )

        # Confirm the exact message is actually present in the composer,
        # from a fresh observation, before ever clicking Send. A message is
        # never submitted on the strength of the press() sequence alone.
        composer_confirmed = self._whatsapp_wait_for_composer_text(
            message,
            timeout_s=max(1.0, min(3.0, timeout_s / 4)),
        )
        if not composer_confirmed:
            raise BrowserSkillError(
                "WhatsApp message composer did not confirm the typed message "
                "before sending",
                code="whatsapp_send_failed",
                data={
                    "contact": contact,
                    "stage": "verify_before_send",
                    "message_body_length": len(message),
                    "semantic_target_category": "message_composer",
                },
            )

        # Typing mutates the UI and invalidates refs. Prefer the semantic Send
        # button when WhatsApp exposes one. Some WhatsApp Web builds expose
        # only the composer and submit on Enter, so Enter is a semantic
        # fallback on the freshly observed composer rather than a DOM/selector
        # shortcut. Enter is never used as the fallback for a multiline
        # message: Shift+Enter is what typed the newlines, and a plain Enter
        # in that same composer would submit early with a truncated message.
        try:
            # fill() invalidates the composer ref-store. Resolve Send from a
            # fresh observation before clicking it.
            send = self._whatsapp_send_target(
                timeout_s=max(0.8, timeout_s / 4),
            )
            result = self.click(send)
        except BrowserSkillError as exc:
            if exc.code != "whatsapp_send_control_not_found":
                raise
            if "\n" in message or "\r" in message:
                raise BrowserSkillError(
                    "WhatsApp send control was not available and Enter is "
                    "not a safe fallback for a multiline message",
                    code="whatsapp_send_failed",
                    data={"contact": contact, "stage": "submit"},
                ) from exc
            composer = self._whatsapp_composer_target(
                timeout_s=max(0.8, timeout_s / 4),
            )
            result = self.press(composer, "Enter")

        if not result.ok:
            raise BrowserSkillError(
                "WhatsApp send action was rejected",
                code="whatsapp_send_failed",
                data={"contact": contact, "stage": "submit"},
            )

        return BrowserSkillResult({
            "ok": True,
            "provider": "whatsapp",
            "recipient": contact,
            "message_length": len(message),
            "action": "send_requested",
            "telemetry": {
                "recipient_target": contact,
                "message_body_length": len(message),
                "semantic_target_categories": [
                    "chat_search",
                    "chat_result",
                    "message_composer",
                    "send_control",
                ],
            },
        })

    def verify_whatsapp_message(
        self,
        contact: str,
        message: str,
        *,
        timeout_s: float = 3.0,
    ) -> BrowserSkillResult:
        """Independently verify an outgoing WhatsApp message.

        Every polling iteration performs a fresh semantic observation.  A
        known current chat with no expected message is FAIL; inability to
        establish the current chat context remains UNKNOWN.
        """
        import time

        wanted_contact = self._whatsapp_normalize(contact)
        wanted_message = self._whatsapp_normalize(message)
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        last_contact_visible = False
        last_text = ""

        while time.monotonic() < deadline:
            obs = self._whatsapp_observation()
            elements = self._whatsapp_semantic_elements(obs)
            url = (obs.url or self.current_url()).casefold()
            if "web.whatsapp.com" not in url:
                return BrowserSkillResult({
                    "ok": False,
                    "status": "FAIL",
                    "code": "whatsapp_not_active",
                })

            text = self._whatsapp_normalize(obs.text)
            semantic = "\n".join(
                self._whatsapp_normalize(
                    f"{e.name} {e.value} {e.raw if isinstance(e.raw, str) else ''}"
                )
                for e in elements
            )
            haystack = f"{text}\n{semantic}"
            message_visible = wanted_message in haystack
            contact_visible = any(
                self._whatsapp_normalize(e.name) == wanted_contact
                and e.role.casefold() in {"heading", "button", "link", "text"}
                for e in elements
            )
            last_contact_visible = contact_visible
            last_text = haystack[:500]

            if message_visible and contact_visible:
                return BrowserSkillResult({
                    "ok": True,
                    "status": "PASS",
                    "code": "whatsapp_message_verified",
                })

            # Once the intended chat is independently visible, absence of the
            # exact expected message is evidence of failure, not uncertainty.
            if contact_visible and not message_visible:
                remaining_ms = max(
                    1,
                    int((deadline - time.monotonic()) * 1000),
                )
                self.wait_ms(min(200, remaining_ms))
                continue

            remaining_ms = max(
                1,
                int((deadline - time.monotonic()) * 1000),
            )
            self.wait_ms(min(200, remaining_ms))

        if last_contact_visible:
            return BrowserSkillResult({
                "ok": False,
                "status": "FAIL",
                "code": "whatsapp_message_not_verified",
                "detail": "intended chat was observed but the exact outgoing message was not observed",
            })

        return BrowserSkillResult({
            "ok": False,
            "status": "UNKNOWN",
            "code": "whatsapp_message_context_unknown",
            "detail": "fresh observations did not expose enough current chat context to verify the message",
            "observation": last_text,
        })

    def play_song(
        self,
        song: str,
        *,
        timeout_s: float = 15.0,
    ) -> BrowserSkillResult:
        """Open the best non-Short result for the requested song and verify playback.

        The YouTube search always uses the user's original song query.
        Selection is based only on the current semantic observation. Lyrics/
        lyrics-video results are preferred, while normal non-Short videos remain
        eligible as fallback. Shorts and speed variants are not selected.

        BrowserSkill refs remain observation-scoped; no ``@eN`` reference is
        guessed, persisted, or selected from a previous observation.
        """
        import time

        if not isinstance(song, str) or not song.strip():
            raise ValueError("song must be non-empty")

        query = song.strip()
        timeout = max(1.0, float(timeout_s))
        deadline = time.monotonic() + timeout

        # Search exactly what the user asked for. Do not append ``official``
        # or perform a second official-specific search. Candidate selection
        # happens from the fresh semantic results visible on this page.
        search_query = query

        self.navigate(
            "https://www.youtube.com/results?search_query="
            + quote_plus(search_query)
        )

        target = self._wait_for_first_video_result(
            query,
            timeout_s=max(0.5, deadline - time.monotonic()),
        )

        # Final pre-click safety boundary. Candidate filtering can never be the
        # only defense because a fresh observation/race may change what a ref
        # resolves to. Never click a resolved Shorts target.
        if self._youtube_target_is_short(target):
            if self.debug:
                self._debug_note(f"YOUTUBE_SELECTION: candidate={target.name!r} url={self.current_url()!r} shorts=true duration=unknown selected=false")
            raise BrowserSkillError(
                "YouTube Short target rejected immediately before execution",
                code="youtube_short_rejected",
                data={"target": target.ref, "url": self.current_url()},
            )
        duration = self._youtube_target_duration(target, self._last_observation)
        if duration is None or duration <= 90.0:
            if self.debug:
                self._debug_note(f"YOUTUBE_SELECTION: candidate={target.name!r} url={self.current_url()!r} shorts=false duration={duration!r} selected=false")
            raise BrowserSkillError(
                "YouTube target has no verified duration greater than 90 seconds",
                code="youtube_short_rejected",
                data={"target": target.ref, "duration": duration},
            )

        click_result = self.click(target)
        if not click_result.ok:
            raise BrowserSkillError(
                "YouTube video click was rejected",
                code="media_click_failed",
                data=click_result,
            )

        # A successful click is not evidence of navigation. Freshly observe the
        # destination. A Shorts URL is never success; recover once by returning
        # to the search and selecting the next valid long-form candidate.
        recovered = False
        while time.monotonic() < deadline:
            current = self.current_url().casefold()
            if "youtube.com/shorts/" in current or "youtube.com/shorts" in current:
                if recovered:
                    raise BrowserSkillError(
                        "YouTube navigated to Shorts twice; playback rejected",
                        code="youtube_short_rejected",
                        data={"current_url": current},
                    )
                recovered = True
                self.navigate(
                    "https://www.youtube.com/results?search_query=" + quote_plus(search_query)
                )
                target = self._wait_for_first_video_result(
                    query, timeout_s=max(0.5, deadline - time.monotonic())
                )
                if self._youtube_target_is_short(target):
                    raise BrowserSkillError(
                        "Recovered YouTube target is still a Short",
                        code="youtube_short_rejected",
                        data={"target": target.ref},
                    )
                click_result = self.click(target)
                continue
            if "youtube.com/watch" in current:
                break
            self.wait_ms(200)
        else:
            current = self.current_url()
            raise BrowserSkillError(
                "YouTube result click did not navigate to a watch page; "
                f"current_url={current!r}; selected_ref={target.ref!r}; "
                f"selected_name={target.name!r}",
                code="media_navigation_failed",
                data={
                    "current_url": current,
                    "selected_ref": target.ref,
                    "selected_name": target.name,
                },
            )

        # Fresh observation before declaring success. This catches semantic Shorts
        # evidence even when the URL was rewritten or redirected.
        self.observe()
        fresh = self._last_observation
        if fresh is not None and self._song_result_is_short(
            BrowserElement("@current", role="link", name=str(self.page_title()), raw={"url": self.current_url(), "title": self.page_title()}),
            observation=fresh,
        ):
            raise BrowserSkillError(
                "Fresh YouTube observation identifies Shorts; playback is not successful",
                code="youtube_short_rejected",
                data={"current_url": self.current_url()},
            )

        playback_timeout = min(5.0, max(1.0, deadline - time.monotonic()))
        try:
            self.wait_for_playback(timeout_s=playback_timeout)
            playback = "verified"
        except BrowserSkillError as exc:
            playback = f"unknown: {exc}"

        return BrowserSkillResult({
            "ok": True,
            "query": query,
            "search_query": search_query,
            "selected_ref": target.ref,
            "selected_name": target.name,
            "playback": playback,
            "result": click_result,
        })

    @staticmethod
    def _observation_has_blocking_layer(text: str) -> bool:
        """Detect BrowserSkill's semantic signal for an occluding layer."""
        folded = str(text or "").casefold()
        return (
            "modal cover=100%" in folded
            or "occluded by" in folded
        )

    @classmethod
    def _song_request_explicitly_selects_variant(cls, query: str) -> bool:
        """Return whether the user's request explicitly asks for a variant."""
        normalized = cls._normalize_text(query)
        return any(
            re.search(pattern, normalized, re.IGNORECASE)
            for pattern in (
                r"\b(?:remix|rework|mix)\b",
                r"\b(?:lyrics?|lyric\s+video)\b",
                r"\b(?:sped\s*up|speed\s*up|speeded\s*up)\b",
                r"\b(?:slowed|slowed\s*down|speed\s*down)\b",
                r"\b(?:nightcore|8d|acoustic|instrumental|karaoke|cover|live)\b",
                r"\b(?:short|shorts)\b",
            )
        )

    @classmethod
    def _song_variant_kind(cls, text: str) -> str:
        """Classify a semantic result title for song-selection policy."""
        normalized = cls._normalize_text(text)
        if re.search(r"\b(?:shorts?|yt\s*shorts?)\b", normalized):
            return "short"
        if re.search(
            r"\b(?:sped\s*up|speed\s*up|speeded\s*up|speed\s*down|speed\s*downed)\b",
            normalized,
        ):
            return "speed"
        if re.search(r"\b(?:slowed|slowed\s*down|slow\s*down|slowed\s*\+\s*reverb)\b", normalized):
            return "speed"
        if re.search(r"\b(?:remix|rework)\b", normalized):
            return "remix"
        if re.search(r"\b(?:lyrics?|lyric\s+video)\b", normalized):
            return "lyrics"
        if re.search(r"\b(?:cover|reaction|live|acoustic|instrumental|karaoke|nightcore|8d|edit|mashup)\b", normalized):
            return "alternate"
        return "normal"

    @classmethod
    def _song_duration_seconds(
        cls, element: BrowserElement, *, observation: BrowserObservation | None = None
    ) -> float | None:
        """Extract a visible YouTube duration from semantic element data.

        Returns ``None`` when BrowserSkill did not expose duration metadata.
        This is intentionally conservative: a known very short duration is
        rejected, while an unknown duration is allowed unless the result is
        explicitly identified as a Short.
        """
        values: list[str] = [str(element.name or ""), str(element.value or "")]

        def collect(value: object) -> None:
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if str(key).casefold() in {
                        "duration", "length", "aria-label", "label", "title",
                    }:
                        collect(item)
                    elif isinstance(item, (dict, list, tuple)):
                        collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(element.raw)
        if observation is not None:
            # BrowserSkill may expose duration only in the fresh semantic
            # observation rather than on the normalized element. Inspect only
            # records belonging to this ref so another result cannot lend its
            # duration to the selected candidate.
            def collect_observation(value: object) -> None:
                if isinstance(value, dict):
                    refs = {str(v) for k, v in value.items() if str(k).casefold() in {"ref", "semantic_ref", "semanticref", "id"}}
                    if element.ref in refs:
                        for k, item in value.items():
                            if str(k).casefold() in {"duration", "length", "aria-label", "label", "title", "text"}:
                                collect(item)
                    for item in value.values():
                        if isinstance(item, (dict, list, tuple)):
                            collect_observation(item)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        collect_observation(item)
            collect_observation(observation.raw)
            for line in str(observation.text or "").splitlines():
                if element.ref in line:
                    values.append(line)

        patterns = (
            re.compile(r"\b(\d+)\s*(?:hours?|hrs?)\s*(?:(\d+)\s*(?:minutes?|mins?))?\s*(?:(\d+)\s*(?:seconds?|secs?))?\b", re.I),
            re.compile(r"\b(\d+)\s*(?:minutes?|mins?)\s*(?:(\d+)\s*(?:seconds?|secs?))?\b", re.I),
            re.compile(r"\b(\d+)\s*(?:seconds?|secs?)\b", re.I),
            re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b"),
        )
        for text in values:
            for pattern in patterns:
                match = pattern.search(text)
                if not match:
                    continue
                groups = match.groups()
                try:
                    if len(groups) == 3 and "hours?" in pattern.pattern:
                        hours = int(groups[0])
                        minutes = int(groups[1] or 0)
                        seconds = int(groups[2] or 0)
                        return float(hours * 3600 + minutes * 60 + seconds)
                    if ":" in pattern.pattern:
                        first, second, third = groups
                        if third is not None:
                            return float(int(first) * 3600 + int(second) * 60 + int(third))
                        return float(int(first) * 60 + int(second))
                    minutes_or_seconds = int(groups[0])
                    if len(groups) > 1 and groups[1] is not None:
                        return float(minutes_or_seconds * 60 + int(groups[1]))
                    return float(minutes_or_seconds)
                except (TypeError, ValueError):
                    continue
        return None

    @classmethod
    def _song_result_is_short(
        cls,
        element: BrowserElement,
        *,
        observation: BrowserObservation | None = None,
    ) -> bool:
        """Hard-reject Shorts using all current semantic evidence available."""
        text_parts = [str(element.name or ""), str(element.value or "")]
        metadata: list[str] = []
        hrefs: list[str] = []

        for key, value in element.attributes.items():
            if str(key).casefold() == "href" and isinstance(value, str):
                hrefs.append(value)

        def collect_links(value: object) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    key_folded = str(key).casefold()
                    if key_folded in {
                        "href", "url", "link", "target_url", "targeturl",
                    } and isinstance(item, str):
                        hrefs.append(item)
                    elif key_folded in {
                        "type", "category", "result_type", "resulttype",
                        "content_type", "contenttype", "kind", "aria-label",
                        "arialabel", "label", "title", "text",
                    } and isinstance(item, str):
                        metadata.append(item)
                    if isinstance(item, (dict, list, tuple)):
                        collect_links(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect_links(item)

        collect_links(element.raw)
        if observation is not None:
            def collect_ref(value: object) -> None:
                if isinstance(value, dict):
                    refs = {
                        str(v) for k, v in value.items()
                        if str(k).casefold() in {"ref", "semantic_ref", "semanticref", "id"}
                        for v in ([v] if not isinstance(v, (list, tuple)) else v)
                    }
                    if element.ref in refs:
                        for k, v in value.items():
                            kf = str(k).casefold()
                            if isinstance(v, str) and kf in {"href", "url", "link", "target_url", "targeturl"}:
                                hrefs.append(v)
                            elif isinstance(v, str) and kf in {"type", "category", "result_type", "resulttype", "content_type", "contenttype", "kind", "aria-label", "arialabel", "label", "title", "text"}:
                                metadata.append(v)
                                text_parts.append(v)
                    for v in value.values():
                        collect_ref(v)
                elif isinstance(value, (list, tuple)):
                    for v in value:
                        collect_ref(v)
            collect_ref(observation.raw)
            for line in str(observation.text or "").splitlines():
                if element.ref in line:
                    text_parts.append(line)

        text = " ".join(text_parts).casefold()
        metadata_text = " ".join(metadata).casefold()
        if re.search(r"\b(?:shorts?|yt\s*shorts?)\b", text):
            return True
        if re.search(r"\b(?:shorts?|yt\s*shorts?)\b", metadata_text):
            return True

        # Shorts can be exposed through any URL-like field rather than the
        # title. Treat every YouTube Shorts URL as a hard rejection.
        for href in hrefs:
            normalized_href = href.strip().casefold()
            if (
                "/shorts/" in normalized_href
                or "youtube.com/shorts" in normalized_href
                or "youtu.be/shorts" in normalized_href
            ):
                return True

        return False

    @classmethod
    def _rank_song_candidates(
        cls,
        query: str,
        elements: Sequence[BrowserElement],
        *,
        observation: BrowserObservation | None = None,
    ) -> list[tuple[int, BrowserElement]]:
        """Rank visible YouTube results, preferring lyrics videos.

        ``play_song`` searches with the user's exact query. From the fresh
        semantic observation, Shorts are hard-rejected. Lyrics is a deterministic
        preference, while normal non-Short videos remain eligible as fallback.
        """
        q = cls._normalize_text(query)
        if not q:
            return []

        ranked: list[tuple[int, BrowserElement]] = []
        for element in elements:
            if not element.ref or element.role.casefold().strip() != "link":
                continue

            name = cls._normalize_text(element.name)
            value = cls._normalize_text(element.value)
            if not name:
                continue

            haystack = f"{name} {value}".strip()

            # Shorts are never eligible for normal song playback. The check
            # uses semantic title/metadata and URL evidence from the current
            # observation rather than coordinates or thumbnail shape.
            if cls._song_result_is_short(element, observation=observation):
                continue

            # Music playback has a hard minimum duration. Unknown duration is
            # also rejected here: without evidence that a result is longer than
            # 90 seconds, the runtime must not select it and discover too late
            # that it is effectively a Short/clip.
            duration = cls._song_duration_seconds(element, observation=observation)
            if duration is None or duration <= 90.0:
                continue

            is_lyrics = bool(
                re.search(
                    r"\b(?:lyrics?|lyric\s+video)\b",
                    haystack,
                    re.IGNORECASE,
                )
            )
            if re.search(
                r"\b(?:sped\s*up|speed\s*up|speeded\s*up|slowed(?:\s*down)?|slow\s*down|speed\s*down|nightcore)\b",
                haystack,
                re.IGNORECASE | re.VERBOSE,
            ):
                continue

            score = cls._target_score(
                q,
                name=name,
                value=value,
                role=element.role.casefold().strip(),
                preferred_roles={"link"},
            )

            name_tokens = set(name.split())
            q_tokens = set(q.split())
            if name == q:
                score += 600
            elif name.startswith(q + " "):
                score += 260
            covered = len(q_tokens & name_tokens)
            score += 80 * covered
            if q_tokens and q_tokens <= name_tokens:
                score += 120

            # Lyrics are the primary music preference. Once Shorts and short
            # clips are excluded, a valid lyrics video must beat every valid
            # normal video, regardless of the normal title-match score.
            if re.search(r"\bofficial\s+lyrics?\b|\blyric\s+video\b", haystack, re.IGNORECASE):
                score += 100000
            elif re.search(r"\blyrics?\b", haystack, re.IGNORECASE):
                score += 100000

            # Keep normal videos eligible, but demote obvious alternate
            # versions so a lyrics result wins over a remix/cover/live edit
            # when the query match is otherwise comparable. This is a small
            # deterministic preference, not a title-specific allow/deny list.
            if re.search(
                r"\b(?:remix|rework|cover|reaction|live|acoustic|instrumental|karaoke|nightcore|8d|edit|mashup)\b",
                haystack,
                re.IGNORECASE,
            ):
                score -= 500

            if len(q_tokens) <= 3 and len(name.split()) > 12:
                score -= 80

            ranked.append((score, element))

        ranked.sort(
            key=lambda item: (
                -item[0],
                cls._normalize_text(item[1].name),
                item[1].ref,
            )
        )
        return ranked


    @classmethod
    def _youtube_target_duration(
        cls, target: BrowserTarget, observation: BrowserObservation | None = None
    ) -> float | None:
        """Read duration from the resolved target or its owning observation."""
        duration = cls._song_duration_seconds(
            BrowserElement(target.ref, role=target.role, name=target.name, raw=target.raw),
            observation=observation,
        )
        return duration

    @classmethod
    def _youtube_target_is_short(cls, target: BrowserTarget) -> bool:
        """Final target guard: reject Shorts immediately before execution."""
        raw = target.raw
        values: list[str] = [str(target.name or ""), str(raw or "")]
        if isinstance(raw, dict):
            for key, value in raw.items():
                if str(key).casefold() in {"href", "url", "link", "target_url", "targeturl", "title", "type", "category", "kind"}:
                    values.append(str(value))
        haystack = " ".join(values).casefold()
        return bool(re.search(r"\bshorts?\b", haystack) or re.search(r"youtube\.com/shorts(?:/|$)", haystack) or re.search(r"youtu\.be/shorts(?:/|$)", haystack))

    def _debug_note(self, message: str) -> None:
        if getattr(self, "debug", False):
            print(message)

    @classmethod
    def _element_href(cls, element: BrowserElement, *, observation: BrowserObservation | None = None) -> str:
        values: list[str] = []
        if isinstance(element.attributes.get("href"), str):
            values.append(element.attributes["href"])
        raw = element.raw
        if isinstance(raw, dict):
            for key, value in raw.items():
                if str(key).casefold() in {"href", "url", "link", "target_url", "targeturl"} and isinstance(value, str):
                    values.append(value)
        if observation is not None:
            for line in str(observation.text or "").splitlines():
                if element.ref in line:
                    match = re.search(r"https?://\S+", line)
                    if match:
                        values.append(match.group(0).rstrip(".,)"))
        return values[0] if values else ""

    def _wait_for_first_video_result(
        self,
        query: str,
        *,
        timeout_s: float,
    ) -> BrowserTarget:
        """Wait for fresh semantic song candidates and choose the best one."""
        import time

        deadline = time.monotonic() + max(0.1, float(timeout_s))
        last_count = 0
        last_names: list[str] = []
        dismiss_attempts = 0

        while True:
            self.observe()
            observation = self._last_observation
            elements = [] if observation is None else list(observation.elements)
            last_count = len(elements)
            last_names = [e.name for e in elements if e.name][:8]

            if (
                self._observation_has_blocking_layer(
                    observation.text if observation is not None else ""
                )
                and dismiss_attempts < 3
                and time.monotonic() < deadline
            ):
                dismiss_attempts += 1
                try:
                    self.press_key("Escape")
                finally:
                    self._invalidate_refs()
                remaining = max(1, int((deadline - time.monotonic()) * 1000))
                self.wait_ms(min(_DEFAULT_POLL_MS, remaining))
                continue

            ranked = self._rank_song_candidates(query, elements, observation=observation)
            if ranked:
                _, best = ranked[0]
                if self.debug:
                    duration = self._song_duration_seconds(best, observation=observation)
                    self._debug_note(
                        f"YOUTUBE_SELECTION: candidate={best.name!r} url={self._element_href(best, observation=observation)!r} "
                        f"shorts={self._song_result_is_short(best, observation=observation)} duration={duration!r} selected=true"
                    )
                return BrowserTarget(
                    best.ref,
                    role=best.role,
                    name=best.name,
                    raw=best.raw,
                    generation=observation.generation if observation is not None else self._generation,
                )

            if time.monotonic() >= deadline:
                raise BrowserSkillError(
                    f"Timed out after {timeout_s:.1f}s waiting for a suitable YouTube song result "
                    f"for {query!r}; last_observation_elements={last_count}; "
                    f"last_names={last_names!r}; url={self.current_url()!r}",
                    code="semantic_song_result_not_ready",
                    data={
                        "query": query,
                        "elements": last_count,
                        "last_names": last_names,
                    },
                )

            remaining = max(1, int((deadline - time.monotonic()) * 1000))
            self.wait_ms(min(_DEFAULT_POLL_MS, remaining))

    def wait_for_playback(self, *, timeout_s: float = 5.0) -> None:
        """Wait for semantic playback evidence, with evaluate as a fallback."""
        deadline = __import__("time").monotonic() + max(0.2, float(timeout_s))
        last = ""
        while __import__("time").monotonic() < deadline:
            obs = self.observe()
            text = str(self._find_first_value(obs, ("text",)) or "")
            lower = text.casefold()
            last = text[:300]
            if re.search(r"\bpause\b", lower) and not re.search(r"\bplay\b", lower):
                return
            # A play control can coexist with a pause control elsewhere.
            if re.search(r"button \"pause\"|button \"pause video\"", lower):
                return
            self.wait_ms(200)
        # Last resort: BrowserSkill evaluate, limited to media UI state.
        try:
            result = self.evaluate("(() => { const v=document.querySelector('video'); return {exists:!!v, paused:v ? v.paused : null, currentTime:v ? v.currentTime : null, readyState:v ? v.readyState : null}; })()")
            state = self._find_first_value(result, ("result", "value", "data"))
            if isinstance(state, dict) and state.get("exists") and state.get("paused") is False and float(state.get("currentTime", 0) or 0) > 0:
                return
        except Exception:
            pass
        raise BrowserSkillError(f"playback could not be independently confirmed; observation={last!r}")

    def open_application(
        self,
        url: str,
    ) -> BrowserSkillResult:
        """Generic compatibility helper."""

        return self.navigate(url)

    def find_refs(self, *, role: str | None = None, labels: Iterable[str] = ()) -> list[str]:
        """Compatibility helper returning refs from one fresh observation."""
        self.observe()
        labels_n = tuple(self._normalize_text(x) for x in labels if str(x).strip())
        refs: list[str] = []
        for e in self._last_observation.elements if self._last_observation else ():
            if role and e.role.casefold() != role.casefold():
                continue
            hay = {self._normalize_text(e.name), self._normalize_text(e.value)}
            if labels_n and not any(any(label == h or label in h for h in hay if h) for label in labels_n):
                continue
            refs.append(e.ref)
        return refs

    def find_ref(self, labels: Iterable[str], *, role: str | None = None) -> str | None:
        labels = tuple(str(x) for x in labels if str(x).strip())
        if not labels:
            return None
        # One observation for the whole alternative-label lookup.
        self.observe()
        for label in labels:
            try:
                target = self.resolve_target(
                    label,
                    preferred_roles=(role,) if role else (),
                    observation=self._last_observation,
                    reject_ambiguous=False,
                )
                return target.ref
            except BrowserSkillError:
                continue
        return None

    # ------------------------------------------------------------------
    # Semantic parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_elements(
        value: Any,
    ) -> list[BrowserElement]:
        """Extract semantic elements from BrowserSkill observation data.

        BrowserSkill's ``observe`` response may contain a semantic text tree
        such as:

            @eN link "Magdalena Bay - Killshot 4 minutes"
            @eN button "29M views • 6 years ago"
            @eN combobox "Killshot [expanded]" ="Killshot"

        Some BrowserSkill versions may additionally expose structured
        dictionaries. Both forms are supported here.
        """

        found: list[BrowserElement] = []

        def add_element(
            element: BrowserElement,
        ) -> None:
            if not element.ref:
                return

            found.append(element)

        def visit(
            node: Any,
        ) -> None:
            # ----------------------------------------------------------
            # Semantic text
            # ----------------------------------------------------------
            if isinstance(node, str):
                for element in BrowserSkillAdapter._parse_observation_text(
                    node
                ):
                    add_element(element)

                # Some BrowserSkill response envelopes contain a serialized
                # JSON semantic payload as a string.  Decode that payload and
                # walk it recursively so live refs such as ``@e11`` are not
                # lost merely because the semantic tree was JSON-encoded.
                stripped = node.strip()
                if stripped.startswith(("{", "[")):
                    try:
                        decoded = json.loads(stripped)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        decoded = None
                    if decoded is not None and decoded is not node:
                        visit(decoded)

                return

            # ----------------------------------------------------------
            # Structured dictionaries
            # ----------------------------------------------------------
            if isinstance(node, dict):
                ref = BrowserSkillAdapter._extract_ref(node)

                if ref:
                    role = BrowserSkillAdapter._extract_string(
                        node,
                        (
                            "role",
                            "type",
                            "tag",
                            "element_type",
                            "elementType",
                        ),
                    )

                    name = BrowserSkillAdapter._extract_string(
                        node,
                        (
                            "name",
                            "label",
                            "text",
                            "description",
                            "title",
                        ),
                    )

                    element_value = BrowserSkillAdapter._extract_string(
                        node,
                        (
                            "value",
                            "input_value",
                            "inputValue",
                        ),
                    )

                    attributes = {
                        key: node[key]
                        for key in (
                            "aria-label", "ariaLabel", "title", "href",
                            "disabled", "checked", "expanded", "selected",
                            "visible", "enabled",
                            # Message ownership metadata is semantic evidence
                            # when the observation provider exposes it. Keep it
                            # on the BrowserElement so downstream consumers do
                            # not have to guess from UI presentation.
                            "from_me", "fromMe", "is_from_me", "isFromMe",
                            "outgoing", "is_outgoing", "isOutgoing", "sent_by_me",
                            "direction", "message_direction", "message_type",
                            "status", "sender", "author", "from",
                        )
                        if key in node
                    }
                    add_element(
                        BrowserElement(
                            ref,
                            role=role,
                            name=name,
                            value=element_value,
                            raw=node,
                            attributes=attributes,
                        )
                    )

                for child in node.values():
                    visit(child)

                return

            # ----------------------------------------------------------
            # Lists
            # ----------------------------------------------------------
            if isinstance(node, list):
                for child in node:
                    visit(child)

        visit(value)

        # --------------------------------------------------------------
        # Deduplicate by ref, keeping the richest element.
        # --------------------------------------------------------------
        merged: dict[str, BrowserElement] = {}

        for element in found:
            existing = merged.get(element.ref)

            if existing is None:
                merged[element.ref] = element
                continue

            existing_score = (
                bool(existing.role) * 2
                + bool(existing.name) * 2
                + bool(existing.value)
                + len(existing.name)
                + len(existing.value)
            )

            new_score = (
                bool(element.role) * 2
                + bool(element.name) * 2
                + bool(element.value)
                + len(element.name)
                + len(element.value)
            )

            if new_score > existing_score:
                merged[element.ref] = element

        return list(merged.values())

    @staticmethod
    def _parse_observation_text(
        text: str,
    ) -> list[BrowserElement]:
        """Parse BrowserSkill's semantic text representation.

        Supported examples:

            @eN link "Magdalena Bay - Killshot 4 minutes"
            @eN button "29M views • 6 years ago"
            @eN combobox "Killshot [expanded]" ="Killshot"
            @eN button "Search"
            @eN textbox "Papa" [ctx: ic-search] [default] ="Papa"
        """

        if not text:
            return []

        elements: list[BrowserElement] = []

        # BrowserSkill semantic refs are normally emitted as:
        #
        #   @eN link "Some name"
        #
        # and sometimes:
        #
        #   @eN combobox "Name" ="Value"
        #
        # Some observations additionally emit one or more bracketed
        # annotations between the quoted name and the ``=value`` assignment,
        # e.g. ``@eN textbox "Papa" [ctx: ic-search] [default] ="Papa"``.
        # These must be consumed (not just skipped over), otherwise the
        # match ends right after the quoted name and the trailing
        # ``=value`` is silently lost for that element.
        pattern = re.compile(
            r"""
            (?m)
            ^[ \t]*
            (?P<ref>@e\d+)
            [ \t]+
            (?P<role>[^\s"]+)
            (?:[ \t]+
                (?P<quoted>"(?:\\.|[^"\\])*")
            )?
            (?P<annotations>(?:[ \t]+\[[^\]\n]*\])*)
            (?:[ \t]*=[ \t]*
                (?P<value>
                    "(?:\\.|[^"\\])*"
                    |
                    [^\s]+
                )
            )?
            """,
            re.VERBOSE,
        )

        annotation_pattern = re.compile(r"\[([^\]]*)\]")

        for match in pattern.finditer(text):
            ref = match.group("ref") or ""
            role = match.group("role") or ""

            quoted_name = match.group("quoted")
            raw_value = match.group("value")

            name = BrowserSkillAdapter._decode_semantic_string(
                quoted_name
            )

            value = BrowserSkillAdapter._decode_semantic_string(
                raw_value
            )

            annotations_blob = match.group("annotations") or ""
            annotations = [
                a.strip()
                for a in annotation_pattern.findall(annotations_blob)
                if a.strip()
            ]
            attributes = {"annotations": annotations} if annotations else None

            elements.append(
                BrowserElement(
                    ref,
                    role=role,
                    name=name,
                    value=value,
                    raw=match.group(0),
                    attributes=attributes,
                )
            )

        # Compatibility with older/local renderers that put the ref at the
        # end of the line: ``button "Compose" @e4``.
        legacy = re.compile(
            r'^[ \t]*(?P<role>[^\s\"]+)\s+(?P<quoted>"(?:\\.|[^"\\])*")\s+(?P<ref>@e\d+)$',
            re.MULTILINE,
        )
        existing = {e.ref for e in elements}
        for match in legacy.finditer(text):
            ref = match.group("ref")
            if ref in existing:
                continue
            elements.append(
                BrowserElement(
                    ref,
                    role=match.group("role"),
                    name=BrowserSkillAdapter._decode_semantic_string(match.group("quoted")),
                    raw=match.group(0),
                )
            )

        return elements

    @staticmethod
    def _decode_semantic_string(
        value: str | None,
    ) -> str:
        """Decode a quoted semantic observation string."""

        if not value:
            return ""

        value = value.strip()

        if (
            len(value) >= 2
            and value[0] == '"'
            and value[-1] == '"'
        ):
            try:
                decoded = json.loads(value)

                if isinstance(decoded, str):
                    return decoded

            except json.JSONDecodeError:
                return value[1:-1]

        return value

    @staticmethod
    def _extract_ref(
        node: dict[str, Any],
    ) -> str:
        """Extract a semantic BrowserSkill ref from a dictionary."""

        for key in (
            "ref",
            "id",
            "element_ref",
            "elementRef",
            "target",
        ):
            value = node.get(key)

            if isinstance(value, str):
                stripped = value.strip()

                if re.fullmatch(
                    r"@e\d+",
                    stripped,
                ):
                    return stripped

                if re.fullmatch(
                    r"e\d+",
                    stripped,
                ):
                    return f"@{stripped}"

        # Some observation structures encode refs as "@eN" keys.
        for key in node:
            if isinstance(key, str):
                if re.fullmatch(
                    r"@e\d+",
                    key,
                ):
                    return key

        return ""

    @staticmethod
    def _extract_string(
        node: dict[str, Any],
        keys: Iterable[str],
    ) -> str:
        for key in keys:
            value = node.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    @staticmethod
    def _find_first_string(value: Any, keys: Iterable[str]) -> str:
        found = BrowserSkillAdapter._find_first_value(value, keys)
        return found.strip() if isinstance(found, str) else ""

    @staticmethod
    def _find_first_list(
        value: Any,
        keys: Iterable[str],
    ) -> list[Any] | None:
        wanted = set(keys)

        def visit(
            node: Any,
        ) -> list[Any] | None:
            if isinstance(node, dict):
                for key, item in node.items():
                    if (
                        key in wanted
                        and isinstance(item, list)
                    ):
                        return item

                for item in node.values():
                    result = visit(item)

                    if result is not None:
                        return result

            elif isinstance(node, list):
                for item in node:
                    result = visit(item)

                    if result is not None:
                        return result

            return None

        return visit(value)

    @staticmethod
    def _find_first_value(
        value: Any,
        keys: Iterable[str],
    ) -> Any:
        wanted = set(keys)

        def visit(
            node: Any,
        ) -> Any:
            if isinstance(node, dict):
                for key, item in node.items():
                    if key in wanted:
                        return item

                for item in node.values():
                    result = visit(item)

                    if result is not None:
                        return result

            elif isinstance(node, list):
                for item in node:
                    result = visit(item)

                    if result is not None:
                        return result

            return None

        return visit(value)

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        value = value.casefold()
        value = re.sub(
            r"\s+",
            " ",
            value,
        )
        return value.strip()

    def _coerce_ref(
        self,
        target: str | BrowserTarget,
    ) -> str:
        if isinstance(target, BrowserTarget):
            ref = target.ref
            if target.generation is not None and target.generation != self._generation:
                raise BrowserSkillError(
                    f"stale BrowserSkill reference {ref!r}: observation generation "
                    f"{target.generation} is no longer current (current={self._generation})"
                )
        else:
            ref = str(target).strip()

        if not ref:
            raise ValueError(
                "browser target reference cannot be empty"
            )

        if re.fullmatch(
            r"e\d+",
            ref,
        ):
            ref = f"@{ref}"

        if not re.fullmatch(
            r"@e\d+",
            ref,
        ):
            raise ValueError(
                "Invalid BrowserSkill semantic reference: "
                f"{ref!r}"
            )

        return ref


def canonical_url(url: str) -> str:
    """Canonicalize a URL for browser verification.

    Fragments are removed and scheme/host are normalized. Query parameters
    and paths are preserved.
    """

    if not isinstance(url, str):
        return ""

    value = url.strip()

    if not value:
        return ""

    try:
        parts = urlsplit(value)

    except ValueError:
        return value

    scheme = parts.scheme.lower()
    hostname = parts.hostname or ""

    try:
        port = parts.port

    except ValueError:
        port = None

    hostname = hostname.lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    netloc = hostname

    if parts.username is not None:
        username = parts.username
        password = parts.password

        credentials = username

        if password is not None:
            credentials += f":{password}"

        netloc = f"{credentials}@{netloc}"

    if port is not None:
        default_port = (
            (
                scheme == "http"
                and port == 80
            )
            or (
                scheme == "https"
                and port == 443
            )
        )

        if not default_port:
            netloc += f":{port}"

    path = parts.path or "/"

    if path != "/":
        path = path.rstrip("/")

    return urlunsplit(
        (
            scheme,
            netloc,
            path,
            parts.query,
            "",
        )
    )


__all__ = [
    "BrowserElement",
    "BrowserObservation",
    "BrowserSkillAdapter",
    "BrowserSkillCLI",
    "BrowserSkillError",
    "BrowserSkillProtocolError",
    "BrowserSkillResult",
    "BrowserSkillUnavailable",
    "BrowserTarget",
    "canonical_url",
]