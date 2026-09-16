"""Deterministic Phase-1 fast interaction routing.

This module is deliberately small: it recognizes only single, explicit browser
state transitions that can be expressed with the existing semantic BrowserSkill
actions. Anything uncertain remains on the normal planner path.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from .skills.browser import BrowserElement, BrowserObservation, BrowserSkillAdapter, BrowserTarget
from .types import Action, Check, Observation, Source, VerificationResult, Verdict


@dataclass(frozen=True)
class FastRoute:
    action_kind: str
    params: dict[str, Any]
    reason: str = ""
    target_query: str | None = None
    target_index: int | None = None
    target_roles: tuple[str, ...] = ()

    def resolve(self, browser: BrowserSkillAdapter, observation: BrowserObservation | None) -> Action:
        params = dict(self.params)
        if self.action_kind == "browser_close_tab" and "tab_id" not in params:
            tab_id = observation.tab_id if observation is not None else None
            if tab_id is None:
                try:
                    tabs = browser.list_tabs("agent")
                except Exception as exc:
                    raise ValueError("active browser tab could not be identified safely") from exc
                active = [
                    t for t in tabs
                    if t.get("active") or t.get("is_active") or t.get("selected")
                ]
                if len(active) != 1 or not isinstance(active[0].get("tab_id"), int):
                    raise ValueError("active browser tab could not be identified safely")
                tab_id = active[0]["tab_id"]
            params["tab_id"] = tab_id
        if self.target_query is not None:
            if observation is None:
                raise ValueError("fast browser target requires an observation")
            if self.target_query == "textbox":
                matches = [
                    e for e in observation.elements
                    if e.role.casefold() in {"textbox", "combobox"}
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"typing requires exactly one visible semantic textbox; found {len(matches)}"
                    )
                element = matches[0]
                target = BrowserTarget(
                    element.ref, role=element.role, name=element.name,
                    raw=element.raw, generation=observation.generation,
                )
            else:
                target = browser.resolve_target(
                    self.target_query,
                    preferred_roles=self.target_roles,
                    observation=observation,
                    reject_ambiguous=True,
                )
            params["target"] = target
        elif self.target_index is not None:
            if observation is None:
                raise ValueError("indexed browser target requires an observation")
            candidates = _indexed_candidates(
                observation.elements,
                self.target_roles,
                youtube_only=(
                    self.action_kind == "browser_click"
                    and self.target_roles == ("link",)
                    and "youtube.com" in (observation.url or "").casefold()
                ),
                observation=observation,
            )
            index = self.target_index - 1
            if index < 0 or index >= len(candidates):
                raise ValueError(
                    f"indexed browser target {self.target_index} is unavailable; "
                    f"only {len(candidates)} semantic candidates are visible"
                )
            element = candidates[index]
            params["target"] = BrowserTarget(element.ref, role=element.role, name=element.name, raw=element.raw, generation=observation.generation)
        return Action(kind=self.action_kind, params=params, consequential=True, rationale=self.reason)


@dataclass(frozen=True)
class FastClassification:
    route: FastRoute | None
    latency_seconds: float

    @property
    def matched(self) -> bool:
        return self.route is not None


_COMPLEX_MARKERS = (
    "find ", "compare ", "choose ", "pick the best", "summarize", "explain",
    "organize ", "why ", "fix ", "do whatever", "research ", "and then", " after ",
)

# Compound requests must reach decomposition intact.
_COMPOUND_ACTION_RE = re.compile(r"\band\b")
_NUMBER = r"(?:first|second|third|fourth|fifth|sixth|\d+(?:st|nd|rd|th)?)"


def classify_fast(request: str) -> FastClassification:
    started = time.perf_counter()
    text = " ".join((request or "").strip().lower().split())
    route: FastRoute | None = None

    # This is one semantic browser workflow even though it contains the word
    # "and"; do not let the generic compound guard swallow it.
    if re.fullmatch(r"open\s+youtube\s+and\s+play\s+.+", text):
        route = FastRoute("browser_play_song", {"query": text.split(" and play ", 1)[1]}, "semantic YouTube song playback")
    elif text and not any(marker in f" {text} " for marker in _COMPLEX_MARKERS) and not _COMPOUND_ACTION_RE.search(text):
        route = _classify_simple(text)

    return FastClassification(route, time.perf_counter() - started)


def _classify_simple(text: str) -> FastRoute | None:
    if text in {"scroll", "scroll down", "scroll downward"}:
        return FastRoute("browser_scroll", {"amount": 600}, "default/explicit scroll down")
    if text in {"scroll up", "scroll upward"}:
        return FastRoute("browser_scroll", {"amount": -600}, "explicit scroll up")

    m = re.fullmatch(r"scroll\s+(down|up)(?:\s+(\d+))?", text)
    if m:
        amount = int(m.group(2) or 600)
        return FastRoute("browser_scroll", {"amount": amount if m.group(1) == "down" else -amount}, "explicit scroll amount")
    m = re.fullmatch(r"scroll\s+(-?\d+)", text)
    if m:
        return FastRoute("browser_scroll", {"amount": int(m.group(1))}, "explicit scroll amount")

    simple = {
        "back": ("browser_go_back", {}),
        "go back": ("browser_go_back", {}),
        "forward": ("browser_go_forward", {}),
        "go forward": ("browser_go_forward", {}),
        "refresh": ("browser_refresh", {}),
        "reload": ("browser_refresh", {}),
        "close tab": ("browser_close_tab", {}),
    }
    if text in simple:
        kind, params = simple[text]
        return FastRoute(kind, params, f"explicit {text}")

    media_queries = {
        "play": "play",
        "pause": "pause",
        "mute": "mute",
        "unmute": "unmute",
        "volume up": "volume up",
        "volume down": "volume down",
    }
    if text in media_queries:
        return FastRoute(
            "browser_click", {}, f"explicit media control: {text}",
            target_query=media_queries[text], target_roles=("button",),
        )

    m = re.fullmatch(r"press\s+(.+)", text)
    if m and m.group(1).strip() and len(m.group(1)) <= 32:
        return FastRoute("browser_press_key", {"key": m.group(1).strip()}, "explicit key press")

    m = re.fullmatch(r"type\s+(.+)", text)
    if m and m.group(1).strip():
        return FastRoute(
            "browser_type", {"text": m.group(1)}, "explicit typing into the current semantic textbox",
            target_query="textbox", target_roles=("textbox", "combobox"),
        )

    m = re.fullmatch(rf"click\s+the\s+({_NUMBER})\s+result", text)
    if m:
        return FastRoute("browser_click", {}, "indexed semantic result click", target_index=_ordinal(m.group(1)), target_roles=("link",))

    m = re.fullmatch(rf"play\s+the\s+({_NUMBER})\s+video", text)
    if m:
        return FastRoute(
            "browser_click",
            {"verify_youtube_playback": True},
            "indexed semantic video click",
            target_index=_ordinal(m.group(1)),
            target_roles=("link",),
        )

    # Song playback is routed through BrowserSkill's dedicated YouTube workflow
    # so Shorts filtering and lyrics preference happen before any click target is
    # selected.  Keep this deterministic route intentionally small; uncertain
    # multi-step requests remain on the planner path.
    m = re.fullmatch(r"(?:open\s+youtube\s+and\s+)?play\s+(.+)", text)
    if m and m.group(1).strip():
        return FastRoute(
            "browser_play_song",
            {"query": m.group(1).strip()},
            "semantic YouTube song playback",
        )

    m = re.fullmatch(r"(?:click|press)\s+(?:the\s+)?(.+)", text)
    if m:
        target = m.group(1).strip()
        if target and not re.fullmatch(_NUMBER, target):
            return FastRoute("browser_click", {}, "explicit semantic click", target_query=target, target_roles=("button", "link", "textbox"))

    m = re.fullmatch(r"open\s+(https?://\S+)", text)
    if m:
        url = m.group(1).rstrip(".,)")
        return FastRoute("browser_open_url", {"url": url, "expected_url": url}, "explicit URL navigation")

    for phrase, url in (("youtube", "https://www.youtube.com"), ("youtube.com", "https://www.youtube.com")):
        if text in {f"open {phrase}", f"go to {phrase}", phrase}:
            return FastRoute("browser_open_url", {"url": url, "expected_url": url}, "explicit site navigation")

    # WhatsApp shorthand is deterministic and must not invoke the LLM merely
    # to discover recipient/message fields.  Keep this deliberately strict so
    # uncertain messaging requests still use the normal planner.
    m = re.fullmatch(r"send\s+(?:a\s+)?(?:whatsapp\s+)?(?:message\s+)?to\s+(.+?)\s+(?:on\s+whatsapp\s+)?(?:saying|telling)\s+(.+)", text)
    if m:
        recipient, message = m.group(1).strip(), m.group(2).strip()
        return FastRoute("whatsapp_send_message", {"recipient": recipient, "message": message}, "deterministic WhatsApp send")
    m = re.fullmatch(r"send\s+(.+?)\s+to\s+(.+?)(?:\s+on\s+whatsapp)?", text)
    if m:
        message, recipient = m.group(1).strip(), m.group(2).strip()
        if message and recipient and recipient.casefold() not in {"whatsapp", "a whatsapp"}:
            return FastRoute("whatsapp_send_message", {"recipient": recipient, "message": message}, "deterministic WhatsApp shorthand")

    return None


def _ordinal(value: str) -> int:
    words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
    if value in words:
        return words[value]
    return int(re.sub(r"\D", "", value))


def _indexed_candidates(
    elements: tuple[BrowserElement, ...],
    roles: tuple[str, ...],
    *,
    youtube_only: bool = False,
    observation: BrowserObservation | None = None,
) -> list[BrowserElement]:
    allowed = {r.casefold() for r in roles}
    result = []
    seen: set[str] = set()
    for element in elements:
        if element.ref in seen or element.role.casefold() not in allowed:
            continue
        if not element.name.strip() and not element.value.strip():
            continue
        if youtube_only and _youtube_short(element, observation=observation):
            continue
        result.append(element)
        seen.add(element.ref)
    return result


def _youtube_short(
    element: BrowserElement,
    *,
    observation: BrowserObservation | None = None,
) -> bool:
    """Hard-exclude Shorts using current semantic evidence."""
    parts = [str(element.name or ""), str(element.value or "")]
    metadata: list[str] = []
    hrefs: list[str] = []

    def collect(value: Any, *, match_ref: bool = False) -> None:
        if isinstance(value, dict):
            ref_values = {
                str(v) for k, v in value.items()
                if str(k).casefold() in {"ref", "semantic_ref", "semanticref", "id"}
                for v in ([v] if not isinstance(v, (list, tuple)) else v)
            }
            matched = match_ref or element.ref in ref_values
            for key, item in value.items():
                kf = str(key).casefold()
                if isinstance(item, str) and kf in {"href", "url", "link", "target_url", "targeturl"}:
                    if matched or match_ref:
                        hrefs.append(item)
                if isinstance(item, str) and kf in {"type", "category", "result_type", "resulttype", "content_type", "contenttype", "kind", "aria-label", "arialabel", "label", "title", "text"}:
                    if matched:
                        metadata.append(item)
                        parts.append(item)
                if isinstance(item, (dict, list, tuple)):
                    collect(item, match_ref=matched)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item, match_ref=match_ref)

    collect(element.raw, match_ref=True)
    if observation is not None:
        collect(observation.raw)
        for line in str(observation.text or "").splitlines():
            if element.ref in line:
                parts.append(line)

    text = " ".join(parts).casefold()
    metadata_text = " ".join(metadata).casefold()
    if re.search(r"\b(?:shorts|yt\s*shorts?)\b", text):
        return True
    if re.search(r"\b(?:shorts|yt\s*shorts?)\b", metadata_text):
        return True
    for href in hrefs:
        normalized = href.strip().casefold()
        if "/shorts/" in normalized or "youtube.com/shorts" in normalized or "youtu.be/shorts" in normalized:
            return True
    return False



class FastMessagingTask:
    """Deterministic one-shot messaging task.

    The action is known from the user's exact utterance, so there is no reason
    to spend an LLM round planning it.  Policy still runs normally and can
    suspend the task for approval.  After approval Session resumes the exact
    structured action through ApprovedMessagingTask/BrowserSkill.
    """
    bucket = "fast_messaging"

    def __init__(self, request: str, action: Action) -> None:
        self.request = request
        self.goal = request
        self.task_id = f"fast-{abs(hash((request, time.time_ns()))) & 0xffffffff:08x}"
        self.action = action

    def action_template(self) -> Action:
        return self.action

    def resolve_current_action(self) -> Action:
        return self.action

    def setup(self, policy: Any) -> None:
        return None

    def observe(self, policy: Any, trace: Any = None) -> dict[str, Observation]:
        browser = getattr(self, "browser", None)
        if browser is not None:
            ensure_ready = getattr(browser, "ensure_ready", None)
            if callable(ensure_ready):
                ensure_ready()
        return {"messaging": Observation(Source.BROWSER, "messaging target", {
            "action": self.action.kind,
            "recipient": self.action.params.get("recipient", ""),
        })}

    def reference_plan(self, policy: Any) -> list[Action]:
        return [self.action]

    def verify_checkpoint(self, policy: Any, action: Action, trace: Any = None) -> VerificationResult | None:
        return None

    def verify_final(self, policy: Any, trace: Any = None) -> VerificationResult:
        try:
            from .skills.messaging import BrowserMessagingBackend
            from .skills.messaging.verifier import MessagingVerifier
            # Use the same task-scoped browser session that the approved action
            # will use.  Verification is fresh and never assumes click success.
            browser = getattr(self, "browser", None)
            if browser is None:
                return VerificationResult([
                    Check("fast_messaging", Verdict.UNKNOWN, {}, "messaging browser was not attached")
                ], label="fast messaging")
            verification = MessagingVerifier(BrowserMessagingBackend(browser)).verify(
                __import__("agent_control.skills.messaging.actions", fromlist=["MessagingAction"]).MessagingAction.from_core(self.action),
                None,
            )
            status = str(getattr(verification, "status", "UNKNOWN")).upper()
            verdict = {"PASS": Verdict.PASS, "FAIL": Verdict.FAIL}.get(status, Verdict.UNKNOWN)
            return VerificationResult([
                Check("fast_messaging", verdict, {"action": self.action.to_json()}, getattr(verification, "detail", ""))
            ], label="fast messaging")
        except Exception as exc:
            return VerificationResult([
                Check("fast_messaging", Verdict.UNKNOWN, {"action": self.action.to_json()}, f"messaging verification unavailable: {type(exc).__name__}: {exc}")
            ], label="fast messaging")

    def teardown(self, policy: Any) -> None:
        return None


class FastInteractionTask:
    """Task adapter used only for a routed fast browser action."""
    bucket = "fast_interaction"

    def __init__(self, request: str, browser: BrowserSkillAdapter, route: FastRoute) -> None:
        self.request = request
        self.goal = request
        self.task_id = f"fast-{abs(hash((request, time.time_ns()))) & 0xffffffff:08x}"
        self.browser = browser
        self.route = route
        self._first_observation: BrowserObservation | None = None
        self.action: Action | None = None

    def action_template(self) -> Action:
        """Return a target-free action used for the existing precondition gate."""
        return Action(
            kind=self.route.action_kind,
            params=dict(self.route.params),
            consequential=True,
            rationale=self.route.reason,
        )

    def resolve_current_action(self) -> Action:
        """Resolve a NEW action against the observation immediately before execution.

        The runner calls this only after its freshness/precondition observation.
        The returned BrowserTarget therefore belongs to that exact observation
        generation and is never reused after execution or another observation.
        """
        observation = getattr(self.browser, "_last_observation", None)
        if observation is None:
            raise ValueError("browser observation unavailable")
        action = self.route.resolve(self.browser, observation)
        self.action = action
        return action

    def setup(self, policy: Any) -> None:
        return None

    def observe(self, policy: Any, trace: Any = None) -> dict[str, Observation]:
        if self._first_observation is not None:
            observation = self._first_observation
            self._first_observation = None
        else:
            ensure_ready = getattr(self.browser, "ensure_ready", None)
            if callable(ensure_ready):
                ensure_ready()
            else:
                self.browser.observe()
            observation = getattr(self.browser, "_last_observation", None)
        if observation is None:
            return {"browser": Observation(Source.BROWSER, "browser.observe", {}, ok=False, error="browser observation unavailable")}

        if self.action is None or self._needs_target_resolution():
            try:
                self.action = self.route.resolve(self.browser, observation)
            except Exception as exc:
                self.action = None
                return {"browser": Observation(Source.BROWSER, "browser.observe", observation.raw, ok=False, error=str(exc))}

        obs = Observation(
            Source.BROWSER,
            "browser.observe",
            {"url": observation.url, "text": observation.text, "elements": len(observation.elements)},
        )
        if trace:
            obs = trace.observation(obs, purpose="fast interaction browser observation")
        return {"browser": obs}

    def _needs_target_resolution(self) -> bool:
        return self.route.target_query is not None or self.route.target_index is not None

    def reference_plan(self, policy: Any) -> list[Action]:
        return [self.action] if self.action is not None else []

    def verify_checkpoint(self, policy: Any, action: Action, trace: Any = None) -> VerificationResult | None:
        return None

    def verify_final(self, policy: Any, trace: Any = None) -> VerificationResult:
        """Use BrowserSkill's existing independent verifier for the final state."""
        if self.action is None:
            return VerificationResult(
                label=f"fast:{self.task_id}",
                checks=[Check(
                    name="fast_action_state",
                    verdict=Verdict.UNKNOWN,
                    reason="no executable fast action was produced",
                )],
            )
        try:
            from .skills.browser.actions import BrowserAction
            from .skills.browser.skill import BrowserSkill
            skill = BrowserSkill(self.browser)
            # Indexed "play the Nth video" actions have a semantic postcondition
            # that generic browser_click verification cannot infer: the selected
            # YouTube watch page must actually be playing. BrowserVerifier owns
            # that independent proof and performs fresh browser reads.
            if self.route.target_index is not None and re.fullmatch(rf"play\s+the\s+{_NUMBER}\s+video", self.request.strip().casefold()):
                verification = skill.verifier().verify_playback("")
                status = str(getattr(verification, "status", "UNKNOWN")).upper()
                verdict = {"PASS": Verdict.PASS, "FAIL": Verdict.FAIL, "UNKNOWN": Verdict.UNKNOWN}.get(status, Verdict.UNKNOWN)
                return VerificationResult(
                    label=f"fast:{self.task_id}",
                    checks=[Check(
                        name="youtube_playback",
                        verdict=verdict,
                        evidence={"action": self.action.to_json(), "detail": getattr(verification, "detail", "")},
                        reason=str(getattr(verification, "detail", "") or "YouTube playback evidence was not established"),
                    )],
                )
            # Verification is a fresh-state read and does not execute the
            # target. The action's BrowserTarget is intentionally generation-
            # bound and is expected to be stale after the click. Strip only
            # the wrapper here so the verifier can inspect the action kind and
            # postcondition without attempting to validate an old ref.
            verify_params = dict(self.action.params)
            target = verify_params.get("target")
            if isinstance(target, BrowserTarget):
                verify_params["target"] = target.ref
            skill_action = BrowserAction.from_json({
                "kind": self.action.kind,
                "params": verify_params,
                "rationale": self.action.rationale,
            })
            verification = skill.verifier().verify(skill_action, None)
        except Exception as exc:
            return VerificationResult(
                label=f"fast:{self.task_id}",
                checks=[Check(
                    name="fast_browser_verification",
                    verdict=Verdict.UNKNOWN,
                    evidence={"action": self.action.to_json()},
                    reason=f"existing browser verifier could not run: {type(exc).__name__}: {exc}",
                )],
            )
        status = str(getattr(verification, "status", "UNKNOWN")).upper()
        verdict = {"PASS": Verdict.PASS, "FAIL": Verdict.FAIL, "UNKNOWN": Verdict.UNKNOWN}.get(status, Verdict.UNKNOWN)
        return VerificationResult(
            label=f"fast:{self.task_id}",
            checks=[Check(
                name="fast_browser_verification",
                verdict=verdict,
                evidence={"action": self.action.to_json(), "detail": getattr(verification, "detail", "")},
                reason=str(getattr(verification, "detail", "") or "existing browser verifier returned no detail"),
            )],
        )

    def teardown(self, policy: Any) -> None:
        return None
