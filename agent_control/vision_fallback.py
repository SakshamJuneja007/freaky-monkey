"""Vision fallback and vision actuation (plan S5, S7).

Two consumers, one implementation:

* the structured-first path uses this only when structured state cannot answer the
  question or perform the action (plan S3, S5 last bullet);
* the vision-only baseline (plan S18) uses it for *everything*, which is the point
  of the comparison.

Grounding caveat, stated up front: synthesising a click from a VLM's pixel guess
is only as good as that model's grounding. A weak baseline number may reflect the
model's grounding rather than the screenshot-first approach in general. The
harness records ``vision_calls`` and the VLM identity so this stays visible when
results are read (plan S27's "honest account of limitations").
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .platform_window import get_backend
from .planner.base import Usage
from .planner.openai_compat import LLMClient
from .types import Action, ActionResult, FailureClass, Observation, Source

#: Input kinds the vision path may synthesise. Deliberately tiny.
VISION_ACTION_KINDS = ("click", "type_text", "press_keys", "wait")

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

VISION_SYSTEM_PROMPT = """\
You are driving a desktop computer by looking at screenshots. You have no access \
to the filesystem, process table, or any structured state. Your only inputs are \
the screenshot and your own action history; your only outputs are mouse and \
keyboard events.

Coordinates are pixels in the screenshot you were given, origin at top-left. The \
image dimensions are stated in the user message.

Reply with a single JSON object, no prose and no code fences:
{"reasoning": "<brief>", "done": <bool>, "action": {"kind": "<kind>", ...}}

Action shapes:
  {"kind": "click", "x": <int>, "y": <int>}
  {"kind": "type_text", "text": "<text>"}
  {"kind": "press_keys", "combo": "ctrl+s"}
  {"kind": "wait", "seconds": <number>}

Set "done": true when the screenshot shows the goal achieved. Emit exactly one \
action per turn, or none when done."""


def screenshot() -> Observation:
    """Capture the screen. Unavailable capture is ok=False, so checks go UNKNOWN."""
    backend = get_backend()
    started = time.time()
    png = backend.screenshot_png()
    if png is None:
        return Observation(
            source=Source.VISION, query="screenshot()", value=None, observed_at=started,
            ok=False, error=f"no screenshot capability on backend {backend.name!r}",
        )
    width = height = None
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(png)) as image:
            width, height = image.size
    except Exception:
        pass
    return Observation(
        source=Source.VISION, query="screenshot()", observed_at=started,
        value={"bytes": len(png), "width": width, "height": height, "png": png},
    )


@dataclass
class VisionStep:
    """One turn of vision-driven output."""

    action: dict[str, Any] | None = None
    done: bool = False
    reasoning: str = ""
    error: str | None = None

    def to_json(self) -> dict:
        return {"action": self.action, "done": self.done,
                "reasoning": self.reasoning[:1000], "error": self.error}


@dataclass
class VisionActor:
    """Screenshot -> model -> mouse/keyboard -> screenshot (plan S18 baseline loop)."""

    client: LLMClient
    system_prompt: str = VISION_SYSTEM_PROMPT
    max_history: int = 6
    name: str = "vision"

    def __post_init__(self) -> None:
        self.name = f"vision:{self.client.model}"

    @property
    def usage(self) -> Usage:
        return self.client.usage

    def propose(self, goal: str, shot: Observation, history: list[dict]) -> VisionStep:
        if not shot.ok:
            return VisionStep(error=f"no screenshot: {shot.error}")
        png: bytes = shot.value["png"]
        data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        context = {
            "goal": goal,
            "image_width": shot.value.get("width"),
            "image_height": shot.value.get("height"),
            "history": history[-self.max_history:],
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": json.dumps(context, indent=2, default=str)},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]
        try:
            text, _, truncated = self.client.chat_checked(messages, json_mode=True, vision=True)
        except Exception as exc:
            return VisionStep(error=f"vision transport error: {type(exc).__name__}: {exc}")
        if truncated:
            return VisionStep(error=f"vision {truncated}")
        return _parse_vision_step(text)

    def apply(self, step: VisionStep) -> ActionResult:
        """Synthesise the proposed input event. No policy gate applies here."""
        return apply_vision_action(step)


def apply_vision_action(step: VisionStep) -> ActionResult:
    """Turn a proposed vision step into real input. Module-level: two callers.

    Synthetic input is unconstrainable by a path allowlist -- keystrokes go
    wherever focus is. That is exactly why the vision path is the fallback of
    last resort in the structured design, and why the baseline condition is
    the riskier one to run. It is confined by running against a disposable
    workspace, not by this function.
    """
    spec = step.action or {}
    kind = spec.get("kind")
    action = Action(kind=f"vision_{kind}", params=spec, consequential=True)
    started = time.time()
    backend = get_backend()

    if kind not in VISION_ACTION_KINDS:
        return ActionResult(action=action, ok=False,
                            error=f"unknown vision action {kind!r}",
                            failure_class=FailureClass.PRECONDITION_FAILED)
    try:
        if kind == "click":
            ok = backend.click(int(spec["x"]), int(spec["y"]))
        elif kind == "type_text":
            ok = backend.type_text(str(spec["text"]))
        elif kind == "press_keys":
            ok = backend.press_keys(str(spec["combo"]))
        else:
            seconds = min(float(spec.get("seconds", 1.0)), 10.0)
            time.sleep(seconds)
            ok = True
    except (KeyError, TypeError, ValueError) as exc:
        return ActionResult(action=action, ok=False,
                            error=f"malformed vision action: {exc}",
                            failure_class=FailureClass.PRECONDITION_FAILED,
                            duration_s=time.time() - started)
    return ActionResult(
        action=action, ok=bool(ok), detail={"backend": backend.name, "spec": spec},
        error=None if ok else f"{kind} not supported by backend {backend.name!r}",
        failure_class=None if ok else FailureClass.ENVIRONMENT,
        duration_s=time.time() - started,
    )


def _parse_vision_step(text: str) -> VisionStep:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", stripped).strip()
    candidates = [stripped]
    match = _JSON_BLOCK.search(stripped)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        action = payload.get("action")
        return VisionStep(
            action=action if isinstance(action, dict) else None,
            done=bool(payload.get("done", False)),
            reasoning=str(payload.get("reasoning", "")),
        )
    return VisionStep(error=f"unparseable vision output: {text[:300]!r}")
