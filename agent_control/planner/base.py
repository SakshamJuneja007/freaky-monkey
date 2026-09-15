"""Planner contract.

The planner owns intent and decomposition. It owns nothing else: it cannot
observe, cannot execute, cannot verify, and cannot grant itself permission
(plan S1 key principle). It proposes semantic actions and claims completion; the
runtime decides what actually happens and whether it worked.

``PlannerStep.done`` is the planner's *claim* that the goal is met. Comparing
that claim against independent verification is precisely the false-success rate
in plan S19, so the claim is recorded and never acted on as truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..types import Action


@dataclass
class PlannerStep:
    """One turn of planner output."""

    actions: list[Action] = field(default_factory=list)

    #: Planner asserts the goal is already achieved. Evidence, not proof.
    done: bool = False

    reasoning: str = ""

    #: Set when the planner itself failed (bad JSON, API error, refusal).
    error: str | None = None

    #: Ordered executable actions are also the minimal sequential workflow representation.
    #: The runtime may persist them as workflow steps without changing the planner contract.

    #: Action kinds/items proposed by the planner that the runtime cannot execute.
    #: Kept structured so the runner can explain an empty executable plan.
    rejected: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "actions": [a.to_json() for a in self.actions],
            "done": self.done,
            "reasoning": self.reasoning[:2000],
            "error": self.error,
            "rejected": list(self.rejected),
        }


@dataclass
class Usage:
    """Call and token accounting for the metrics table (plan S19)."""

    calls: int = 0
    vision_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(
        self,
        *,
        prompt: int = 0,
        completion: int = 0,
        vision: bool = False,
    ) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion

        if vision:
            self.vision_calls += 1

    def cost_usd(
        self,
        price_in_per_m: float,
        price_out_per_m: float,
    ) -> float:
        return (
            self.prompt_tokens / 1_000_000 * price_in_per_m
            + self.completion_tokens / 1_000_000 * price_out_per_m
        )

    def to_json(self) -> dict:
        return {
            "calls": self.calls,
            "vision_calls": self.vision_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@runtime_checkable
class Planner(Protocol):
    """Anything the runner can plan with."""

    name: str
    usage: Usage

    def plan(
        self,
        goal: str,
        state: dict,
        history: list[dict],
    ) -> PlannerStep:
        """Propose the next actions given fresh state and what has happened so far."""
        ...


#: Action kinds the planner is told about. Kept in sync with os_tools.DISPATCH by
#: ``tests/test_planner_contract.py`` so the prompt can never drift from reality.
ALLOWED_ACTION_KINDS: tuple[str, ...] = (
    "launch_app",
    "open_url",
    "whatsapp_send_message",
    "whatsapp_search_contact",
    "gmail_send_email",
    "gmail_search_mail",
    "gmail_read_mail",
    "browser_open_url",
    "browser_search",
    "browser_get_current_page",
    "browser_list_tabs",
    "browser_open_new_tab",
    "browser_switch_tab",
    "browser_close_tab",
    "browser_go_back",
    "browser_go_forward",
    "browser_refresh",
    "browser_page_state",
    "browser_extract_text",
    "browser_click",
    "browser_type",
    "browser_press_key",
    "browser_scroll",
    "browser_scroll_to",
    "browser_select",
    "browser_upload_file",
    "browser_download_file",
    "browser_wait",
    "browser_borrow_tab",
    "browser_return_tab",
    "browser_play_song",
    "browser_apply_job",
    "create_dir",
    "write_file",
    "fetch_file",
    "open_file",

    # Read-only inspection actions.
    "list_directory",
    "read_text_file",
    "search_files",

    "create_venv",
    "install_requirements",
    "run_command",
)