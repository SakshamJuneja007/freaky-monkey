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

    def to_json(self) -> dict:
        return {
            "actions": [a.to_json() for a in self.actions],
            "done": self.done,
            "reasoning": self.reasoning[:2000],
            "error": self.error,
        }


@dataclass
class Usage:
    """Call and token accounting for the metrics table (plan S19)."""

    calls: int = 0
    vision_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, *, prompt: int = 0, completion: int = 0, vision: bool = False) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        if vision:
            self.vision_calls += 1

    def cost_usd(self, price_in_per_m: float, price_out_per_m: float) -> float:
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
    "create_dir",
    "write_file",
    "fetch_file",
    "open_file",
    "create_venv",
    "install_requirements",
    "run_command",
)
