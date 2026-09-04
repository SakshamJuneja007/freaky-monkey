"""Planner backends. Swappable by design (plan S24: model choice is a variable)."""

from __future__ import annotations

from .base import ALLOWED_ACTION_KINDS, Planner, PlannerStep, Usage
from .mock import MockPlanner
from .openai_compat import (
    LLMClient,
    LLMUnavailable,
    OpenAICompatPlanner,
    load_env,
)

__all__ = [
    "ALLOWED_ACTION_KINDS",
    "LLMClient",
    "LLMUnavailable",
    "MockPlanner",
    "OpenAICompatPlanner",
    "Planner",
    "PlannerStep",
    "Usage",
    "load_env",
    "build_planner",
]


def build_planner(kind: str, *, reference_plan=None, **kwargs):
    """Construct a planner by CLI name.

    ``mock`` needs a reference plan to replay; ``llm`` needs credentials in .env.
    """
    if kind == "mock":
        return MockPlanner(reference_plan=list(reference_plan or []), **kwargs)
    if kind == "llm":
        return OpenAICompatPlanner(client=LLMClient.from_env(), **kwargs)
    raise ValueError(f"unknown planner {kind!r}; expected 'mock' or 'llm'")
