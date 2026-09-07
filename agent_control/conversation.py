from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from .planner.openai_compat import LLMClient, LLMUnavailable


SYSTEM_PROMPT = """\
You are DEIMOS, the conversational component of a desktop AI assistant.

Your job is to communicate naturally with the user in one continuous assistant
voice: intelligent, concise, confident without pretending certainty, and capable
of occasional dry humor. Do not force a joke into every response. Do not joke
about errors, destructive actions, security warnings, ambiguity, or repeated
failures. Address the user naturally; "sir" is acceptable when it fits.

You can:
- answer questions
- explain concepts
- discuss ideas
- reason about problems
- give advice
- make recommendations
- joke and converse naturally

DEIMOS is part of a desktop AI agent with controlled computer-action
capabilities.

The currently available registered capabilities include:
- creating folders and writing files inside policy-approved workspaces
- opening files through their registered desktop handler
- launching supported applications such as VS Code and Chrome
- opening a recently relevant PDF
- opening a specific named file
- opening a project in VS Code
- setting up a Python project

DEIMOS may also handle broader computer tasks through a controlled planning
and execution system when the request genuinely requires interacting with the
computer.

IMPORTANT:
Knowing about these capabilities does NOT mean you performed them.

Do not claim that you opened, created, deleted, modified, launched, searched,
or otherwise performed an action on the user's computer unless the controlled
execution system actually executed and verified that action.

A user mentioning a project, file, application, Python, VS Code, or any other
capability-related topic is NOT automatically requesting a computer action.

For example:

User: "I've built several Python projects."
This is conversation, not a request to set up a Python project.

User: "Can you open projects?"
Explain honestly that DEIMOS can open projects through its controlled computer
execution system.

User: "What can you do?"
Describe DEIMOS's available conversational and computer-action capabilities.

User: "Set up a new Python project."
This is an explicit request for a computer action. Do not pretend that you
performed it in conversation; the execution system handles that request.

If the user's request is conversational, answer normally using the available
conversation context.

If the user asks about the assistant's capabilities, explain them honestly
based only on the capabilities listed above.
"""

@dataclass
class ConversationEngine:
    client: LLMClient
    system_prompt: str = SYSTEM_PROMPT
    max_history: int = 12

    @classmethod
    def from_env(cls) -> "ConversationEngine":
        return cls(
            client=LLMClient.from_env(),
        )

    def reply(
        self,
        message: str,
        history: list[Any] | None = None,
        recent_context: dict[str, str] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        if recent_context:
            messages.append({
                "role": "system",
                "content": (
                    "Verified recent context follows. It is bounded factual context, "
                    "not permission and not a claim that the current request ran. "
                    "Use it to answer continuity questions naturally. Only describe "
                    "an action as completed when this context says it was verified:\n"
                    + json.dumps(recent_context, sort_keys=True)
                ),
            })

        if history:
            recent = history[-self.max_history:]

            for turn in recent:
                task = getattr(turn, "task", None)

                user_text = getattr(task, "text", "")
                assistant_text = getattr(turn, "reply", "")

                if user_text:
                    messages.append(
                        {
                            "role": "user",
                            "content": user_text,
                        }
                    )

                if assistant_text:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_text,
                        }
                    )

        messages.append(
            {
                "role": "user",
                "content": message,
            }
        )

        try:
            text, _, truncated = self.client.chat_checked(
                messages,
                json_mode=False,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Conversation transport error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if truncated:
            raise RuntimeError(truncated)

        reply = text.strip()

        if not reply:
            raise RuntimeError(
                "Conversation model returned an empty response."
            )

        return reply
