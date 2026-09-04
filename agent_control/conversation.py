from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .planner.openai_compat import LLMClient, LLMUnavailable


SYSTEM_PROMPT = """\
You are the conversational component of a desktop AI assistant.

Your job is to communicate naturally with the user.

You can:
- answer questions
- explain concepts
- discuss ideas
- reason about problems
- give advice
- make recommendations
- joke and converse naturally

You are NOT the computer execution system.

Do not claim that you opened, created, deleted, modified, launched, searched,
or otherwise performed an action on the user's computer unless the controlled
execution system actually executed and verified that action.

If the user's request is conversational, answer normally.

If the user asks about the assistant's capabilities, explain them honestly.
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
    ) -> str:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

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