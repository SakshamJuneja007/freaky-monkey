from __future__ import annotations

from typing import Any

from ...types import Action, ActionResult, FailureClass
from ..base import Skill, SkillAction, SkillExecutor, SkillInfo, SkillVerifier
from ..manifest import SkillManifest
from ..security import Capability
from .actions import MESSAGING_ACTION_KINDS, MessagingAction
from .backend import MessagingBackend
from .verifier import MessagingVerifier


class MessagingExecutor:
    def __init__(self, backend: MessagingBackend) -> None:
        self._backend = backend

    def execute(self, action: MessagingAction) -> Any:
        p = action.params
        try:
            if action.kind.value == "whatsapp_send_message":
                detail = self._backend.send_whatsapp(str(p["recipient"]), str(p["message"]))
            elif action.kind.value == "whatsapp_search_contact":
                detail = self._backend.search_whatsapp(str(p["query"]))
            elif action.kind.value == "gmail_send_email":
                detail = self._backend.send_gmail(
                    str(p["recipient"]), str(p["subject"]), str(p["body"]),
                    str(p.get("cc", "")), str(p.get("bcc", "")),
                )
            elif action.kind.value == "gmail_search_mail":
                detail = self._backend.search_gmail(str(p["query"]))
            elif action.kind.value == "gmail_read_mail":
                detail = self._backend.read_gmail(str(p["query"]))
            else:
                raise ValueError(f"unsupported messaging action: {action.kind.value}")
            return ActionResult(action=Action(kind=action.kind.value, params=dict(action.params)), ok=True, detail=detail)
        except Exception as exc:
            return ActionResult(
                action=Action(kind=action.kind.value, params=dict(action.params)),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                failure_class=FailureClass.ACTION_FAILED,
            )


class MessagingSkill(Skill):
    _INFO = SkillInfo(
        name="messaging",
        version="1.0.0",
        description="WhatsApp and Gmail messaging through the isolated DEIMOS browser profile.",
        actions=tuple(SkillAction(kind=k, description=k.replace("_", " ")) for k in sorted(MESSAGING_ACTION_KINDS)),
        manifest=SkillManifest(
            capabilities=frozenset({Capability.BROWSER, Capability.NETWORK}),
            side_effecting=True,
        ),
    )

    def __init__(self, backend: MessagingBackend) -> None:
        self._backend = backend
        self._executor = MessagingExecutor(backend)
        self._verifier = MessagingVerifier(backend)

    @property
    def info(self) -> SkillInfo:
        return self._INFO

    def executor(self) -> SkillExecutor:
        return self._executor

    def verifier(self) -> SkillVerifier:
        return self._verifier

    def adapt_action(self, action: Any) -> MessagingAction:
        if not isinstance(action, Action):
            raise TypeError("messaging skill expects a core Action")
        if not self.supports(action.kind):
            raise ValueError(f"messaging skill does not support {action.kind!r}")
        return MessagingAction.from_core(action)
