"""The prompt and the executor must describe the same machine.

``base.ALLOWED_ACTION_KINDS`` claims in its own comment to be "kept in sync with
os_tools.DISPATCH by ``tests/test_planner_contract.py``". That file did not exist,
and the drift it was meant to prevent had already happened: ``open_file`` was in
``DISPATCH``, absent from the planner's schema, and therefore invisible to the LLM.
Asked to open a file, the planner reached for the only door it could see --
``run_command explorer`` -- which the executable allowlist denied, so the run ended
POLICY_BLOCKED with the file never opened.

Nothing was broken. Every layer did its job: the planner proposed, policy refused,
recovery declined to retry, and the report said so. The bug was that the planner had
been told a smaller machine than the one it was driving.

These tests are the promised enforcement. Three claims:

1. **The two lists are the same list.** Every executable kind is offered to the
   planner, and every offered kind is executable.
2. **Every offered kind is documented with its parameters.** A kind named in the
   schema with no shape is an invitation to guess.
3. **What the tasks actually need is in there.** The reference plans are the
   ground truth for "which actions does this system have to be able to emit".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_control import os_tools
from agent_control.planner.base import ALLOWED_ACTION_KINDS
from agent_control.planner.openai_compat import ACTION_SCHEMA, SYSTEM_PROMPT
from agent_control.policy import Policy
from benchmark.tasks import TASK_IDS, build_task


# ----------------------------------------------------------------------
# 1. The two lists are the same list
# ----------------------------------------------------------------------

def test_every_executable_action_is_offered_to_the_planner():
    """The direction that actually bit: an action the runtime can perform but the
    prompt never mentions is a capability the agent does not have."""
    missing = sorted(set(os_tools.DISPATCH) - set(ALLOWED_ACTION_KINDS))

    assert missing == [], (
        f"os_tools can execute {missing} but the planner is never told they exist; "
        "add them to ALLOWED_ACTION_KINDS and to ACTION_SCHEMA"
    )


def test_every_offered_action_can_actually_be_executed():
    """The other direction: offering a kind with no dispatch entry would make the
    planner emit actions that are discarded after the model was paid for them."""
    from agent_control.skills.builtin import build_builtin_registry
    policy = Policy(workspace=Path.cwd() / ".test-workspace", refuse_if_elevated=False)
    skill_actions = set(build_builtin_registry(policy).action_kinds())
    invented = sorted(set(ALLOWED_ACTION_KINDS) - (set(os_tools.DISPATCH) | skill_actions))

    assert invented == [], f"the planner is offered {invented}, which nothing executes"


def test_the_list_has_no_duplicates():
    assert len(set(ALLOWED_ACTION_KINDS)) == len(ALLOWED_ACTION_KINDS)


# ----------------------------------------------------------------------
# 2. Every offered kind is documented with its parameters
# ----------------------------------------------------------------------

def _schema_kinds() -> set[str]:
    """The kind at the start of each schema line."""
    return {
        line.split()[0]
        for line in ACTION_SCHEMA.splitlines()
        if line.strip() and not line.startswith(" ")
    }


def test_the_schema_documents_exactly_the_allowed_kinds():
    assert _schema_kinds() == set(ALLOWED_ACTION_KINDS)


@pytest.mark.parametrize("kind", ALLOWED_ACTION_KINDS)
def test_each_kind_is_shown_with_a_parameter_shape(kind: str):
    """A name with no shape beside it is a guess waiting to happen -- and a guessed
    parameter name fails at dispatch, after the planner call has been spent."""
    line = next(l for l in ACTION_SCHEMA.splitlines() if l.split()[0] == kind)
    shape = line[len(kind):].strip()

    assert shape.startswith("{") and shape.endswith("}")
    assert '":' in shape or '": ' in shape


def test_the_schema_reaches_the_prompt():
    """The schema is interpolated, not merely defined."""
    assert ACTION_SCHEMA in SYSTEM_PROMPT
    for kind in ALLOWED_ACTION_KINDS:
        assert kind in SYSTEM_PROMPT


def test_the_prompt_says_which_action_shows_a_file():
    """The observed failure was substituting a file manager for ``open_file``. The
    rule that prevents it has to be in the prompt, not only in this test."""
    assert "open_file" in SYSTEM_PROMPT
    assert "explorer" not in SYSTEM_PROMPT.lower(), (
        "name the action to use, not a program to avoid; the allowlist is the "
        "enforcement and it is not a list of two"
    )


# ----------------------------------------------------------------------
# 3. What the tasks actually need is in there
# ----------------------------------------------------------------------

@pytest.mark.parametrize("task_id", TASK_IDS)
def test_a_task_never_needs_an_action_the_planner_cannot_emit(task_id: str,
                                                              tmp_path: Path):
    """The reference plan is the ground truth for "what must be emittable". If a
    task's own solution uses a kind the planner is not offered, that task is
    unachievable with the real planner however well the mock one does."""
    target = tmp_path / "specimen.mp4"
    target.write_bytes(b"\x00" * 32)

    params = {"path": str(target)} if task_id == "open_named_file" else {}
    task = build_task(task_id, **params)
    policy = Policy(workspace=tmp_path / "ws", readable_roots=(target,))

    for action in task.reference_plan(policy):
        assert action.kind in ALLOWED_ACTION_KINDS, (
            f"{task_id} needs {action.kind!r}, which the planner is never told about"
        )
