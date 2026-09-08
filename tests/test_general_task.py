import json
from pathlib import Path

import pytest

from agent_control import api
from agent_control import os_tools
from agent_control import session as sess
from agent_control.api import AgentResult, TaskStatus
from agent_control.general_task import GeneralTask
from agent_control.memory import Entry, FileMemory
from agent_control.planner import openai_compat
from agent_control.planner.base import PlannerStep, Usage
from agent_control.policy import Policy
from agent_control.response import Narrator
from agent_control.session import Session
from agent_control.types import Action, Verdict


def test_real_three_turn_file_context_uses_exact_verified_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise the real Session/API/Policy/executor/verifier/ledger path.

    Only the external planner transport is scripted. Filesystem actions and
    independent verification are the production implementations.

    Diagnostic output is intentionally verbose so a failure identifies
    whether the problem occurs on the first, second, or third conversational
    turn.
    """

    workspace = (tmp_path / "conversation").resolve()

    emitted: list[Action] = []
    executions: list[dict] = []

    class ScriptedPlanner:
        name = "scripted-context"

        def __init__(self) -> None:
            self.usage = Usage()

        def plan(self, goal, state, history):
            self.usage.add(prompt=0, completion=0)

            print("\n" + "=" * 80)
            print("SCRIPTED PLANNER CALLED")
            print("GOAL:", repr(goal))
            print("HISTORY:", repr(history))

            recent_context = state.get("recent_context")
            path_permissions = state.get("path_permissions")

            print(
                "RECENT CONTEXT:",
                repr(recent_context),
            )

            print(
                "PATH PERMISSIONS:",
                repr(path_permissions),
            )

            # Each conversational turn has its own goal.
            #
            # IMPORTANT:
            # `done=True` means the planner claims that the goal represented
            # by this planner step has been achieved. The runtime still
            # independently executes and verifies the action.
            #
            # Without done=True, the task runner can continue asking the
            # planner for another step after the action executes. Because
            # this planner is scripted to emit the same action for the same
            # goal, that causes repeated execution until the step ceiling.

            if goal == "create folder TestContext":
                write_root = Path(
                    path_permissions["write_root"]
                )

                action = Action(
                    "create_dir",
                    {
                        "path": str(
                            write_root / "TestContext"
                        )
                    },
                )

            elif goal == "create notes.txt in that folder":
                directory = Path(
                    recent_context["last_verified_directory"]
                )

                action = Action(
                    "write_file",
                    {
                        "path": str(
                            directory / "notes.txt"
                        ),
                        "content": "",
                    },
                )

            elif goal == 'write "updated content" to that file':
                target = Path(
                    recent_context["last_verified_file"]
                )

                action = Action(
                    "write_file",
                    {
                        "path": str(target),
                        "content": "updated content",
                        # The file already exists because it was created
                        # during the previous conversational turn.
                        "overwrite": True,
                    },
                )

            else:
                raise AssertionError(
                    f"Unexpected goal: {goal!r}"
                )

            emitted.append(action)

            print("EMITTED ACTION:")
            print("  kind:", action.kind)
            print("  params:", dict(action.params))

            # IMPORTANT FIX:
            # Tell the task runner that this planner step is complete.
            return PlannerStep(
                actions=[action],
                done=True,
            )

    # ------------------------------------------------------------------
    # Replace ONLY the external planner transport.
    # ------------------------------------------------------------------

    monkeypatch.setattr(
        openai_compat.LLMClient,
        "from_env",
        classmethod(
            lambda cls, **kwargs: object()
        ),
    )

    monkeypatch.setattr(
        openai_compat,
        "OpenAICompatPlanner",
        lambda client: ScriptedPlanner(),
    )

    # ------------------------------------------------------------------
    # Keep the real filesystem executor.
    # Wrap it only for diagnostics.
    # ------------------------------------------------------------------

    real_execute = os_tools.execute

    def execute_with_audit(policy, action):
        print("\n" + "-" * 80)
        print("EXECUTE CALLED")
        print("WORKSPACE:", policy.workspace)
        print("ACTION KIND:", action.kind)
        print("RAW PARAMS:", dict(action.params))

        result = real_execute(policy, action)

        print("EXECUTION RESULT:")
        print("  result:", result)

        print(
            "  detail:",
            repr(result.detail),
        )

        executions.append(
            {
                "workspace": policy.workspace,
                "kind": action.kind,
                "raw_params": dict(action.params),
                "resolved_path": result.detail.get("path"),
            }
        )

        return result

    monkeypatch.setattr(
        os_tools,
        "execute",
        execute_with_audit,
    )

    # ------------------------------------------------------------------
    # Real Session.
    # ------------------------------------------------------------------

    printed: list[str] = []

    chat = Session(
        narrator=Narrator(
            speaker=None,
            write=printed.append,
        ),
        planner="llm",
        workspace=workspace,
        keep_workspace=True,
        debug=True,
    )

    # ------------------------------------------------------------------
    # FIRST TURN
    # ------------------------------------------------------------------

    print("\n" + "#" * 80)
    print("SUBMITTING FIRST TURN")

    first = chat.submit(
        "create folder TestContext"
    )

    print("\nFIRST TURN COMPLETE")
    print("FIRST OK:", first.ok)
    print("FIRST EXECUTED:", first.executed)
    print("FIRST RESULT:", repr(first.result))

    print(
        "RECENT DIRECTORY:",
        repr(
            chat.recent_context.last_verified_directory
        ),
    )

    print(
        "RECENT FILE:",
        repr(
            chat.recent_context.last_verified_file
        ),
    )

    directory_context = (
        chat.recent_context.last_verified_directory
    )

    # ------------------------------------------------------------------
    # SECOND TURN
    # ------------------------------------------------------------------

    print("\n" + "#" * 80)
    print("SUBMITTING SECOND TURN")

    second = chat.submit(
        "create notes.txt in that folder"
    )

    print("\nSECOND TURN COMPLETE")
    print("SECOND OK:", second.ok)
    print("SECOND EXECUTED:", second.executed)
    print("SECOND RESULT:", repr(second.result))

    print(
        "RECENT DIRECTORY:",
        repr(
            chat.recent_context.last_verified_directory
        ),
    )

    print(
        "RECENT FILE:",
        repr(
            chat.recent_context.last_verified_file
        ),
    )

    file_context = (
        chat.recent_context.last_verified_file
    )

    # ------------------------------------------------------------------
    # THIRD TURN
    # ------------------------------------------------------------------

    print("\n" + "#" * 80)
    print("SUBMITTING THIRD TURN")

    third = chat.submit(
        'write "updated content" to that file'
    )

    print("\nTHIRD TURN COMPLETE")
    print("THIRD OK:", third.ok)
    print("THIRD EXECUTED:", third.executed)
    print("THIRD RESULT:", repr(third.result))

    print(
        "RECENT DIRECTORY:",
        repr(
            chat.recent_context.last_verified_directory
        ),
    )

    print(
        "RECENT FILE:",
        repr(
            chat.recent_context.last_verified_file
        ),
    )

    # ------------------------------------------------------------------
    # EXPECTED PATHS
    # ------------------------------------------------------------------

    expected_dir = (
        workspace / "TestContext"
    ).resolve()

    expected_file = (
        expected_dir / "notes.txt"
    ).resolve()

    # ------------------------------------------------------------------
    # FULL DIAGNOSTICS
    # ------------------------------------------------------------------

    for name, turn in (
        ("FIRST", first),
        ("SECOND", second),
        ("THIRD", third),
    ):
        print("\n" + "=" * 80)
        print(f"DIAGNOSTIC: {name} TURN")
        print("=" * 80)

        print("turn.ok:", turn.ok)

        print(
            "turn.executed:",
            turn.executed,
        )

        print(
            "turn.result:",
            repr(turn.result),
        )

        if turn.result is None:
            print("RESULT IS NONE")
            continue

        print(
            "result.status:",
            repr(turn.result.status),
        )

        print(
            "result.verified:",
            repr(turn.result.verified),
        )

        print(
            "result.detail:",
            repr(turn.result.detail),
        )

        outcome = getattr(
            turn.result,
            "outcome",
            None,
        )

        if outcome is None:
            print("OUTCOME IS NONE")
            continue

        print(
            "outcome.verified:",
            repr(outcome.verified),
        )

        print(
            "outcome.aborted_reason:",
            repr(
                getattr(
                    outcome,
                    "aborted_reason",
                    None,
                )
            ),
        )

        final = getattr(
            outcome,
            "final",
            None,
        )

        if final is None:
            print(
                "FINAL VERIFICATION IS NONE"
            )
            continue

        print("FINAL CHECKS:")

        for check in final.checks:
            evidence = getattr(
                check,
                "evidence",
                None,
            )

            print(
                "  CHECK:",
                {
                    "name": check.name,
                    "verdict": str(
                        check.verdict
                    ),
                    "detail": getattr(
                        check,
                        "detail",
                        None,
                    ),
                    "evidence": evidence,
                },
            )

    # ------------------------------------------------------------------
    # EXECUTION DIAGNOSTICS
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ALL EXECUTIONS")
    print("=" * 80)

    for index, execution in enumerate(
        executions,
        start=1,
    ):
        print(
            f"EXECUTION {index}:"
        )
        print(execution)

    # ------------------------------------------------------------------
    # ACTION DIAGNOSTICS
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EMITTED ACTIONS")
    print("=" * 80)

    for index, action in enumerate(
        emitted,
        start=1,
    ):
        print(
            index,
            action.kind,
            dict(action.params),
        )

    # ------------------------------------------------------------------
    # FINAL CONTEXT
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL RECENT CONTEXT")
    print("=" * 80)

    print(
        "last_verified_directory:",
        chat.recent_context.last_verified_directory,
    )

    print(
        "last_verified_file:",
        chat.recent_context.last_verified_file,
    )

    # ------------------------------------------------------------------
    # EXPECTED PATH DIAGNOSTICS
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXPECTED PATHS")
    print("=" * 80)

    print(
        "workspace:",
        workspace,
    )

    print(
        "expected_dir:",
        expected_dir,
    )

    print(
        "expected_file:",
        expected_file,
    )

    print(
        "expected_dir exists:",
        expected_dir.exists(),
    )

    print(
        "expected_dir is_dir:",
        expected_dir.is_dir(),
    )

    print(
        "expected_file exists:",
        expected_file.exists(),
    )

    print(
        "expected_file is_file:",
        expected_file.is_file(),
    )

    if expected_file.exists():
        print(
            "expected_file content:",
            repr(
                expected_file.read_text(
                    encoding="utf-8"
                )
            ),
        )

    # ------------------------------------------------------------------
    # ORIGINAL ASSERTIONS
    # ------------------------------------------------------------------

    assert all(
        turn.ok
        for turn in (
            first,
            second,
            third,
        )
    ), (
        "One or more conversational turns failed. "
        "See diagnostics above."
    )

    # Exactly one planner action per conversational turn.
    assert [
        action.kind
        for action in emitted
    ] == [
        "create_dir",
        "write_file",
        "write_file",
    ]

    assert (
        len(emitted)
        == 3
    )

    assert (
        emitted[0].kind
        == "create_dir"
    )

    assert (
        emitted[1].kind
        == "write_file"
    )

    assert (
        emitted[2].kind
        == "write_file"
    )

    assert (
        emitted[1].kind
        != "create_dir"
    )

    assert (
        Path(
            emitted[1].params["path"]
        ).resolve()
        == expected_file
    )

    assert (
        Path(
            emitted[2].params["path"]
        ).resolve()
        == expected_file
    )

    assert (
        emitted[2].params["overwrite"]
        is True
    )

    assert {
        entry["workspace"]
        for entry in executions
    } == {
        workspace
    }

    assert (
        len(executions)
        == 3
    )

    assert (
        Path(
            executions[0]["resolved_path"]
        ).resolve()
        == expected_dir
    )

    assert (
        Path(
            executions[1]["resolved_path"]
        ).resolve()
        == expected_file
    )

    assert (
        Path(
            executions[2]["resolved_path"]
        ).resolve()
        == expected_file
    )

    assert (
        directory_context
        == expected_dir
    )

    assert (
        file_context
        == expected_file
    )

    assert (
        chat.recent_context
        .last_verified_directory
        == expected_dir
    )

    assert (
        chat.recent_context
        .last_verified_file
        == expected_file
    )

    assert expected_dir.is_dir()

    assert expected_file.exists()

    assert expected_file.is_file()

    assert not (
        workspace / "notes.txt"
    ).is_dir()

    assert (
        expected_file.read_text(
            encoding="utf-8"
        )
        == "updated content"
    )

    # ------------------------------------------------------------------
    # VERIFIER PATH ASSERTIONS
    # ------------------------------------------------------------------

    expected_verifier_paths = [
        expected_dir,
        expected_file,
        expected_file,
    ]

    for turn, expected in zip(
        (
            first,
            second,
            third,
        ),
        expected_verifier_paths,
    ):
        assert (
            turn.result is not None
        )

        assert (
            turn.result.outcome
            is not None
        )

        assert (
            turn.result.outcome.final
            is not None
        )

        evidence_paths: set[Path] = set()

        for check in (
            turn.result
            .outcome
            .final
            .checks
        ):
            evidence = getattr(
                check,
                "evidence",
                None,
            )

            if not isinstance(
                evidence,
                dict,
            ):
                continue

            if "path" not in evidence:
                continue

            evidence_paths.add(
                Path(
                    evidence["path"]
                ).resolve()
            )

        assert evidence_paths == {
            expected
        }

        trace_file = Path(
            turn.result.trace_file
        )

        assert trace_file.exists(), (
            f"Trace file does not exist: "
            f"{trace_file}"
        )

        trace_rows = [
            json.loads(line)
            for line in trace_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        recorded = [
            row["effect"]
            for row in trace_rows
            if row.get("event")
            == "effect_recorded"
        ]

        assert len(recorded) == 1, (
            f"Expected exactly one effect_recorded "
            f"event, got {len(recorded)}"
        )

        assert (
            Path(
                recorded[0]["target"]
            ).resolve()
            == expected
        )

    # ------------------------------------------------------------------
    # Planner completion diagnostics.
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PLANNER COMPLETION CHECK")
    print("=" * 80)

    print(
        "Exactly three planner actions were emitted:",
        len(emitted) == 3,
    )

    assert len(emitted) == 3, (
        "Planner emitted more than one action per conversational turn. "
        "This usually means PlannerStep(done=True) was not honored."
    )

    # ------------------------------------------------------------------
    # Narrator/debug output.
    #
    # Do not require exact strings unless Session's public narrator
    # contract guarantees them. The planner/execution diagnostics above
    # already prove the actual context flow.
    # ------------------------------------------------------------------

    debug_output = "\n".join(printed)

    print("\n" + "=" * 80)
    print("NARRATOR OUTPUT")
    print("=" * 80)

    print(debug_output)

    # Intentionally no exact narrator string assertions.