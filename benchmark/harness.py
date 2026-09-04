"""The V1 experiment (plan S31): tasks x conditions x trials -> one results table.

Plan S31 asks for five OS tasks, the structured-first path, the vision-only
baseline, hand-written verifiers, trace logging, bounded recovery, at least three
trials per task per condition, and a first results table. This module runs that
and writes the table.

Four properties are deliberate.

*One code path per condition.* Every structured condition is the same
``run_task`` call with different ``RunConfig`` flags -- ``recovery_enabled`` for
plan S18's recovery ablation, ``fresh_precondition`` for hypothesis H2. An
ablation implemented as a second function would be measuring the second
function (plan S23).

*A fresh workspace and a fresh recovery budget per trial.* ``RecoveryBudget`` is
consumed in place, so one shared ``RunConfig`` would hand trial 2 an
already-spent budget and make recovery look unavailable. Workspaces are
per-trial directories so trial 2 cannot inherit trial 1's artefacts and pass
without acting -- the inflated result plan S22 names as a kill condition.

*Model-reported and verified success are never merged.* Every row carries both,
and the false-success rate between them is the headline number (plan S30).

*UNKNOWN gets its own column.* Plan S8 says UNKNOWN is not success; a table that
folds it into either bucket has already lost that.

Cost is *estimated* from token counts times a price passed on the command line,
because the provider is an experimental variable (plan S24). With no price given
the column reads 0.0 and the header says so. With the mock planner there are no
tokens to price at all, and every row of such a run is marked ``synthetic``:
those are plumbing checks, not results.

Usage::

    python -m benchmark.harness --planner mock --trials 3
    python -m benchmark.harness --planner llm --trials 3 \\
        --conditions structured_hybrid,no_recovery,vision_only \\
        --price-in 0.15 --price-out 0.60
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_control.api import readable_roots_for_task
from agent_control.planner import LLMUnavailable, MockPlanner
from agent_control.planner.openai_compat import LLMClient, OpenAICompatPlanner
from agent_control.policy import Policy
from agent_control.recovery import RecoveryBudget
from agent_control.runner import RunConfig, RunOutcome, run_task
from agent_control.trace import DEFAULT_TRACE_DIR, Trace
from agent_control.types import Verdict
from agent_control.vision_fallback import VisionActor

from . import inject
from .baselines.vision_only import ScriptedVisionActor, run_vision_task

#: The *benchmark* set, not every registered task. Parameterized interactive tasks
#: (``open_named_file``) are excluded on purpose: they have no fixed starting state,
#: so ``--tasks all`` must not reach them.
from .tasks import BENCHMARK_IDS as TASK_IDS
from .tasks import build_task

RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: 128 + SIGINT. An interrupted grid is incomplete, not failed, and a CI job that
#: read exit 1 would file it as the experiment having gone wrong.
EXIT_INTERRUPTED = 130


@dataclass(frozen=True)
class Condition:
    """One experimental arm. Flags, not code paths."""

    name: str
    #: ``structured`` -> ``runner.run_task``; ``vision`` -> the plan S18 baseline.
    kind: str
    fresh_precondition: bool = True
    recovery_enabled: bool = True
    #: Plan S20: mutate state between plan and act.
    adversarial: bool = False
    #: One action per planner turn. Required for adversarial arms, which need a
    #: mid-run window to interfere with.
    incremental_plan: bool = False
    note: str = ""

    def config(self, *, max_steps: int) -> RunConfig:
        """A fresh config -- and so a fresh recovery budget -- for one trial."""
        return RunConfig(
            condition=self.name, max_steps=max_steps,
            fresh_precondition=self.fresh_precondition,
            recovery_enabled=self.recovery_enabled,
            budget=RecoveryBudget(),
        )

CONDITIONS: dict[str, Condition] = {
    "structured_hybrid": Condition(
        name="structured_hybrid", kind="structured",
        note="the experimental condition: structured state first, fresh precondition, bounded recovery",
    ),
    "no_recovery": Condition(
        name="no_recovery", kind="structured", recovery_enabled=False,
        note="plan S18 ablation: every classified failure aborts immediately (H3)",
    ),
    "no_fresh_state": Condition(
        name="no_fresh_state", kind="structured", fresh_precondition=False,
        note="H2 ablation: act on the reading taken before planning, however old",
    ),
    "vision_only": Condition(
        name="vision_only", kind="vision",
        note="plan S18 baseline: screenshots in, synthetic mouse/keyboard out, same model",
    ),
    "adversarial": Condition(
        name="adversarial", kind="structured", adversarial=True, incremental_plan=True,
        note="plan S20: an external controller changes the machine between plan and act",
    ),
    "adversarial_no_fresh": Condition(
        name="adversarial_no_fresh", kind="structured", adversarial=True,
        incremental_plan=True, fresh_precondition=False,
        note="plan S20 with the H2 check switched off: the same interference, unnoticed",
    ),
}

#: Plan S18's default comparison set. The adversarial and H2 arms are opt-in
#: because they answer separate questions and cost separate wall-clock.
DEFAULT_CONDITIONS = ("structured_hybrid", "no_recovery", "vision_only")

#: Where an adversary can actually change something the task is watching:
#: ``task_id -> (injections, planner turn to fire after)``.
#:
#: Only two Group A tasks qualify. The other three are single-action tasks whose
#: artefacts do not exist until that one action runs, so there is nothing in the
#: observed state for an adversary to move. They are reported as not-applicable
#: rather than quietly counted as passes.
ADVERSARIAL_PLAN: dict[str, tuple[list[inject.Injection], int]] = {
    "install_dependencies": (
        [inject.Injection(kind="delete_path", target="project/.venv")],
        1,  # after create_venv verified, before install_requirements
    ),
    "full_project_setup": (
        [inject.Injection(kind="delete_path", target="project/.venv")],
        4,  # same window, mid-way through a six-action plan
    ),
}

def _trial_workspace(root: Path, condition: str, task_id: str, trial: int) -> Path:
    """A directory of its own per trial, wiped first if a previous run left one.

    Plan S10 wants a disposable workspace; plan S22 wants trials that cannot
    inherit each other's artefacts. One path satisfies both.
    """
    path = root / condition / task_id / f"t{trial}"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _structured_planner(kind: str, task: Any, policy: Policy, condition: Condition,
                        client: LLMClient | None):
    if kind == "mock":
        return MockPlanner(reference_plan=task.reference_plan(policy),
                           incremental=condition.incremental_plan)
    return OpenAICompatPlanner(client=client)


def _vision_actor(kind: str, client: LLMClient | None):
    if kind == "mock":
        # Empty script: claims completion having emitted no input at all. That is
        # a plumbing check *and* a live test that the false-success metric fires.
        return ScriptedVisionActor(script=[], claim_done_at_end=True)
    return VisionActor(client=client)


def run_trial(condition: Condition, task_id: str, trial: int, *, planner_kind: str,
              root: Path, max_steps: int, llm: LLMClient | None,
              vlm: LLMClient | None) -> RunOutcome:
    """One (condition, task, trial) cell of the grid."""
    workspace = _trial_workspace(root, condition.name, task_id, trial)
    # The same read-only grant ``run_agent_task`` makes, from the same function.
    # policy.py's own docstring states the intent -- "This allows benchmark tasks to
    # read existing user files, such as a PDF in Downloads, without granting write
    # access there" -- but this call site never made the grant, so every
    # ``open_last_day_pdf`` trial died on ``PolicyDenied: read outside permitted
    # roots`` before its first action and the grid reported 0% verified. A benchmark
    # that cannot read its own precondition measures nothing.
    policy = Policy(workspace=workspace,
                    readable_roots=readable_roots_for_task(task_id))
    task = build_task(task_id)
    config = condition.config(max_steps=max_steps)
    trace = Trace(task_id=task_id, condition=condition.name, trial=trial)
    trace.note("condition", **{"kind": condition.kind, "description": condition.note})

    try:
        if condition.kind == "vision":
            actor = _vision_actor(planner_kind, vlm)
            return run_vision_task(task, actor, policy, config, trial=trial, trace=trace)

        planner = _structured_planner(planner_kind, task, policy, condition, llm)
        if condition.adversarial:
            injections, fire_on_step = ADVERSARIAL_PLAN[task_id]
            planner = inject.attach(planner, policy, injections, trace=trace,
                                    fire_on_step=fire_on_step)
        return run_task(task, planner, policy, config, trial=trial, trace=trace)
    finally:
        trace.close()

def _rate(flags: list[bool]) -> float:
    return round(sum(1 for f in flags if f) / len(flags), 4) if flags else 0.0


def _counter_total(outcomes: list[RunOutcome], key: str) -> int:
    return sum(int(o.trace.get("counters", {}).get(key, 0)) for o in outcomes)


def aggregate(outcomes: list[RunOutcome], *, price_in: float, price_out: float) -> dict:
    """Plan S19's columns for one group of trials.

    ``verified_success_rate`` and ``reported_success_rate`` are kept apart on
    purpose, and ``unknown_rate`` is separate from ``fail_rate`` because plan S8
    makes UNKNOWN its own verdict rather than a soft failure.
    """
    if not outcomes:
        return {"trials": 0}

    latencies = [o.wall_clock_s for o in outcomes]
    prompt_tokens = sum(int(o.usage.get("prompt_tokens", 0)) for o in outcomes)
    completion_tokens = sum(int(o.usage.get("completion_tokens", 0)) for o in outcomes)
    attempts = _counter_total(outcomes, "recovery_attempts")
    successes = _counter_total(outcomes, "recovery_successes")
    categories: Counter = Counter()
    for outcome in outcomes:
        categories.update(outcome.failure_categories)

    return {
        "trials": len(outcomes),
        "reported_success_rate": _rate([o.reported_success for o in outcomes]),
        "verified_success_rate": _rate([o.verified_success for o in outcomes]),
        "false_success_rate": _rate([o.false_success for o in outcomes]),
        "silent_failure_rate": _rate([o.silent_failure for o in outcomes]),
        "unknown_rate": _rate([o.verified is Verdict.UNKNOWN for o in outcomes]),
        "fail_rate": _rate([o.verified is Verdict.FAIL for o in outcomes]),
        "latency_median_s": round(statistics.median(latencies), 3),
        "latency_mean_s": round(statistics.fmean(latencies), 3),
        "steps_mean": round(statistics.fmean([o.steps_used for o in outcomes]), 2),
        "llm_calls": _counter_total(outcomes, "llm_calls"),
        "vision_calls": _counter_total(outcomes, "vision_calls"),
        "screenshots": _counter_total(outcomes, "screenshots"),
        "observations": _counter_total(outcomes, "observations"),
        "injections": _counter_total(outcomes, "injections"),
        "recovery_attempts": attempts,
        "recovery_successes": successes,
        "recovery_success_rate": round(successes / attempts, 4) if attempts else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "est_cost_usd": round(prompt_tokens / 1e6 * price_in
                              + completion_tokens / 1e6 * price_out, 6),
        "failure_categories": dict(categories.most_common()),
        "synthetic": any(o.synthetic for o in outcomes),
    }

def _group(outcomes: list[RunOutcome], key) -> dict[str, list[RunOutcome]]:
    grouped: dict[str, list[RunOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(key(outcome), []).append(outcome)
    return grouped


def build_report(outcomes: list[RunOutcome], *, meta: dict, price_in: float,
                 price_out: float, buckets: dict[str, str]) -> dict:
    """Aggregate three ways: by condition, by condition x task, by task bucket.

    The bucket cut is plan S18's third ablation -- it is what distinguishes "the
    structured path is better" from "the structured path is better on filesystem
    work and no different on window state", which is the claim the evidence can
    actually support.
    """
    def agg(group: list[RunOutcome]) -> dict:
        return aggregate(group, price_in=price_in, price_out=price_out)

    by_condition = {name: agg(group)
                    for name, group in _group(outcomes, lambda o: o.condition).items()}
    by_condition_task: dict[str, dict[str, dict]] = {}
    by_bucket: dict[str, dict[str, dict]] = {}
    for name, group in _group(outcomes, lambda o: o.condition).items():
        by_condition_task[name] = {
            task_id: agg(rows)
            for task_id, rows in _group(group, lambda o: o.task_id).items()
        }
        by_bucket[name] = {
            bucket: agg(rows)
            for bucket, rows in _group(group,
                                       lambda o: buckets.get(o.task_id, "?")).items()
        }
    return {
        "meta": meta,
        "by_condition": by_condition,
        "by_condition_task": by_condition_task,
        "by_bucket": by_bucket,
        "outcomes": [o.to_json() for o in outcomes],
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)

_HEADER_NOTES = """\
How to read this table:

* **verified** is the only success number. It comes from the task's own verifier
  reading the machine, not from the agent's report (plan S30).
* **reported** is what the agent claimed. **false** is the gap: claimed done,
  verification disagreed. A condition with high *reported* and low *verified* is
  worse than one that fails honestly.
* **unknown** is counted separately from failure. Plan S8: UNKNOWN is not success,
  and it is not proof of failure either.
* **vision calls** are a subset of **LLM calls**, not an addition to them: a
  screenshot turn is one model invocation that happened to carry an image. The
  count of captures taken is a separate `screenshots` field in the JSON.
* **cost** is estimated as tokens x the price passed on the command line. It is
  0.00 when no price was given and meaningless for a synthetic run.
"""


def render_markdown(report: dict) -> str:
    meta = report["meta"]
    out = [f"# Group A results -- {meta['started_at_iso']}", ""]
    if meta.get("interrupted_at"):
        out += [
            f"> **Interrupted run.** Stopped by the user at {meta['interrupted_at']}, so the",
            "> grid below is incomplete: cells after that point were never run, and the",
            "> cancelled cell itself is excluded rather than counted as a non-success.",
            "> Trial counts here are what actually executed, not what was requested.",
            "",
        ]
    if meta.get("synthetic"):
        out += [
            "> **Synthetic run.** The planner was the deterministic mock, which replays each",
            "> task's reference plan. These rows exercise the harness, the verifiers, the",
            "> tracing, and the metric definitions. They are not experimental results and",
            "> no hypothesis is supported or refuted by them.",
            "",
        ]
    out += [
        f"- planner: `{meta['planner']}`  model: `{meta.get('model') or 'n/a'}`",
        f"- trials per task per condition: {meta['trials']}"
        f"  (plan S21 minimum: 3)",
        f"- tasks: {', '.join(meta['tasks'])}",
        f"- price per 1M tokens: in ${meta['price_in']}, out ${meta['price_out']}",
        f"- platform: {meta['platform']}  python: {meta['python']}",
        f"- workspace root: `{meta['workspace_root']}`",
        f"- traces: `{meta['trace_dir']}`",
        "",
        _HEADER_NOTES,
        "## By condition",
        "",
    ]
    header = ["condition", "trials", "verified", "reported", "false", "unknown",
              "median s", "mean s", "LLM calls", "vision calls",
              "recovery a/s", "est $"]
    rows = []
    for name, agg in report["by_condition"].items():
        rows.append([
            f"`{name}`", str(agg["trials"]),
            _pct(agg["verified_success_rate"]), _pct(agg["reported_success_rate"]),
            _pct(agg["false_success_rate"]), _pct(agg["unknown_rate"]),
            f"{agg['latency_median_s']:.2f}", f"{agg['latency_mean_s']:.2f}",
            str(agg["llm_calls"]), str(agg["vision_calls"]),
            f"{agg['recovery_attempts']}/{agg['recovery_successes']}",
            f"{agg['est_cost_usd']:.4f}",
        ])
    out += [_table(header, rows), ""]

    out += ["### Conditions", ""]
    for name, condition in CONDITIONS.items():
        if name in report["by_condition"]:
            out.append(f"- `{name}` -- {condition.note}")
    out.append("")

    task_ids = sorted({o["task_id"] for o in report["outcomes"]})
    for label, metric in (("Verified success rate by task", "verified_success_rate"),
                          ("False-success rate by task", "false_success_rate")):
        out += [f"## {label}", ""]
        rows = []
        for name, per_task in report["by_condition_task"].items():
            rows.append([f"`{name}`"] + [
                _pct(per_task[t][metric]) if t in per_task else "n/a" for t in task_ids
            ])
        out += [_table(["condition"] + task_ids, rows), ""]

    out += ["## By task bucket (plan S18)", ""]
    buckets = sorted({b for per in report["by_bucket"].values() for b in per})
    rows = []
    for name, per_bucket in report["by_bucket"].items():
        rows.append([f"`{name}`"] + [
            _pct(per_bucket[b]["verified_success_rate"]) if b in per_bucket else "n/a"
            for b in buckets
        ])
    out += [_table(["condition"] + buckets, rows),
            "", "Cells are verified success rate. `n/a` means the condition did not "
            "run that bucket.", ""]

    out += ["## Failure categories", ""]
    any_category = False
    for name, agg in report["by_condition"].items():
        if agg.get("failure_categories"):
            any_category = True
            listed = ", ".join(f"{k} x{v}" for k, v in agg["failure_categories"].items())
            out.append(f"- `{name}`: {listed}")
    if not any_category:
        out.append("No classified failures in this run.")
    out.append("")

    out += ["## Per-trial rows", ""]
    trial_rows = [
        [f"`{o['condition']}`", o["task_id"], str(o["trial"]), o["verified"],
         "yes" if o["reported_success"] else "no",
         "yes" if o["false_success"] else "no",
         f"{o['wall_clock_s']:.2f}", str(o["steps_used"]),
         (o["aborted_reason"] or "")[:70]]
        for o in report["outcomes"]
    ]
    out += [_table(["condition", "task", "trial", "verified", "reported", "false",
                    "seconds", "steps", "abort reason"], trial_rows), ""]

    return "\n".join(out)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.harness",
        description="Run the Group A benchmark and write the results table.",
    )
    parser.add_argument("--planner", choices=("mock", "llm"), default="mock",
                        help="mock replays each task's reference plan (synthetic)")
    parser.add_argument("--trials", type=int, default=3,
                        help="trials per task per condition (plan S21 minimum: 3)")
    parser.add_argument("--tasks", default="all",
                        help=f"comma-separated, or 'all' ({', '.join(TASK_IDS)})")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                        help=f"comma-separated from: {', '.join(CONDITIONS)}")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--workspace-root", default="",
                        help="default: <results>/workspaces/<stamp>")
    parser.add_argument("--price-in", type=float, default=0.0,
                        help="USD per 1M prompt tokens, for the estimated-cost column")
    parser.add_argument("--price-out", type=float, default=0.0,
                        help="USD per 1M completion tokens")
    parser.add_argument("--keep-workspaces", action="store_true",
                        help="do not delete the per-trial workspaces afterwards")
    return parser.parse_args(argv)


def _resolve_tasks(spec: str) -> list[str]:
    if spec.strip() == "all":
        return list(TASK_IDS)
    wanted = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in wanted if t not in TASK_IDS]
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(unknown)}; known: {', '.join(TASK_IDS)}")
    return wanted


def _resolve_conditions(spec: str) -> list[Condition]:
    wanted = [c.strip() for c in spec.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition(s): {', '.join(unknown)}; "
                         f"known: {', '.join(CONDITIONS)}")
    return [CONDITIONS[c] for c in wanted]


def _clients(planner_kind: str, conditions: list[Condition]) -> tuple[Any, Any]:
    """One transport for the structured planner, one for vision. Same provider.

    Plan S18 requires the baseline to use the same model. ``LLMClient.from_env``
    falls back to the ``LLM_*`` variables when no ``VLM_*`` are set, so "same
    model" is the default rather than something to remember.
    """
    if planner_kind != "llm":
        return None, None
    llm = LLMClient.from_env()
    vlm = LLMClient.from_env(vision=True) if any(c.kind == "vision" for c in conditions) else None
    return llm, vlm


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    task_ids = _resolve_tasks(args.tasks)
    conditions = _resolve_conditions(args.conditions)
    if args.trials < 3:
        print(f"note: --trials {args.trials} is below plan S21's minimum of 3.",
              file=sys.stderr)

    try:
        llm, vlm = _clients(args.planner, conditions)
    except LLMUnavailable as exc:
        print(f"planner unavailable: {exc}", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(args.workspace_root) if args.workspace_root else \
        RESULTS_DIR / "workspaces" / stamp
    buckets = {task_id: getattr(build_task(task_id), "bucket", "?") for task_id in task_ids}

    outcomes: list[RunOutcome] = []
    skipped: list[str] = []
    #: Where a Ctrl+C landed, if one did. Empty string means the grid ran to the end.
    interrupted = ""
    for condition in conditions:
        for task_id in task_ids:
            if condition.adversarial and task_id not in ADVERSARIAL_PLAN:
                skipped.append(f"{condition.name}/{task_id}: single-action task, "
                               "no plan-to-act window an adversary can perturb")
                continue
            for trial in range(args.trials):
                label = f"{condition.name:<22} {task_id:<20} trial {trial + 1}/{args.trials}"
                print(f"[run] {label}", flush=True)
                outcome = run_trial(
                    condition, task_id, trial, planner_kind=args.planner,
                    root=root, max_steps=args.max_steps, llm=llm, vlm=vlm,
                )
                if outcome.cancelled:
                    # ``run_task`` returns a cancelled outcome instead of raising, so
                    # without this the interrupt would be swallowed and the grid would
                    # march on to the next trial. The cancelled cell is also kept out
                    # of ``outcomes``: it verified nothing, and a rate computed over
                    # it would report a non-success nobody measured (plan S19).
                    interrupted = f"{condition.name}/{task_id} trial {trial + 1}"
                    print(f"      cancelled: {outcome.aborted_reason}\n"
                          f"      stopping the grid. Trials already finished are kept.",
                          flush=True)
                    break
                outcomes.append(outcome)
                print(f"      verified={outcome.verified.value:<7} "
                      f"reported={'yes' if outcome.reported_success else 'no':<3} "
                      f"false={'yes' if outcome.false_success else 'no':<3} "
                      f"{outcome.wall_clock_s:.1f}s "
                      f"{outcome.aborted_reason or ''}", flush=True)
            if interrupted:
                break
        if interrupted:
            break

    if interrupted and not outcomes:
        # Nothing completed, so there is nothing to aggregate. Writing a report of
        # zero trials -- and overwriting run_latest.json with it -- would replace
        # real evidence with the absence of any.
        print(f"\ninterrupted at {interrupted} before any trial finished; "
              f"no report written.\nworkspaces kept at {root}", file=sys.stderr)
        return EXIT_INTERRUPTED

    meta = {
        "started_at_iso": stamp,
        "planner": args.planner,
        "model": getattr(llm, "model", None),
        "trials": args.trials,
        "tasks": task_ids,
        "conditions": [c.name for c in conditions],
        "price_in": args.price_in,
        "price_out": args.price_out,
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "workspace_root": str(root),
        "trace_dir": str(DEFAULT_TRACE_DIR),
        "skipped_cells": skipped,
        "synthetic": any(o.synthetic for o in outcomes),
        "interrupted_at": interrupted or None,
    }

    report = build_report(outcomes, meta=meta, price_in=args.price_in,
                          price_out=args.price_out, buckets=buckets)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"run_{stamp}.json"
    md_path = RESULTS_DIR / f"results_{stamp}.md"
    markdown = render_markdown(report)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    # Stable filenames so the README can link to something that does not move.
    (RESULTS_DIR / "run_latest.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    (RESULTS_DIR / "results_latest.md").write_text(markdown, encoding="utf-8")

    if not args.keep_workspaces and not interrupted:
        shutil.rmtree(root, ignore_errors=True)

    print()
    print(markdown)
    for line in skipped:
        print(f"[skipped] {line}", file=sys.stderr)
    print(f"\nwrote {json_path}\nwrote {md_path}", file=sys.stderr)
    if interrupted:
        print(f"interrupted at {interrupted}; the grid above is incomplete.\n"
              f"workspaces kept at {root}", file=sys.stderr)
        return EXIT_INTERRUPTED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
