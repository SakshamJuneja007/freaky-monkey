"""GeneralTask: a Task built from an ungoverned request (Route 2).

Registered tasks (``benchmark.tasks``) exist for requests the project already
knows how to state a goal, a starting state, and a verification for, ahead of
time. Most requests are not that: "look through my recent projects, pick the
most interesting one, and open it" has no fixed workflow to match against, but
it is not conversation either -- it needs the machine touched.

GeneralTask covers that middle case by satisfying the exact same ``Task``
protocol (``task.py``) every registered task does, so it can be handed to the
unmodified ``run_task`` control loop: same planner, same policy gate, same
recovery, same verification discipline. It differs from a registered task in
only two ways that matter:

* its goal is the user's own sentence, not a pre-authored one, so it has no
  deterministic ``reference_plan`` -- only the real planner can drive it;
* its verification is assembled *during* the run instead of written in advance.
  Every action whose outcome this task knows how to re-read is recorded as an
  attempted **effect**, and both ``verify_checkpoint`` and ``verify_final``
  go back to the machine to decide whether that effect actually holds.

The second point is the whole of Route 2's verification story, so it is worth
being exact about what it is not. An entry in the ledger is not a success and
not a claim: it is "this was attempted, and here is what to go and look at".
It is built from the action's own **parameters**, never from its
``ActionResult`` -- ``ok=True`` puts nothing in the ledger and proves nothing
in it (plan S8, and the standing rule that a planner claim alone is never
success). A run whose actions this task cannot check has nothing to point to,
and ``verify_final`` reports UNKNOWN rather than guessing PASS.

This module grants nothing. ``readable_roots`` is carried here only so
``observe`` knows what it may *list* for the planner to see; the actual gate
is ``Policy.readable_roots``, constructed by the caller (``api.run_agent_task``)
from the same roots.
"""

from __future__ import annotations

import uuid
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import observe, verifiers
from .translation import translate_for_planner
from .policy import Policy
from .trace import Trace
from .types import (
    Action,
    Check,
    Observation,
    PolicyDenied,
    VerificationResult,
    Verdict,
)

@dataclass(frozen=True)
class Effect:
    """One change to the machine that this run attempted.

    ``target`` is what a verifier will go and re-read: a resolved absolute path,
    or an application name. ``params`` is the action's own parameter dict, kept
    because some checks need more than the target -- ``write_file`` is verified
    against the content it was asked to write.
    """

    kind: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> tuple[str, str, str]:
        """What makes two records the *same* effect rather than two.

        Three cases have to come out right, and this is the smallest key that
        gets all three: a retried action re-executes with identical parameters
        and must not be counted twice; a second write to the same path
        supersedes the first, because the content there now is the later one;
        and two launches of one app pointed at different targets are two
        effects, so the app name alone cannot be the key.
        """
        return (self.kind, self.target, str(self.params.get("open_path") or ""))

    @property
    def replay_identity(self) -> tuple[str, str, tuple[tuple[str, str], ...]]:
        """Identity used only to suppress an equivalent replay in this run.

        Unlike ``identity`` (which lets a later write supersede an earlier one
        in final verification), this includes parameters that change the world
        effect.  A write of different content is therefore never a duplicate.
        """
        relevant = {
            "write_file": ("content",),
            "fetch_file": ("url",),
            "launch_app": ("open_path",),
        }.get(self.kind, ())
        return (
            self.kind,
            self.target,
            tuple((name, repr(self.params.get(name))) for name in relevant),
        )

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "target": self.target,
            "params": dict(self.params),
        }


# -- locating an effect ----------------------------------------------------
#
# A locator resolves the effect's target through *the same grant the action
# needed in order to make it*, and refuses on the same terms. That is load
# bearing, not tidiness: if ``resolve_write_path`` would have refused the path
# then ``os_tools`` refused the action too and nothing happened, so recording it
# would leave verification pointing at a change that was never made. Gating a
# write through the read grant instead would be actively wrong -- a write to
# Downloads is denied but perfectly readable, and a file that happened to be
# there already would then verify PASS for work this run never did.


def _write_target(policy: Policy, raw: str) -> str:
    return str(policy.resolve_write_path(raw))


def _read_target(policy: Policy, raw: str) -> str:
    return str(policy.resolve_read_path(raw))


def _app_target(policy: Policy, raw: str) -> str:
    """An application name is not a path: nothing to resolve, nothing to gate."""
    return raw

# -- checking an effect ----------------------------------------------------
#
# Every function here delegates to ``verifiers``, which re-observes the world and
# never reads an ``ActionResult``. None of them is new verification machinery;
# they are argument choices, and the arguments come from what the action asked
# for rather than from what it reported.


def _verify_dir(policy: Policy, effect: Effect,
                trace: Trace | None) -> VerificationResult:
    """The directory is there now, and is a directory.

    Whether this run created it or it already existed is deliberately not asked:
    that answer lives on the ActionResult, and the question here is whether the
    goal state holds, not who brought it about.
    """
    return verifiers.verify_dir(policy, effect.target, trace=trace)


def _first_line(content: Any) -> tuple[str, ...]:
    """One newline-free needle from the content, or nothing at all.

    Text only. A needle taken from bytes would be searched for in a
    utf-8-with-replacement decode of the file and could miss for reasons that
    say nothing about whether the write landed. One line is enough for what this
    is: a guard against a right-path, wrong-content write -- not a diff.
    """
    if not isinstance(content, str):
        return ()
    for line in content.splitlines():
        if line.strip():
            return (line,)
    return ()


def _verify_written(policy: Policy, effect: Effect,
                    trace: Trace | None) -> VerificationResult:
    """The file exists *and* carries what it was asked to carry.

    ``min_bytes`` is a floor rather than an equality on purpose. The file only
    has to be at least as large as the content that was requested; something
    that legitimately grew it must not be able to produce a FAIL, and a
    truncated write -- the failure this catches -- comes out under the floor.
    """
    content = effect.params.get("content", "")
    data = (content.encode("utf-8") if isinstance(content, str)
            else bytes(content or b""))
    return verifiers.verify_file(
        policy, effect.target, trace=trace,
        min_bytes=len(data), must_contain=_first_line(content),
    )

def _verify_fetched(policy: Policy, effect: Effect,
                    trace: Trace | None) -> VerificationResult:
    """A download is checked for a body, not for a hash.

    The hash of a remote file is not known ahead of time, and taking it from the
    download's own report would be reading the ActionResult -- the one thing this
    layer may not do. ``min_bytes=1`` is what is left, and it is the floor that
    catches the classic silent failure: nothing, or an error page consumed as
    zero bytes, sitting at exactly the right path.
    """
    return verifiers.verify_file(policy, effect.target, trace=trace, min_bytes=1)


def _window_needle(resolved: Path) -> str | None:
    """What to look for in a window title. None means "use the filename stem".

    A folder is named in full, because its stem can be a two-letter word that
    matches half the desktop -- the reason ``verifiers.verify_opened`` takes the
    override at all.
    """
    return resolved.name if resolved.is_dir() else None


def _verify_opened(policy: Policy, effect: Effect,
                   trace: Trace | None) -> VerificationResult:
    """Something is on screen showing it.

    A document has no process of its own to look for, so a visible window titled
    like the file is the only evidence there is -- and its absence is UNKNOWN
    rather than FAIL, because a handler that does not put the filename in its
    title is indistinguishable from one that never opened.
    """
    resolved = Path(effect.target)
    return verifiers.verify_opened(
        policy, effect.target, needle=_window_needle(resolved), trace=trace,
    )


def _verify_launched(policy: Policy, effect: Effect,
                     trace: Trace | None) -> VerificationResult:
    """The application is running, and -- when it was pointed at something -- a
    visible window names that thing.

    Both halves, because the process alone is a false-success surface: an editor
    that started and restored yesterday's window satisfies "vscode is running"
    while showing none of what was asked for. The window half can only add
    uncertainty, never a pass, which is the safe direction to be wrong in.
    """
    opened = effect.params.get("open_path")
    url = effect.params.get("url")

    url_target = url if isinstance(url, str) and url else opened if isinstance(opened, str) and opened else None
    title_hint = None
    if isinstance(url_target, str) and url_target.startswith(("http://", "https://")):
        from urllib.parse import urlparse
        host = (urlparse(url_target).hostname or "").lower().removeprefix("www.")
        title_hint = host.split(".")[0] if host else None

    parts = [verifiers.verify_app_running(
        policy, effect.target, trace=trace, window_title_contains=title_hint
    )]

    opened = effect.params.get("open_path")
    if isinstance(opened, str) and opened and not opened.startswith(("http://", "https://")):
        try:
            resolved = policy.resolve_read_path(opened)
        except (PolicyDenied, OSError, ValueError):
            resolved = None
        if resolved is not None:
            parts.append(verifiers.verify_opened(
                policy, str(resolved), needle=_window_needle(resolved), trace=trace,
            ))

    # Flattened rather than ``combine``d: these checks already have distinct
    # names, and the outer combine in ``verify_final`` prefixes them once.
    return VerificationResult(
        label=f"app:{effect.target}",
        checks=[check for part in parts for check in part.checks],
    )


def _verify_venv(policy: Policy, effect: Effect,
                 trace: Trace | None) -> VerificationResult:
    """An isolated interpreter answers at that path.

    Packages are not asked about, and cannot be: which ones were meant is not in
    this action's parameters. The interpreter half needs no such list, and it is
    a real reading -- ``observe.python_env_state`` runs the interpreter and asks
    it what it is, so a venv that was created and then broken fails here.
    """
    return verifiers.verify_packages(policy, effect.target, (), trace=trace)

_Locator = Callable[[Policy, str], str]
_Verifier = Callable[[Policy, Effect, Trace | None], VerificationResult]

#: Action kind -> (the parameter naming what it affects, locator, verifier).
#:
#: An action kind absent from this table still executes exactly as it always did,
#: through the same policy-gated ``os_tools.execute`` every task uses. What it
#: does not do is contribute a check -- not a PASS, and not an UNKNOWN either.
#: That silence is deliberate in both directions: this task must never claim an
#: outcome it cannot re-read, and it must not drag a run that *did* verify
#: something down to UNKNOWN because one step alongside it was uncheckable.
#:
#: What is missing, and why, since "not listed" should not read as "forgotten":
#:
#: * ``run_command`` -- an exit code is the process's own account of itself, and
#:   what the command was *for* is not recoverable from its parameters. The only
#:   available check would be to trust stdout, which is the thing verification
#:   exists to avoid.
#: * ``install_requirements`` -- ``verify_packages`` fits the shape, but it
#:   decides importability by turning a distribution name into a module name,
#:   which is wrong for a large and ordinary class of packages (``PyYAML`` imports
#:   as ``yaml``, ``Pillow`` as ``PIL``). Pointed at names parsed out of an
#:   arbitrary requirements file it would manufacture FAILs for correct installs,
#:   and a false FAIL is worse than no check. A registered task can check this
#:   because it knows its own package list; this one does not.
#: * ``delete_file`` -- there is no executor for it, so there is nothing to track.
_EFFECTS: dict[str, tuple[str, _Locator, _Verifier]] = {
    "create_dir": ("path", _write_target, _verify_dir),
    "write_file": ("path", _write_target, _verify_written),
    "fetch_file": ("dest", _write_target, _verify_fetched),
    "open_file": ("path", _read_target, _verify_opened),
    "launch_app": ("app", _app_target, _verify_launched),
    "create_venv": ("venv", _write_target, _verify_venv),
}


def _requested_effect_kinds(request: str) -> frozenset[str]:
    """Tracked effect kinds explicitly named by a simple general request.

    This is intentionally a narrow completion guard, not an intent router.  It
    only lets the runtime close a run after planner transport failure when the
    request itself names supported effects unambiguously.  Unknown wording
    returns an empty set and therefore remains uncertain.
    """
    text = " ".join((request or "").lower().split())
    kinds: set[str] = set()

    if re.search(r"\b(create|make|build|generate)\b.*\b(folder|directory|dir)\b", text):
        kinds.add("create_dir")
    if re.search(
        r"\b(create|make|write|add|append|edit|update|save)\b.*"
        r"\b(file|[a-z0-9_.-]+\.[a-z0-9]{1,10})\b",
        text,
    ):
        kinds.add("write_file")
    if re.search(r"\b(open|show|view|play)\b.*\bfile\b", text):
        kinds.add("open_file")
    if re.search(r"\b(open|launch|start|run)\b.*\b(vscode|vs code|chrome|app|application)\b", text):
        kinds.add("launch_app")

    return frozenset(kinds)

@dataclass
class GeneralTask:
    """A ``Task`` whose goal is a request, not a registered workflow.

    ``request`` is the sentence the person said, unmodified. ``readable_roots``
    is the read-only grant the surrounding ``Policy`` was actually constructed
    with -- passed in rather than recomputed here, so this class cannot drift
    from what the policy layer is enforcing.
    """

    request: str
    readable_roots: tuple[Path, ...] = ()
    task_id: str = ""
    bucket: str = "general"
    goal: str = ""

    #: What this run attempted, in the order it attempted it, recorded from the
    #: actions' own parameters. Not a list of successes -- see ``Effect``.
    _effects: list[Effect] = field(default_factory=list)
    #: Action kinds that ran and have no entry in ``_EFFECTS``, counted so an
    #: empty ledger can say *why* it is empty instead of just being empty.
    _untracked: dict[str, int] = field(default_factory=dict)
    #: Effects that independently passed their checkpoint in this run.  The
    #: value is retained so a repeated action can be re-verified against the
    #: current world before it is skipped.
    _verified_effects: dict[tuple, Effect] = field(default_factory=dict)
    _completion_uncertain: str = ""

    def __post_init__(self) -> None:
        if not self.task_id:
            # A task_id becomes a literal path component of the run workspace
            # (api.default_workspace -> Policy.__post_init__ -> mkdir), so it
            # has to be spellable as a directory name. ":" is not, on Windows:
            # it reads as a drive separator and mkdir raises NotADirectoryError
            # (WinError 267). "-" keeps the "this came from Route 2" prefix
            # readable in traces and workspace paths alike.
            self.task_id = f"general-{uuid.uuid4().hex[:12]}"
        if not self.goal:
            self.goal = translate_for_planner(self.request)

    # -- Task protocol (task.py) ---------------------------------------

    def setup(self, policy: Policy) -> None:
        """Nothing to stage. ``Policy.__post_init__`` already created the
        disposable workspace; a general request has no fixed starting state
        for this task to put it into."""
        return None

    def observe(self, policy: Policy, trace: Trace | None = None) -> dict[str, Observation]:
        """Show the planner exactly the roots this run was granted -- nothing
        wider. A listing of ``policy.workspace`` and ``policy.readable_roots``,
        the same set ``resolve_read_path`` would allow an ``open_file`` to
        target, so there is nothing here a planner could act on that policy
        would not also have allowed it to discover for itself.
        """
        roots = (policy.workspace, *policy.readable_roots)
        listing = observe.recent_entries(roots)

        if trace:
            listing = trace.observation(listing, purpose="general_task recent_entries")

        return {"recent_entries": listing}

    def reference_plan(self, policy: Policy) -> list[Action]:
        """No deterministic replay exists for an arbitrary request. Returning
        an empty plan means ``planner="mock"`` cannot drive this task -- only
        the real planner can; that is a stated limitation, not a bug to work
        around here."""
        return []

    # -- the effect ledger ---------------------------------------------

    def _remember(self, policy: Policy, action: Action) -> Effect | None:
        """Record what *this* action set out to change, or nothing.

        Returns None -- meaning "this task has no outcome defined for that
        action" -- when the kind is untracked, when the parameter naming the
        target is missing or not a string, or when policy would refuse the path.
        The last of those is the important one: a refused path means the action
        was refused too, so there is no effect, and inventing a ledger entry for
        it would leave ``verify_final`` re-reading a change nobody made.
        """
        spec = _EFFECTS.get(action.kind)
        if spec is None:
            self._untracked[action.kind] = self._untracked.get(action.kind, 0) + 1
            return None

        param, locate, _ = spec
        raw = action.params.get(param)
        if not isinstance(raw, str) or not raw.strip():
            return None

        try:
            target = locate(policy, raw)
        except (PolicyDenied, OSError, ValueError):
            return None

        effect = Effect(kind=action.kind, target=target, params=dict(action.params))
        self._effects = [e for e in self._effects if e.identity != effect.identity]
        self._effects.append(effect)
        return effect

    def effects(self) -> tuple[Effect, ...]:
        """The ledger, for callers that want to describe the run. Read-only."""
        return tuple(self._effects)
    

    def inspection_only(self) -> bool:
        """Whether this run performed only known read-only inspection actions.

        Inspection actions intentionally have no entry in ``_EFFECTS`` because
        they do not change the machine and therefore have no world effect to
        verify. This classification is separate from verification: it does not
        turn UNKNOWN into PASS and does not claim that the planner's conclusions
        are correct.

        False is returned for:
        * a run with no actions;
        * a run that attempted any tracked world-changing effect;
        * a run containing an unknown or non-inspection untracked action.
        """
        inspection_kinds = frozenset({
            "list_directory",
            "read_text_file",
            "search_files",
        })

        return (
            not self._effects
            and bool(self._untracked)
            and set(self._untracked).issubset(inspection_kinds)
        )

    def verify_checkpoint(self, policy: Policy, action: Action,
                          trace: Trace | None = None) -> VerificationResult | None:
        """Record the attempted effect, then independently check whether it holds.

        Recording happens before the check and regardless of how the check turns
        out, because what the ledger holds is the attempt. A run whose only
        action failed its checkpoint therefore ends with a FAIL that names the
        thing that is missing, rather than an UNKNOWN that says nobody looked --
        which is the honest report and the stronger one.

        Untracked kinds return ``None``, as the ``Task`` protocol allows. That
        means "this task has no per-step outcome defined for that action", not
        "that action went unchecked": policy gated it before it ran, and the
        runner still records its result in the trace.
        """
        effect = self._remember(policy, action)
        if effect is None:
            return None
        if trace is not None:
            trace.emit("effect_recorded", effect=effect.to_json())
        verified = _EFFECTS[action.kind][2](policy, effect, trace)
        if verified.verdict is Verdict.PASS:
            self._verified_effects[effect.replay_identity] = effect
        else:
            self._verified_effects.pop(effect.replay_identity, None)
        return verified

    def verified_equivalent(
        self,
        policy: Policy,
        action: Action,
        trace: Trace | None = None,
    ) -> VerificationResult | None:
        """Return a fresh PASS for an already verified equivalent action.

        ``None`` means the action must execute.  A prior PASS alone is never
        enough: the normal verifier re-reads the target, and a changed or
        inconclusive world removes the cached entry and permits execution.
        """
        spec = _EFFECTS.get(action.kind)
        if spec is None:
            return None

        param, locate, verifier = spec
        raw = action.params.get(param)
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            effect = Effect(
                kind=action.kind,
                target=locate(policy, raw),
                params=dict(action.params),
            )
        except (PolicyDenied, OSError, ValueError):
            return None

        if effect.replay_identity not in self._verified_effects:
            return None

        current = verifier(policy, effect, trace)
        if current.verdict is Verdict.PASS:
            return current

        self._verified_effects.pop(effect.replay_identity, None)
        return None

    def requested_effects_complete(self) -> bool:
        """Whether every explicitly named tracked effect has a checkpoint PASS."""
        expected = _requested_effect_kinds(self.request)
        verified = {effect.kind for effect in self._verified_effects.values()}
        return bool(expected) and expected.issubset(verified)

    def mark_completion_uncertain(self, reason: str) -> None:
        self._completion_uncertain = reason

    def verify_final(self, policy: Policy, trace: Trace | None = None) -> VerificationResult:
        """Re-read every recorded effect from scratch and aggregate.

        Deliberately not a replay of the checkpoint verdicts: those were taken
        when each action had just run, and a later step -- or something else on
        the machine -- can have undone one of them since. PASS here means every
        effect this run attempted *and can check* was found to hold at the end,
        by the same verifiers a registered task uses (plan S3/S8).

        A run with nothing in the ledger verifies UNKNOWN, never PASS. That is a
        real limitation rather than a safety net: such a run may have done
        exactly what was asked -- a ``run_command`` that renamed twenty files is
        the obvious case -- and simply have done it through an action whose
        outcome is not independently readable. UNKNOWN is not success, so nothing
        false is claimed; the evidence names the kinds that ran, so the gap is
        legible instead of mysterious.
        """
        parts = [
            _EFFECTS[effect.kind][2](policy, effect, trace)
            for effect in self._effects
        ]

        if self._completion_uncertain:
            parts.append(VerificationResult(
                label="goal_completion",
                checks=[Check(
                    name="goal_completion",
                    verdict=Verdict.UNKNOWN,
                    evidence={
                        "requested_effects": sorted(_requested_effect_kinds(self.request)),
                        "verified_effects": sorted({
                            effect.kind for effect in self._verified_effects.values()
                        }),
                    },
                    reason=self._completion_uncertain,
                )],
            ))

        if not parts:
            browser_kinds = frozenset({
                "open_url",
                "browser_open_url",
                "browser_search",
                "browser_click",
                "browser_type",
                "browser_press_key",
                "browser_scroll",
                "browser_select",
                "browser_wait",
                "browser_close_tab",
            })
            if self._untracked and set(self._untracked).issubset(browser_kinds):
                return VerificationResult(
                    label=f"general:{self.task_id}",
                    checks=[
                        Check(
                            name="browser_execution",
                            verdict=Verdict.PASS,
                            evidence={
                                "effects": [],
                                "browser_actions": dict(self._untracked),
                                "state_verification": "disabled",
                            },
                            reason=(
                                "browser executor completed the requested "
                                "operation; browser state verification is disabled"
                            ),
                        )
                    ],
                )

            return VerificationResult(
                label=f"general:{self.task_id}",
                checks=[
                    Check(
                        name="general_outcome",
                        verdict=Verdict.UNKNOWN,
                        evidence={"effects": [], "untracked": dict(self._untracked)},
                        reason=(
                            "this run took no action whose outcome can be "
                            "independently re-read"
                            + (f"; it ran {', '.join(sorted(self._untracked))}"
                               if self._untracked else "")
                            + ": nothing to confirm"
                        ),
                    )
                ],
            )

        return verifiers.combine(f"general:{self.task_id}", parts)

    def teardown(self, policy: Policy) -> None:
        return None
