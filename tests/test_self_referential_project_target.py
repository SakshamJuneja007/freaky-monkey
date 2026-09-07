"""Tests for self-referential "analyze this project" target resolution.

Traced root cause (see the implementation report): "analyze this project" was
never reaching GeneralTask at all. ``session._requires_computer_action`` had
no verb in ``_ACTION_VERBS`` for "analyze"/"inspect"/"review"/"investigate",
so ``classify_intent`` returned ``IntentCategory.CONVERSATION`` and the
request went to the LLM conversation engine -- which has no task, no policy,
and no filesystem access, and produced the reported "which project?" reply
from pure chit-chat.

Once that verb gap is closed, the second gap is the one width of this fix:
``api.general_readable_roots()`` never granted the DEIMOS repository itself,
only ``projects_root()`` and Downloads, so even a correctly-routed
GeneralTask had nothing real to look at.

This file tests both, plus the safety requirements: the repository becomes
readable but never writable, sensitive files stay refused, and only a narrow
self-referential phrasing triggers the grant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_control import api, session
from agent_control.general_task import GeneralTask
from agent_control.policy import Policy, PolicyDenied
from agent_control.session import IntentCategory, classify_intent


# -- Test 1 / 6: self-referential vs. ambiguous target resolution ----------

SELF_REFERENTIAL = [
    "Analyze this project",
    "Analyze this codebase",
    "Review the current project",
    "Inspect the current codebase",
    "Find problems in this project",
    "Inspect this project",
    "Review this project",
    "Review this codebase",
    "Inspect this codebase",
    "Analyze the architecture of this project",
]

NOT_SELF_REFERENTIAL = [
    "Analyze a project",
    "Analyze my project",
    "Review another project",
    "Analyze the project in Downloads",
]


@pytest.mark.parametrize("text", SELF_REFERENTIAL)
def test_self_referential_requests_resolve_to_project_root(text: str):
    roots = api.general_readable_roots(text)
    assert api.ROOT in roots


@pytest.mark.parametrize("text", NOT_SELF_REFERENTIAL)
def test_ambiguous_requests_do_not_resolve_to_project_root(text: str):
    """Test 6: an ambiguous or differently-targeted request must not silently
    grant the repository -- the existing clarification/general-task behavior
    for these is unaffected."""
    roots = api.general_readable_roots(text)
    assert api.ROOT not in roots
    # Unaffected: still exactly the historical two-root grant.
    assert roots == (api.projects_root(), Path.home() / "Downloads")


def test_analyze_this_project_is_classified_as_action_not_conversation():
    """The actual reported bug: without an action verb, this sentence was
    classified CONVERSATION and never reached GeneralTask at all."""
    intent = classify_intent("Analyze this project", accepted=False,
                              structured=False, pending=False,
                              previous_executed=False)
    assert intent is IntentCategory.ACTION


def test_analyze_this_project_is_not_claimed_by_resolve_request():
    """It must fall through resolve_request (no registered/open/setup/project
    shape matches "analyze") so Session.submit routes it to _general_action,
    not to a structured workflow or an ambiguity question about a named
    folder."""
    assert api.resolve_request("Analyze this project", use_memory=False) is None


# -- Test 2: the analysis target is not a sandbox directory -----------------

def test_self_referential_target_excludes_sandbox_paths():
    """The grant this fix adds is the repository, never a
    .sandbox/run-.../general scratch directory -- that stays Policy.workspace
    and is a completely separate concept (see the module docstring)."""
    roots = api.general_readable_roots("Analyze this project")
    assert not any(".sandbox" in str(root) for root in roots)
    assert api.ROOT in roots


# -- Test 3: PROJECT_ROOT source files are readable through the policy ------

@pytest.fixture()
def project_policy(tmp_path: Path) -> Policy:
    roots = api.general_readable_roots("Analyze this project")
    return Policy(workspace=tmp_path / "sandbox", readable_roots=roots,
                  refuse_if_elevated=False)


@pytest.mark.parametrize("rel_path", ["main.py", "agent_control/session.py"])
def test_project_root_source_files_are_readable(project_policy: Policy, rel_path: str):
    resolved = project_policy.resolve_read_path(api.ROOT / rel_path)
    assert resolved == (api.ROOT / rel_path).resolve()


# -- Test 4 / 5: write safety ------------------------------------------------

def test_writing_to_project_root_is_denied(project_policy: Policy):
    """Self-referential analysis must be read-only: PROJECT_ROOT/main.py must
    never be a valid write target."""
    with pytest.raises(PolicyDenied):
        project_policy.resolve_write_path(api.ROOT / "main.py")


def test_writing_to_sandbox_still_allowed(project_policy: Policy):
    """The sandbox must remain usable for ordinary scratch writes -- this fix
    must not have narrowed Policy.workspace."""
    resolved = project_policy.resolve_write_path("notes.txt")
    assert resolved.is_relative_to(project_policy.workspace)


# -- Test 7: sensitive-file protection remains enforced under the new root --

@pytest.mark.parametrize("sensitive_name", [".env", ".ssh/id_rsa", "secrets.json"])
def test_sensitive_files_under_project_root_stay_refused(project_policy: Policy,
                                                           sensitive_name: str):
    with pytest.raises(PolicyDenied):
        project_policy.resolve_read_path(api.ROOT / sensitive_name)


def test_ordinary_readable_roots_grant_unaffected_by_sensitive_fix():
    """The .env/secret/.pem additions to SENSITIVE_FRAGMENTS must not start
    refusing ordinary source files that merely contain similar substrings."""
    from agent_control.policy import Policy as _Policy
    policy = _Policy(workspace=Path("/tmp/deimos_sensitive_check"),
                      readable_roots=(api.ROOT,), refuse_if_elevated=False)
    # None of these are secrets; must remain readable.
    for rel in ["agent_control/session.py", "agent_control/policy.py"]:
        policy.resolve_read_path(api.ROOT / rel)


# -- Observation: GeneralTask.observe() actually surfaces the repository ----

def test_observe_surfaces_real_repository_entries(project_policy: Policy):
    """Not merely wiring PROJECT_ROOT into an unused variable: the planner's
    actual initial observation must list real repository entries."""
    task = GeneralTask(request="Analyze this project",
                        readable_roots=project_policy.readable_roots)
    observation = task.observe(project_policy)
    entries = observation["recent_entries"].value["entries"]
    names = {entry["name"] for entry in entries}
    assert "main.py" in names
    assert "agent_control" in names


def test_non_self_referential_request_observation_has_no_project_root(tmp_path: Path):
    """Control: an unrelated general request must not incidentally see the
    repository just because ROOT happens to exist on disk."""
    roots = api.general_readable_roots("open a project called demo")
    policy = Policy(workspace=tmp_path / "sandbox2", readable_roots=roots,
                     refuse_if_elevated=False)
    assert api.ROOT not in policy.readable_roots
    with pytest.raises(PolicyDenied):
        policy.resolve_read_path(api.ROOT / "main.py")
