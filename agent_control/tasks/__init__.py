"""Tasks that exist to serve the assistant, not the benchmark.

The split from ``benchmark/tasks/`` is about ownership, not mechanism. Both
packages define the same ``Task`` protocol and both are built by the same
``build_task``; the difference is that a task in here is a *capability the
assistant offers a person*, whose target is chosen at request time, while a task
in ``benchmark/tasks/`` is a *measurement* whose starting state is fixed so that
trials are comparable.

Keeping them apart matters in one direction specifically: production code must
not have to import the research package to know what it can do. ``benchmark``
imports from here, never the reverse.
"""

from .open_project import OpenProjectInEditor
from .setup_project import SetupPythonProject

#: Every assistant-owned task. ``benchmark.tasks`` folds this into its
#: ``INTERACTIVE_CLASSES`` so there is still exactly one task registry and one
#: ``build_task``, rather than a second lookup path that could disagree.
ASSISTANT_CLASSES = (OpenProjectInEditor, SetupPythonProject)

ASSISTANT_IDS: tuple[str, ...] = tuple(cls().task_id for cls in ASSISTANT_CLASSES)

__all__ = [
    "ASSISTANT_CLASSES",
    "ASSISTANT_IDS",
    "OpenProjectInEditor",
    "SetupPythonProject",
]
