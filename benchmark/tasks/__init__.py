"""Task definitions, in two sets that must not be confused.

``BENCHMARK_IDS`` is the measured set: Group A only, which plan S31 forbids
extending before the first published result. ``TASK_IDS`` is everything the
interactive interfaces may run, which is the benchmark set plus the tasks that only
make sense with a runtime parameter.

The split exists because ``harness.py`` treats ``--tasks all`` as "every registered
id". A task that needs a filename cannot run unattended, so enrolling it in the
benchmark would either crash a trial or, worse, contribute a meaningless row to the
results table.
"""

from agent_control.tasks import ASSISTANT_CLASSES

from .group_a import TASK_CLASSES as BENCHMARK_CLASSES
from .group_a import TASK_IDS as BENCHMARK_IDS
from .open_named import OpenNamedFile

#: Tasks whose target is chosen per request. Not benchmarked: the point of the
#: benchmark is an identical starting state across trials, and these do not have one.
#:
#: ``ASSISTANT_CLASSES`` comes from ``agent_control.tasks``, where the assistant's
#: own capabilities live. They are listed here rather than defined here so that
#: there is still exactly one ``build_task``, while production code stays able to
#: name what it can do without importing the research package.
INTERACTIVE_CLASSES = (OpenNamedFile,) + ASSISTANT_CLASSES

INTERACTIVE_IDS: tuple[str, ...] = tuple(
    cls().task_id for cls in INTERACTIVE_CLASSES
)

#: Everything a person may ask for by name. What ``api.registered_tasks`` lists.
TASK_CLASSES = BENCHMARK_CLASSES + INTERACTIVE_CLASSES

TASK_IDS: tuple[str, ...] = BENCHMARK_IDS + INTERACTIVE_IDS


def build_task(task_id: str, **params):
    """Build one fresh task instance, parameterized if it takes parameters.

    ``params`` reaches the dataclass constructor unchanged, so an unknown keyword
    is a ``TypeError`` naming the field -- which is the right failure for a caller
    that invented a parameter, rather than a silently ignored argument.
    """
    for cls in TASK_CLASSES:
        if cls().task_id == task_id:
            return cls(**params)

    raise KeyError(
        f"unknown task {task_id!r}; known: {list(TASK_IDS)}"
    )


def build_tasks(
    task_ids: list[str] | tuple[str, ...] | None = None,
):
    """Build the requested tasks, or the full current *benchmark*.

    The default is deliberately ``BENCHMARK_IDS`` and not ``TASK_IDS``: every
    caller that passes nothing is asking for the measured set.
    """
    return [
        build_task(task_id)
        for task_id in (task_ids or BENCHMARK_IDS)
    ]


__all__ = [
    "BENCHMARK_CLASSES",
    "BENCHMARK_IDS",
    "INTERACTIVE_CLASSES",
    "INTERACTIVE_IDS",
    "TASK_CLASSES",
    "TASK_IDS",
    "OpenNamedFile",
    "build_task",
    "build_tasks",
]
