"""Real end-to-end smoke test: create a project on disk and open it in VS Code.

Not a unit test. This launches the real planner, writes real files, builds a real
virtual environment and starts a real VS Code window, then prints the verdict list
the assistant would speak. Run it deliberately:

    .venv/Scripts/python.exe scripts/smoke_setup_project.py

It goes through ``Session.submit``, which is the same entry point the typed CLI
and the voice loop use, so what passes here is the thing a person gets.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_control import api  # noqa: E402
from agent_control.session import Session  # noqa: E402

REQUEST = "Set up a Python project called test_project and open it in VS Code."


def _holders(target: Path) -> list[psutil.Process]:
    """Processes whose own executable lives inside ``target``.

    Windows pins a directory that holds a running image, so a second run of this
    script cannot delete the first run's project while anything is executing out
    of its ``.venv`` -- neither ``rmtree`` nor even ``rename`` succeeds. In
    practice the holder is a VS Code language server: the editor adopts the new
    project's interpreter and starts ``lsp_server.py`` with it.

    Deliberately narrow. A process is only listed if its *executable* is inside
    the project this script created, which means it exists because this script
    ran. The editor itself, and any interpreter anywhere else, is never touched.
    """
    resolved = target.resolve()
    found: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            exe = proc.info.get("exe")
            if exe and Path(exe).resolve().is_relative_to(resolved):
                found.append(proc)
        except (psutil.Error, OSError, ValueError):
            continue
    return found


def _clear(target: Path) -> bool:
    """Remove the previous run's project. ``False`` if it could not be removed."""
    if not target.exists():
        return True

    print(f"[smoke] removing the previous run's project at {target}")
    try:
        shutil.rmtree(target)
        return True
    except PermissionError as exc:
        print(f"[smoke] locked: {exc}")

    holding = _holders(target)
    for proc in holding:
        print(f"[smoke] stopping pid {proc.pid} ({proc.info.get('name')}) "
              f"running from inside the project")
        try:
            proc.terminate()
        except psutil.Error:
            pass
    if holding:
        gone, alive = psutil.wait_procs(holding, timeout=5)
        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass
        time.sleep(0.5)

    try:
        shutil.rmtree(target)
        return True
    except OSError as exc:
        print(f"[smoke] could not remove the folder itself: {exc}")

    # Windows also pins a directory that is any process's working directory, and
    # VS Code makes the folder it has open exactly that -- for the window, for an
    # extension, and for the integrated terminal. Those are the user's editor, not
    # this script's leftovers, so they are not killed. It does not matter:
    # ``api.resolve_setup_request`` refuses a project folder that is *occupied*,
    # and an empty one is a legal starting state, so an emptied directory is as
    # good as an absent one.
    try:
        if not any(target.iterdir()):
            print(f"[smoke] {target} is empty, which is a legal starting state.")
            return True
    except OSError:
        pass

    print(f"[smoke] {target} still has contents in it.")
    print("[smoke] close the VS Code window on that project and run again.")
    return False


def main() -> int:
    target = api.projects_root() / "test_project"

    if not _clear(target):
        return 2

    print(f"[smoke] projects root : {api.projects_root()}")
    print(f"[smoke] request       : {REQUEST}")
    print()

    session = Session.build(speech=False, planner="llm", max_steps=8)
    try:
        turn = session.submit(REQUEST)
    finally:
        session.close()

    print()
    print(f"[smoke] status        : {turn.result.status.value if turn.result else 'no result'}")
    print(f"[smoke] project on disk: {target.exists()}")
    if target.exists():
        listing = sorted(
            str(p.relative_to(target)) for p in target.rglob("*")
            if ".venv" not in p.parts and "__pycache__" not in p.parts
        )
        print(f"[smoke] files          : {listing}")
        print(f"[smoke] venv python    : {(target / '.venv' / 'Scripts' / 'python.exe').exists()}")

    if turn.result is not None:
        print(f"[smoke] steps used     : {turn.result.steps_used}")
        # Every check name a run may legitimately report without a project-relative
        # prefix: the project's own directory, and the three the response layer
        # matches on to phrase an editor failure. Anything else bare means a
        # checkpoint named a file differently from final verification, and
        # ``api._derive`` keys by name -- so the report would carry two rows for
        # one file and bury every write but the last.
        bare_by_design = {"dir_exists", "app_process", "app_window", "viewer_window"}
        leaked = sorted(
            name for name in turn.result.completed + turn.result.failed
            if "/" not in name and name not in bare_by_design
        )
        print(f"[smoke] unprefixed leaks: {leaked or 'none'}")

    return 0 if turn.result is not None and turn.result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
