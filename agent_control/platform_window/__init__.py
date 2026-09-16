"""Platform seam for window / accessibility / synthetic-input state.

Filesystem (pathlib) and process (psutil) state are already portable, so this is
the only genuinely platform-divergent layer. It exists as a Protocol with one
backend per OS so the Linux/AT-SPI backend named in the plan (S5, S13) can drop
in without touching observe.py, verifiers.py, or the runner.

Backends report ``available`` honestly. An unavailable capability must surface as
UNKNOWN, never as a passing check (plan S3: unknown is not success).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    title: str
    pid: int
    visible: bool
    focused: bool
    bbox: tuple[int, int, int, int] | None = None

    def to_json(self) -> dict:
        return {
            "handle": self.handle,
            "title": self.title,
            "pid": self.pid,
            "visible": self.visible,
            "focused": self.focused,
            "bbox": list(self.bbox) if self.bbox else None,
        }


@runtime_checkable
class WindowBackend(Protocol):
    name: str
    available: bool

    def list_windows(self) -> list[WindowInfo]: ...
    def focus(self, handle: int) -> bool: ...
    def screenshot_png(self) -> bytes | None: ...
    def click(self, x: int, y: int) -> bool: ...
    def type_text(self, text: str) -> bool: ...
    def press_keys(self, combo: str) -> bool: ...
    def accessibility_tree(self, handle: int) -> dict | None: ...
    def read_text(self, handle: int) -> str | None: ...

    def read_browser_text_observation(self) -> "TextObservation": ...


class NullBackend:
    """Used when no backend supports the host. Everything is UNKNOWN, not False."""

    name = "null"
    available = False

    def __init__(self, reason: str = "no window backend for this platform") -> None:
        self.reason = reason

    def list_windows(self) -> list[WindowInfo]:
        return []

    def focus(self, handle: int) -> bool:
        return False

    def screenshot_png(self) -> bytes | None:
        return None

    def click(self, x: int, y: int) -> bool:
        return False

    def type_text(self, text: str) -> bool:
        return False

    def press_keys(self, combo: str) -> bool:
        return False

    def accessibility_tree(self, handle: int) -> dict | None:
        return None

    def read_text(self, handle: int) -> str | None:
        return None

    def read_browser_text_observation(self) -> "TextObservation":
        from ..text_observation import TextObservation
        return TextObservation(
            text="", source="unavailable", target={"kind": "browser_global"},
            fresh=True, ok=False, error=self.reason,
        )


_backend: WindowBackend | None = None


def get_backend() -> WindowBackend:
    """Return the process-wide window backend for the host platform."""
    global _backend
    if _backend is not None:
        return _backend
    if sys.platform == "win32":
        from ._win32 import Win32Backend

        _backend = Win32Backend()
    elif sys.platform.startswith("linux"):
        from ._linux import LinuxBackend

        _backend = LinuxBackend()
    else:
        _backend = NullBackend(f"unsupported platform {sys.platform!r}")
    return _backend
