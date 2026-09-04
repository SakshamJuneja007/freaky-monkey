"""Linux window/accessibility backend (plan S5, S13) -- seam target, not V1 work.

The plan names AT-SPI as the Linux accessibility source. This backend reports
``available`` from what is actually importable/installed, so running the same
benchmark on Linux degrades to UNKNOWN checks instead of silently passing.

Fill in ``accessibility_tree`` with pyatspi when the Linux reference environment
comes online; nothing above this module needs to change.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from . import WindowInfo


class LinuxBackend:
    name = "linux"

    def __init__(self) -> None:
        self._wmctrl = shutil.which("wmctrl")
        self._xdotool = shutil.which("xdotool")
        self._has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        self.available = bool(self._has_display and (self._wmctrl or self._xdotool))

    def list_windows(self) -> list[WindowInfo]:
        if not (self.available and self._wmctrl):
            return []
        try:
            out = subprocess.run(
                [self._wmctrl, "-lp"], capture_output=True, text=True, timeout=5, check=False
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        windows: list[WindowInfo] = []
        for line in out.splitlines():
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            handle_s, _desktop, pid_s, _host, title = parts
            try:
                handle, pid = int(handle_s, 16), int(pid_s)
            except ValueError:
                continue
            windows.append(
                WindowInfo(handle=handle, title=title, pid=pid, visible=True, focused=False)
            )
        return windows

    def accessibility_tree(self, handle: int) -> dict | None:
        """AT-SPI hook. Returns None until the Linux reference env is stood up."""
        return None

    def focus(self, handle: int) -> bool:
        if not (self.available and self._wmctrl):
            return False
        rc = subprocess.run(
            [self._wmctrl, "-i", "-a", hex(handle)], capture_output=True, timeout=5, check=False
        ).returncode
        return rc == 0

    def screenshot_png(self) -> bytes | None:
        if not self._has_display:
            return None
        for tool, argv in (
            ("gnome-screenshot", ["-f", "-"]),
            ("import", ["-window", "root", "png:-"]),
        ):
            path = shutil.which(tool)
            if not path:
                continue
            try:
                proc = subprocess.run([path, *argv], capture_output=True, timeout=15, check=False)
            except (OSError, subprocess.SubprocessError):
                continue
            if proc.returncode == 0 and proc.stdout[:8] == b"\x89PNG\r\n\x1a\n":
                return proc.stdout
        return None

    def click(self, x: int, y: int) -> bool:
        return self._xdo(["mousemove", str(x), str(y), "click", "1"])

    def type_text(self, text: str) -> bool:
        return self._xdo(["type", "--delay", "12", text])

    def press_keys(self, combo: str) -> bool:
        return self._xdo(["key", combo.replace("+", "+")])

    def _xdo(self, argv: list[str]) -> bool:
        if not (self.available and self._xdotool):
            return False
        try:
            return subprocess.run(
                [self._xdotool, *argv], capture_output=True, timeout=10, check=False
            ).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
