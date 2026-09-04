"""Windows window/input/screenshot backend via ctypes (no pywin32 dependency).

Accessibility (UI Automation) is deliberately *not* implemented for V1:
``accessibility_tree`` returns None so dependent checks resolve to UNKNOWN
rather than silently passing. That gap is the Windows analogue of the AT-SPI gap
the plan expects to measure and report honestly (plan S27).
"""

from __future__ import annotations

import ctypes
import io
from ctypes import wintypes

from . import WindowInfo

user32 = ctypes.WinDLL("user32", use_last_error=True)

_ENUM_PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

user32.EnumWindows.argtypes = [_ENUM_PROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]

_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004

_VK = {
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B,
    "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
    "backspace": 0x08, "delete": 0x2E, "space": 0x20,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
}


class Win32Backend:
    name = "win32"
    available = True

    # -- observation -------------------------------------------------------
    def list_windows(self) -> list[WindowInfo]:
        found: list[WindowInfo] = []
        foreground = user32.GetForegroundWindow()

        def callback(hwnd, _lparam):
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            rect = wintypes.RECT()
            bbox = None
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                bbox = (rect.left, rect.top, rect.right, rect.bottom)
            found.append(
                WindowInfo(
                    handle=int(hwnd),
                    title=buf.value,
                    pid=int(pid.value),
                    visible=bool(user32.IsWindowVisible(hwnd)),
                    focused=int(hwnd) == int(foreground),
                    bbox=bbox,
                )
            )
            return True

        user32.EnumWindows(_ENUM_PROC(callback), 0)
        return found

    def accessibility_tree(self, handle: int) -> dict | None:
        """Not implemented on Windows for V1 -> callers must yield UNKNOWN."""
        return None

    # -- actuation (vision-fallback path only) ------------------------------
    def focus(self, handle: int) -> bool:
        return bool(user32.SetForegroundWindow(wintypes.HWND(handle)))

    def screenshot_png(self) -> bytes | None:
        try:
            from PIL import ImageGrab
        except ImportError:
            return None
        try:
            image = ImageGrab.grab(all_screens=False)
            buf = io.BytesIO()
            image.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
        except Exception:
            return None

    def click(self, x: int, y: int) -> bool:
        if not user32.SetCursorPos(int(x), int(y)):
            return False
        user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        return True

    def type_text(self, text: str) -> bool:
        for char in text:
            code = ord(char)
            user32.keybd_event(0, code, _KEYEVENTF_UNICODE, 0)
            user32.keybd_event(0, code, _KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP, 0)
        return True

    def press_keys(self, combo: str) -> bool:
        parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
        codes: list[int] = []
        for part in parts:
            if part in _VK:
                codes.append(_VK[part])
            elif len(part) == 1:
                codes.append(ord(part.upper()))
            else:
                return False
        for code in codes:
            user32.keybd_event(code, 0, 0, 0)
        for code in reversed(codes):
            user32.keybd_event(code, 0, _KEYEVENTF_KEYUP, 0)
        return True
