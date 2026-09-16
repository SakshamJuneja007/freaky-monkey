"""Windows window/input/screenshot backend via ctypes (no pywin32 dependency).

Native text observation uses Windows UI Automation first, with a narrow Win32
window-text fallback. Accessibility-tree export remains separate and may still
return unavailable; text verification does not treat that gap as success.
"""

from __future__ import annotations

import ctypes
import io
import base64
import shutil
import subprocess
from ctypes import wintypes

from . import WindowInfo
from ..text_observation import TextObservation

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
user32.EnumChildWindows.argtypes = [_ENUM_PROC, wintypes.LPARAM]
user32.EnumChildWindows.restype = wintypes.BOOL

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
                    focused=foreground is not None and int(hwnd) == int(foreground),
                    bbox=bbox,
                )
            )
            return True

        user32.EnumWindows(_ENUM_PROC(callback), 0)
        return found

    def accessibility_tree(self, handle: int) -> dict | None:
        """Not implemented on Windows for V1 -> callers must yield UNKNOWN."""
        return None

    def validate_window_target(self, handle: int, *, pid: int | None = None, title_contains: str | None = None) -> bool:
        """Cheap identity check for a cached window target.

        This validates the handle's current existence, process identity,
        visibility, and title. It never reads application content, so it is
        target metadata validation rather than cached state or verification.
        """
        hwnd = wintypes.HWND(int(handle))
        if not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd):
            return False
        if pid is not None:
            current_pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(current_pid))
            if int(current_pid.value) != int(pid):
                return False
        if title_contains:
            length = int(user32.GetWindowTextLengthW(hwnd))
            buf = ctypes.create_unicode_buffer(max(1, length + 1))
            user32.GetWindowTextW(hwnd, buf, len(buf))
            if title_contains.casefold() not in buf.value.casefold():
                return False
        return True

    def read_browser_text_observation(self) -> TextObservation:
        """Read editable text from the currently visible browser UI globally.

        Uses Windows UI Automation only. This is read-only and intentionally
        independent of BrowserSkill executor results.
        """
        import json
        import time

        started = time.time()
        shell = shutil.which("powershell") or shutil.which("pwsh")
        target = {"kind": "browser_global"}
        if not shell:
            return TextObservation("", "unavailable", target, True, started, {}, False,
                                    "PowerShell is unavailable for Windows UI Automation")

        script = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$browserNames = @('chrome.exe', 'msedge.exe', 'brave.exe', 'chromium.exe')
$editType = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    [System.Windows.Automation.ControlType]::Edit)
$comboType = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    [System.Windows.Automation.ControlType]::ComboBox)
$condition = New-Object System.Windows.Automation.OrCondition($editType, $comboType)
function ReadValue($el) {
    try {
        $pattern = $null
        if ($el.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$pattern)) {
            $value = [string]$pattern.Current.Value
            if ($value) { return $value }
        }
    } catch {}
    try {
        $pattern = $null
        if ($el.TryGetCurrentPattern([System.Windows.Automation.TextPattern]::Pattern, [ref]$pattern)) {
            $value = [string]$pattern.DocumentRange.GetText(-1)
            if ($value) { return $value }
        }
    } catch {}
    return ''
}
$focused = [System.Windows.Automation.AutomationElement]::FocusedElement
$focusedPid = 0
try { $focusedPid = [int]$focused.Current.ProcessId } catch {}
$items = New-Object System.Collections.Generic.List[object]
try {
    $value = ReadValue $focused
    if ($value) {
        $c = $focused.Current
        $items.Add([pscustomobject]@{ text=$value.Trim(); role=[string]$c.ControlType.ProgrammaticName; name=[string]$c.Name; automation_id=[string]$c.AutomationId; class_name=[string]$c.ClassName; process_id=[int]$c.ProcessId; focused=$true; source='uia_focused_edit' })
    }
} catch {}
foreach ($win in [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
    [System.Windows.Automation.TreeScope]::Children,
    [System.Windows.Automation.Condition]::TrueCondition)) {
    try {
        $pid = [int]$win.Current.ProcessId
        if ($pid -le 0) { continue }
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($null -eq $proc -or $browserNames -notcontains ($proc.ProcessName.ToLower() + '.exe')) { continue }
        $controls = $win.FindAll([System.Windows.Automation.TreeScope]::Descendants, $condition)
        foreach ($el in $controls) {
            $value = ReadValue $el
            if (-not $value) { continue }
            $c = $el.Current
            $items.Add([pscustomobject]@{ text=$value.Trim(); role=[string]$c.ControlType.ProgrammaticName; name=[string]$c.Name; automation_id=[string]$c.AutomationId; class_name=[string]$c.ClassName; process_id=[int]$c.ProcessId; focused=([int]$c.ProcessId -eq $focusedPid); source='uia_browser_edit' })
            if ($items.Count -ge 64) { break }
        }
    } catch {}
    if ($items.Count -ge 64) { break }
}
[pscustomobject]@{ browser_controls=@($items); focused_process_id=$focusedPid } | ConvertTo-Json -Compress -Depth 6
"""
        try:
            proc = subprocess.run(
                [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-STA", "-Command", script],
                capture_output=True, timeout=10, check=False, text=True, encoding="utf-8", errors="replace",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return TextObservation("", "unavailable", target, False, started, {}, False, f"{type(exc).__name__}: {exc}")
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if proc.returncode != 0 or not lines:
            return TextObservation("", "unavailable", target, True, started, {}, False, proc.stderr.strip() or "Windows UI Automation returned no browser text")
        try:
            payload = json.loads(lines[-1])
        except (ValueError, TypeError) as exc:
            return TextObservation("", "unavailable", target, True, started, {}, False, f"invalid UI Automation response: {exc}")
        controls = payload.get("browser_controls", []) if isinstance(payload, dict) else []
        if isinstance(controls, dict):
            controls = [controls]
        normalized, values, seen = [], [], set()
        for item in controls if isinstance(controls, list) else []:
            if not isinstance(item, dict):
                continue
            value = str(item.get("text") or "").strip()
            if not value:
                continue
            key = (int(item.get("process_id") or 0), value, str(item.get("automation_id") or ""))
            if key in seen:
                continue
            seen.add(key); normalized.append(item); values.append(value)
        return TextObservation(
            text="\n".join(dict.fromkeys(values)), source="win32_browser_uia", target=target,
            fresh=True, observed_at=started,
            metadata={"controls": normalized, "focused_process_id": payload.get("focused_process_id") if isinstance(payload, dict) else None,
                      "control_count": len(normalized), "editable_text_available": bool(values)}, ok=True,
        )

    def read_text_observation(self, handle: int, *, target: dict | None = None) -> TextObservation:
        """Read semantic application text, preferring Windows UI Automation.

        Target discovery is deliberately scoped to one application window: the
        focused UIA element, its ancestors, and editable/document descendants.
        No coordinates, screenshots, OCR, or requested-text hints are involved.
        """
        import time
        started = time.time()
        descriptor = {"kind": "window", "handle": int(handle), **(target or {})}

        shell = shutil.which("powershell") or shutil.which("pwsh")
        if shell:
            result = self._uia_text_observation(shell, int(handle), descriptor, started)
            if result is not None:
                return result

        chunks: list[dict[str, object]] = []

        def collect(hwnd, _lparam):
            length = int(user32.GetWindowTextLengthW(hwnd))
            if length <= 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if buf.value:
                chunks.append({"handle": int(hwnd), "text": buf.value})
            return True

        Enum = _ENUM_PROC(collect)
        user32.EnumChildWindows(wintypes.HWND(handle), Enum, 0)
        if chunks:
            return TextObservation(
                text="\n".join(str(item["text"]) for item in chunks),
                source="win32_window_text",
                target=descriptor,
                fresh=True,
                observed_at=started,
                metadata={"backend": self.name, "controls": chunks[:20]},
            )

        return TextObservation(
            text="", source="unavailable", target=descriptor, fresh=True,
            observed_at=started, metadata={"backend": self.name}, ok=False,
            error="Windows UI Automation and Win32 text did not expose readable application text",
        )

    def _uia_text_observation(self, shell: str, handle: int, target: dict, started: float) -> TextObservation | None:
        """Ask inbox UIAutomationClient for one semantically relevant control."""
        script = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$root = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]HANDLE)
if ($null -eq $root) { exit 2 }

function PatternText($el) {
    try {
        $pattern = $null
        if ($el.TryGetCurrentPattern([System.Windows.Automation.TextPattern]::Pattern, [ref]$pattern)) {
            $text = $pattern.DocumentRange.GetText(-1)
            if ($null -ne $text -and $text.Length -gt 0) { return @('uia_text_pattern', $text) }
        }
    } catch {}
    try {
        $pattern = $null
        if ($el.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$pattern)) {
            $text = $pattern.Current.Value
            if ($null -ne $text -and $text.Length -gt 0) { return @('uia_value_pattern', $text) }
        }
    } catch {}
    return $null
}

function Candidate($el, [bool]$focused) {
    if ($null -eq $el) { return $null }
    $hit = PatternText $el
    if ($null -eq $hit) { return $null }
    $c = $el.Current
    return [pscustomobject]@{
        source = [string]$hit[0]
        text = [string]$hit[1]
        focused = $focused
        control_type = [string]$c.ControlType.ProgrammaticName
        name = [string]$c.Name
        automation_id = [string]$c.AutomationId
        class_name = [string]$c.ClassName
        process_id = [int]$c.ProcessId
    }
}

$candidates = New-Object System.Collections.Generic.List[object]
try {
    $focused = [System.Windows.Automation.AutomationElement]::FocusedElement
    $ancestor = $focused
    for ($i = 0; $i -lt 8 -and $null -ne $ancestor; $i++) {
        $isSame = $false
        try { $isSame = $ancestor.Current.NativeWindowHandle -eq HANDLE } catch {}
        $hit = Candidate $ancestor $true
        if ($null -ne $hit) { [void]$candidates.Add($hit) }
        if ($isSame) { break }
        try { $ancestor = [System.Windows.Automation.TreeWalker]::ControlViewWalker.GetParent($ancestor) } catch { break }
    }
} catch {}

$edit = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    [System.Windows.Automation.ControlType]::Edit)
$document = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    [System.Windows.Automation.ControlType]::Document)
$condition = New-Object System.Windows.Automation.OrCondition($edit, $document)
try {
    $controls = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $condition)
    foreach ($el in $controls) {
        $hit = Candidate $el $false
        if ($null -ne $hit) { [void]$candidates.Add($hit) }
        if ($candidates.Count -ge 32) { break }
    }
} catch {}

$chosen = $candidates | Select-Object -First 1
if ($null -ne $chosen) { $chosen | ConvertTo-Json -Compress }
""".replace("HANDLE", str(int(handle)))
        try:
            proc = subprocess.run(
                [shell, "-NoLogo", "-NoProfile", "-NonInteractive",
                 "-ExecutionPolicy", "Bypass", "-STA", "-Command", script],
                capture_output=True, timeout=10, check=False, text=True,
                encoding="utf-8", errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return None
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if proc.returncode != 0 or not lines:
            return None
        try:
            import json
            data = json.loads(lines[-1])
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict) or not data.get("text"):
            return None
        return TextObservation(
            text=str(data["text"]), source=str(data.get("source") or "uia_text_pattern"),
            target=target, fresh=True, observed_at=started,
            metadata={
                "backend": self.name, "control_type": data.get("control_type"),
                "name": data.get("name"), "automation_id": data.get("automation_id"),
                "class_name": data.get("class_name"), "process_id": data.get("process_id"),
                "focused": bool(data.get("focused")),
            },
        )

    def read_text(self, handle: int) -> str | None:
        """Compatibility text accessor backed by the universal reader."""
        observation = self.read_text_observation(int(handle))
        return observation.text if observation.ok else None

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
