from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.skipif(__import__('sys').platform != 'win32', reason='Win32 backend requires Windows')


def _backend():
    mod = importlib.import_module('agent_control.platform_window._win32')
    return mod, mod.Win32Backend()


def test_list_windows_marks_only_actual_foreground(monkeypatch):
    mod, backend = _backend()
    windows = [101, 202, 303]
    monkeypatch.setattr(mod.user32, 'GetForegroundWindow', lambda: 202)
    monkeypatch.setattr(mod.user32, 'GetWindowTextLengthW', lambda hwnd: 1)
    monkeypatch.setattr(mod.user32, 'GetWindowTextW', lambda hwnd, buf, size: setattr(buf, 'value', f'Window {hwnd}') or 1)
    monkeypatch.setattr(mod.user32, 'GetWindowThreadProcessId', lambda hwnd, pid: setattr(pid, 'value', hwnd))
    monkeypatch.setattr(mod.user32, 'GetWindowRect', lambda hwnd, rect: False)
    monkeypatch.setattr(mod.user32, 'IsWindowVisible', lambda hwnd: True)

    def enum_windows(callback, lparam):
        for hwnd in windows:
            assert callback(hwnd, lparam)
        return True

    monkeypatch.setattr(mod.user32, 'EnumWindows', enum_windows)
    found = backend.list_windows()
    assert [w.handle for w in found] == windows
    assert [w.focused for w in found] == [False, True, False]


def test_list_windows_with_null_foreground_completes_without_callback_exception(monkeypatch):
    mod, backend = _backend()
    windows = [11, 22, 33]
    monkeypatch.setattr(mod.user32, 'GetForegroundWindow', lambda: None)
    monkeypatch.setattr(mod.user32, 'GetWindowTextLengthW', lambda hwnd: 1)
    monkeypatch.setattr(mod.user32, 'GetWindowTextW', lambda hwnd, buf, size: setattr(buf, 'value', f'Window {hwnd}') or 1)
    monkeypatch.setattr(mod.user32, 'GetWindowThreadProcessId', lambda hwnd, pid: setattr(pid, 'value', hwnd))
    monkeypatch.setattr(mod.user32, 'GetWindowRect', lambda hwnd, rect: False)
    monkeypatch.setattr(mod.user32, 'IsWindowVisible', lambda hwnd: True)

    def enum_windows(callback, lparam):
        for hwnd in windows:
            assert callback(hwnd, lparam)
        return True

    monkeypatch.setattr(mod.user32, 'EnumWindows', enum_windows)
    found = backend.list_windows()
    assert [w.handle for w in found] == windows
    assert all(not w.focused for w in found)


def test_uia_text_pattern_observation_is_structured(monkeypatch):
    mod, backend = _backend()
    monkeypatch.setattr(mod.shutil, 'which', lambda name: 'powershell.exe')
    payload = {
        'source': 'uia_text_pattern',
        'text': 'hello\nworld',
        'focused': True,
        'control_type': 'ControlType.Document',
        'name': 'Document',
        'automation_id': 'editor',
        'class_name': 'RichEdit',
        'process_id': 123,
    }
    monkeypatch.setattr(mod.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps(payload) + '\n', stderr=''))
    monkeypatch.setattr(mod.user32, 'EnumChildWindows', lambda callback, lparam: True)

    result = backend.read_text_observation(1234, target={'kind': 'window', 'app': 'editor'})
    assert result.ok is True
    assert result.source == 'uia_text_pattern'
    assert result.text == 'hello\nworld'
    assert result.target['handle'] == 1234
    assert result.metadata['control_type'] == 'ControlType.Document'
    assert result.metadata['focused'] is True


def test_uia_value_pattern_observation_is_supported(monkeypatch):
    mod, backend = _backend()
    monkeypatch.setattr(mod.shutil, 'which', lambda name: 'powershell.exe')
    payload = {
        'source': 'uia_value_pattern',
        'text': 'search value',
        'focused': True,
        'control_type': 'ControlType.Edit',
        'name': 'Search',
        'automation_id': 'searchBox',
        'class_name': 'Edit',
        'process_id': 456,
    }
    monkeypatch.setattr(mod.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps(payload) + '\n', stderr=''))
    monkeypatch.setattr(mod.user32, 'EnumChildWindows', lambda callback, lparam: True)

    result = backend.read_text_observation(5678)
    assert result.ok is True
    assert result.source == 'uia_value_pattern'
    assert result.text == 'search value'
