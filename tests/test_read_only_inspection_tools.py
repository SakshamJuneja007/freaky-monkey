from __future__ import annotations
from pathlib import Path
import pytest
from agent_control import os_tools
from agent_control.policy import Policy
from agent_control.types import Action, FailureClass, PolicyDenied


def pol(tmp_path: Path, readable: Path) -> Policy:
    return Policy(workspace=tmp_path / 'ws', readable_roots=(readable,), refuse_if_elevated=False)


def test_list_directory_permitted_and_bounded(tmp_path):
    project=tmp_path/'project'; project.mkdir()
    (project/'main.py').write_text('x')
    (project/'pkg').mkdir()
    result=os_tools.execute(pol(tmp_path, project), Action('list_directory', {'path':str(project), 'max_entries':1}))
    assert result.ok and result.detail['entry_count']==1 and result.detail['truncated'] is True


def test_list_directory_outside_root_denied(tmp_path):
    allowed=tmp_path/'allowed'; allowed.mkdir(); outside=tmp_path/'outside'; outside.mkdir()
    result=os_tools.execute(pol(tmp_path, allowed), Action('list_directory', {'path':str(outside)}))
    assert not result.ok and result.failure_class is FailureClass.PERMISSION_DENIED


def test_read_text_file_and_bounds(tmp_path):
    project=tmp_path/'project'; project.mkdir(); f=project/'a.txt'; f.write_text('one\ntwo\nthree\nfour')
    result=os_tools.execute(pol(tmp_path, project), Action('read_text_file', {'path':str(f), 'start_line':2, 'max_lines':2}))
    assert result.ok and result.detail['content']=='two\nthree' and result.detail['truncated'] is True


def test_read_text_file_refuses_env(tmp_path):
    project=tmp_path/'project'; project.mkdir(); env=project/'.env'; env.write_text('API_KEY=nope')
    result=os_tools.execute(pol(tmp_path, project), Action('read_text_file', {'path':str(env)}))
    assert not result.ok and result.failure_class is FailureClass.PERMISSION_DENIED


def test_search_finds_symbol_but_not_env(tmp_path):
    project=tmp_path/'project'; project.mkdir(); (project/'main.py').write_text('def target_symbol():\n    return 1\n')
    (project/'.env').write_text('target_symbol=SECRET')
    result=os_tools.execute(pol(tmp_path, project), Action('search_files', {'path':str(project), 'query':'target_symbol'}))
    assert result.ok and result.detail['match_count']==1
    assert result.detail['matches'][0]['path'].endswith('main.py')


def test_search_outside_root_denied(tmp_path):
    allowed=tmp_path/'allowed'; allowed.mkdir(); outside=tmp_path/'outside'; outside.mkdir(); (outside/'x.txt').write_text('needle')
    result=os_tools.execute(pol(tmp_path, allowed), Action('search_files', {'path':str(outside), 'query':'needle'}))
    assert not result.ok and result.failure_class is FailureClass.PERMISSION_DENIED


def test_search_skips_binary_and_generated_dirs(tmp_path):
    project=tmp_path/'project'; project.mkdir(); (project/'ok.txt').write_text('needle')
    (project/'bin.dat').write_bytes(b'needle\x00binary')
    (project/'.git').mkdir(); (project/'.git'/'hidden.txt').write_text('needle')
    result=os_tools.execute(pol(tmp_path, project), Action('search_files', {'path':str(project), 'query':'needle'}))
    assert result.ok and result.detail['match_count']==1


def test_policy_write_access_not_expanded(tmp_path):
    project=tmp_path/'project'; project.mkdir(); (project/'main.py').write_text('x')
    policy=pol(tmp_path, project)
    with pytest.raises(PolicyDenied):
        policy.resolve_write_path(project/'main.py')
