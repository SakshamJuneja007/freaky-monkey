from pathlib import Path
from agent_control.persistent_memory import MemoryExtractor, PersistentMemory

def put(m,p): m.put(**{k:v for k,v in p.items() if k!='op'})
def test_01_explicit_user_preference_persists(tmp_path):
 p=tmp_path/'m.sqlite3'
 with PersistentMemory(p) as m: put(m,MemoryExtractor.extract_user('Remember that I prefer concise coding prompts.')[0])
 with PersistentMemory(p) as m:
  x=m.search('concise coding prompts')[0]; assert x.memory_type=='USER_PREFERENCE' and x.source=='user_explicit'
def test_02_survives_restart(tmp_path):
 p=tmp_path/'m.sqlite3'
 with PersistentMemory(p) as m: m.put(memory_type='USER_FACT',content=r'DEIMOS project is at D:\agentic-project',source='user_explicit',confidence=.98,key='deimos-project-location')
 with PersistentMemory(p) as m: assert 'agentic-project' in m.search('where is my DEIMOS project')[0].content
def test_03_relevant_retrieval(tmp_path):
 with PersistentMemory(tmp_path/'m.sqlite3') as m:
  m.put(memory_type='USER_FACT',content=r'DEIMOS project is at D:\agentic-project',source='user_explicit',confidence=.98,key='deimos-project-location'); assert m.search('where is DEIMOS project')
def test_04_irrelevant_filtered(tmp_path):
 with PersistentMemory(tmp_path/'m.sqlite3') as m:
  m.put(memory_type='USER_FACT',content=r'user music folder is D:\Music',source='user_explicit',confidence=.98); assert m.search('DEIMOS project location')==[]
def test_05_duplicate_deduplicated(tmp_path):
 with PersistentMemory(tmp_path/'m.sqlite3') as m:
  a=m.put(memory_type='USER_FACT',content=r'DEIMOS project is at D:\agentic-project',source='user_explicit',confidence=.98,key='deimos-project-location'); b=m.put(memory_type='USER_FACT',content=r'DEIMOS project is at D:\agentic-project',source='user_explicit',confidence=.98,key='deimos-project-location'); assert a.id==b.id and m.status()['active']==1
def test_06_conflict_supersedes(tmp_path):
 with PersistentMemory(tmp_path/'m.sqlite3') as m:
  old=m.put(memory_type='USER_FACT',content=r'DEIMOS project is at D:\agentic-project',source='user_explicit',confidence=.98,key='deimos-project-location'); new=m.put(memory_type='USER_FACT',content=r'DEIMOS project is at D:\new-agentic-project',source='user_explicit',confidence=.98,key='deimos-project-location'); assert m.get(old.id).status=='SUPERSEDED' and m.get(new.id).status=='ACTIVE' and [x.content for x in m.search('DEIMOS project location')]==[new.content]
def test_07_invalidated_not_retrieved(tmp_path):
 with PersistentMemory(tmp_path/'m.sqlite3') as m:
  x=m.put(memory_type='USER_FACT',content=r'DEIMOS project is at D:\agentic-project',source='user_explicit',confidence=.98,key='deimos-project-location'); assert m.invalidate(x.id); assert m.search('DEIMOS project')==[]
def test_08_memory_does_not_mutate_live_state(tmp_path):
 with PersistentMemory(tmp_path/'m.sqlite3') as m:
  m.put(memory_type='ENVIRONMENT_FACT',content='Chrome is on HiAnime',source='verified_observation',confidence=.98); assert m.search('Chrome page')[0].content=='Chrome is on HiAnime' and not hasattr(m,'execute')
def test_09_memory_has_no_approval_authority(tmp_path):
 with PersistentMemory(tmp_path/'m.sqlite3') as m: assert not hasattr(m,'approve')
def test_10_memory_has_no_verification_authority(tmp_path):
 with PersistentMemory(tmp_path/'m.sqlite3') as m: assert not hasattr(m,'verify')
def test_11_success_episode():
 x=MemoryExtractor.outcome(goal='open notepad and type hello',task_id='task-1',status='SUCCESS',verified='PASS',duration=1.2); assert x['memory_type']=='TASK_EPISODE' and len(x['content'])<250 and x['source']=='verified_task_outcome'
def test_12_failure_episode():
 x=MemoryExtractor.outcome(goal='search Chrome',task_id='task-2',status='FAILED',verified='FAIL',failure='browser observation unavailable'); assert x['memory_type']=='FAILURE_EPISODE' and 'did not complete' in x['content']
def test_13_transient_commands_ignored(): assert MemoryExtractor.extract_user('open chrome')==[] and MemoryExtractor.extract_user('play do i wanna know')==[]
def test_14_secrets_filtered(tmp_path):
 with PersistentMemory(tmp_path/'m.sqlite3') as m: assert m.put(memory_type='USER_FACT',content='my api_key=sk-abcdefghijklmnopqrstuvwxyz',source='user_explicit',confidence=.98) is None and m.status()['active']==0
def test_15_retrieval_bounded(tmp_path):
 with PersistentMemory(tmp_path/'m.sqlite3') as m:
  [m.put(memory_type='USER_FACT',content=f'DEIMOS project fact {i}',source='user_explicit',confidence=.9,key=f'fact-{i}') for i in range(40)]; assert len(m.search('DEIMOS project',limit=100))<=8
def test_16_file_location_memory_separate():
 from agent_control import memory as fm
 assert hasattr(fm,'FileMemory') and fm.DEFAULT_STORE != Path('.agent_memory')/'agent_memory.sqlite3'
def test_17_session_has_memory_hooks():
 from agent_control.session import Session
 assert hasattr(Session,'_persistent_memory') and hasattr(Session,'_extract_user_memory') and hasattr(Session,'_remember_task_outcome')
def test_18_browser_context_is_data(tmp_path):
 with PersistentMemory(tmp_path/'m.sqlite3') as m:
  m.put(memory_type='CONTEXT_ENTITY',content='the current project refers to DEIMOS',source='user_explicit',confidence=.98); x=m.search('current project')[0]; assert x.memory_type=='CONTEXT_ENTITY'
def test_19_project_move_is_current(tmp_path):
 p=tmp_path/'m.sqlite3'
 with PersistentMemory(p) as m: put(m,MemoryExtractor.extract_user(r'My DEIMOS project moved to D:\new-agentic-project')[0])
 with PersistentMemory(p) as m: assert len(m.search('where is my DEIMOS project'))==1 and 'new-agentic-project' in m.search('where is my DEIMOS project')[0].content
def test_20_superseded_history_retained(tmp_path):
 with PersistentMemory(tmp_path/'m.sqlite3') as m:
  old=m.put(memory_type='USER_PREFERENCE',content='user prefers dark mode',source='user_explicit',confidence=.98,key='display-preference'); new=m.put(memory_type='USER_PREFERENCE',content='user prefers light mode',source='user_explicit',confidence=.98,key='display-preference'); assert m.get(old.id).status=='SUPERSEDED' and m.get(new.id).status=='ACTIVE' and m.status()['superseded']==1

def test_21_session_extracts_and_recalls_across_instances(tmp_path, monkeypatch):
    monkeypatch.setenv('DEIMOS_AGENT_MEMORY', str(tmp_path/'agent.sqlite3'))
    from agent_control.response import Narrator
    from agent_control.session import Session
    out=[]
    s1=Session(narrator=Narrator(speaker=None, write=out.append), planner='mock')
    s1._extract_user_memory(r'Remember that my DEIMOS project is in D:\agentic-project.')
    s2=Session(narrator=Narrator(speaker=None, write=out.append), planner='mock')
    found=s2._persistent_memory_records('Where is my DEIMOS project?')
    assert found and 'agentic-project' in found[0]['content']
    s1._persistent_memory().close(); s2._persistent_memory().close()
