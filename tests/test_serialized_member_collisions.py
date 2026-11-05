"""Writer must not publish JSON that its duplicate-safe reader rejects."""
import copy
import json
from pathlib import Path
import pytest
from resumecheck.artifacts import save_comparison_artifacts,replay_comparison_artifacts
from resumecheck.compare import compare_trajectories
from resumecheck.recorder import StepRecord,Trajectory

PAIRS=[(1,'1'),(None,'null'),(True,'true'),(False,'false'),(1.5,'1.5')]


def trajectory(a,b,nested):
    payload={a:'SYNTHETIC_PRIVATE_MEMBER',b:'SYNTHETIC_PRIVATE_VALUE'}
    meta={'outer':[{'inner':payload}]} if nested else payload
    return Trajectory('x',[StepRecord(0,loss=1.)],meta=meta)

@pytest.mark.parametrize('a,b',PAIRS)
@pytest.mark.parametrize('nested',[False,True])
def test_trajectory_save_rejects_wire_collision_before_truncating(tmp_path,a,b,nested):
    value=trajectory(a,b,nested)
    path=tmp_path/'saved.json';path.write_text('KEEP EXISTING')
    with pytest.raises(ValueError,match='duplicate JSON object member') as error:
        value.save(path)
    assert path.read_text()=='KEEP EXISTING'
    assert 'SYNTHETIC_PRIVATE' not in str(error.value)

@pytest.mark.parametrize('a,b',PAIRS)
@pytest.mark.parametrize('side',['left','right'])
def test_bundle_rejects_collision_before_creating_destination(tmp_path,a,b,side):
    bad=trajectory(a,b,True);good=Trajectory('good',[StepRecord(0,loss=1.)])
    left,right=(bad,good) if side=='left' else (good,bad)
    report=compare_trajectories(left,right);dest=tmp_path/'bundle'
    with pytest.raises(ValueError,match='duplicate JSON object member'):
        save_comparison_artifacts(dest,left,right,report,rtol=1e-5,atol=1e-7)
    assert not dest.exists()

@pytest.mark.parametrize('meta',[{1:'a','2':'b'},{None:'a','other':'b'},
                               {'a':[{True:'x'}, {'true':'y'}]}])
def test_noncolliding_coerced_keys_keep_existing_roundtrip(tmp_path,meta):
    value=Trajectory('x',[StepRecord(0,loss=1.)],meta=meta)
    path=tmp_path/'trajectory.json';value.save(path)
    expected=json.loads(json.dumps(meta))
    assert Trajectory.load(path).meta==expected
    bundle=tmp_path/'bundle'
    save_comparison_artifacts(bundle,value,value,compare_trajectories(value,value),rtol=1e-5,atol=1e-7)
    assert replay_comparison_artifacts(bundle).passed


def test_new_path_is_not_created_on_serialized_collision(tmp_path):
    path=tmp_path/'new.json'
    with pytest.raises(ValueError):trajectory(1,'1',False).save(path)
    assert not path.exists()


def test_selected_extra_collision_cannot_publish_ambiguous_trajectory(tmp_path):
    value=Trajectory('x',[StepRecord(0,loss=1.,extras={1:2.,'1':2.})])
    report=compare_trajectories(value,value,check_extras=['1'])
    assert report.passed
    with pytest.raises(ValueError,match='duplicate JSON object member'):
        save_comparison_artifacts(tmp_path/'bundle',value,value,report,rtol=1e-5,atol=1e-7,check_extras=['1'])
    assert not (tmp_path/'bundle').exists()


def test_valid_payload_preserves_existing_format_and_input(tmp_path):
    value=Trajectory('x',[StepRecord(0,loss=float('nan'),extras={'lr':.1})],meta={'seed':7})
    before=copy.deepcopy(value.meta)
    path=tmp_path/'new.json';value.save(path)
    text=path.read_text()
    assert 'NaN' in text
    assert value.meta==before
    assert Trajectory.load(path).meta==before
