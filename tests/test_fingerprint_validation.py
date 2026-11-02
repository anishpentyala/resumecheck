"""Malformed present fingerprint evidence must fail, not raise or disappear."""
import json
import pytest
from resumecheck.recorder import Trajectory,StepRecord
from resumecheck.compare import compare_trajectories

@pytest.mark.parametrize('value',[True,False,0,1,1.5,[],{},['v2:abc']])
@pytest.mark.parametrize('field',['batch_fingerprint','model_fingerprint'])
@pytest.mark.parametrize('side',['both','a','b'])
def test_bad_fingerprint_types_fail_without_exception(value,field,side):
    a=StepRecord(1,loss=1.0);b=StepRecord(1,loss=1.0)
    if side in ('both','a'):setattr(a,field,value)
    if side in ('both','b'):setattr(b,field,value)
    report=compare_trajectories(Trajectory('a',[a]),Trajectory('b',[b]))
    assert report.passed is False
    assert any(field in message for message in report.missing_fields)

@pytest.mark.parametrize('field',['batch_fingerprint','model_fingerprint'])
def test_malformed_loaded_fingerprint_fails(tmp_path,field):
    path=tmp_path/'trajectory.json'
    path.write_text(json.dumps({'name':'x','steps':[{'step':1,'loss':1.0,field:True}]}))
    trajectory=Trajectory.load(path)
    report=compare_trajectories(trajectory,trajectory)
    assert report.passed is False

@pytest.mark.parametrize('field',['batch_fingerprint','model_fingerprint'])
def test_disabled_fingerprint_check_remains_explicit(field):
    trajectory=Trajectory('x',[StepRecord(1,loss=1.0,**{field:True})])
    flag='check_batch' if field=='batch_fingerprint' else 'check_model_state'
    assert compare_trajectories(trajectory,trajectory,**{flag:False}).passed

@pytest.mark.parametrize('field',['batch_fingerprint','model_fingerprint'])
def test_opaque_versioned_labels_remain_supported(field):
    trajectory=Trajectory('x',[StepRecord(1,loss=1.0,**{field:'v2:caller-label'})])
    assert compare_trajectories(trajectory,trajectory).passed
