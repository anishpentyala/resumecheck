"""New bundles must not contain a verdict from another comparison scope."""
from dataclasses import asdict, replace
import json
import math

import pytest
from resumecheck.artifacts import save_comparison_artifacts, replay_comparison_artifacts
from resumecheck.compare import compare_trajectories
from resumecheck.recorder import Trajectory, StepRecord


def pair(state=False):
    a=Trajectory('full',[StepRecord(1,loss=1.,model_fingerprint='v2:a' if state else None)])
    b=Trajectory('resumed',[StepRecord(1,loss=2.,model_fingerprint='v2:b' if state else None)])
    return a,b


@pytest.mark.parametrize('state',[False,True])
@pytest.mark.parametrize('change',['passed','steps_compared','first_divergence','missing_fields'])
def test_any_report_disagreement_rejected_before_destination(tmp_path,state,change):
    a,b=pair(state); expected=compare_trajectories(a,b,rtol=0,atol=0)
    values={'passed':True,'summary':'PASS: stale report','steps_compared':99,
            'first_divergence':None,'missing_fields':['not the saved comparison']}
    report=replace(expected,**{change:values[change]})
    path=tmp_path/'bundle'
    with pytest.raises(ValueError,match='report.*scope'):
        save_comparison_artifacts(path,a,b,report,rtol=0,atol=0)
    assert not path.exists()


@pytest.mark.parametrize('state',[False,True])
def test_report_from_relaxed_tolerance_cannot_be_saved_under_strict_settings(tmp_path,state):
    a,b=pair(state)
    if state: b.steps[0].model_fingerprint='v2:a'
    report=compare_trajectories(a,b,rtol=2,atol=0)
    assert report.passed
    with pytest.raises(ValueError,match='report'):
        save_comparison_artifacts(tmp_path/'bundle',a,b,report,rtol=0,atol=0)
    assert not (tmp_path/'bundle').exists()


@pytest.mark.parametrize('value',[True,'0',float('nan'),float('inf'),-1,10**400])
@pytest.mark.parametrize('name',['rtol','atol'])
def test_bad_saved_tolerance_never_creates_unreplayable_bundle(tmp_path,value,name):
    a,b=pair(); report=compare_trajectories(a,b)
    settings={'rtol':1e-5,'atol':1e-7};settings[name]=value
    with pytest.raises(ValueError,match=name):
        save_comparison_artifacts(tmp_path/'bundle',a,b,report,**settings)
    assert not (tmp_path/'bundle').exists()


@pytest.mark.parametrize('left,right',[(1.,1.),(1.,2.),(float('nan'),1.),
                                     (float('nan'),float('nan')),(float('inf'),float('inf'))])
@pytest.mark.parametrize('state',[False,True])
def test_valid_nonfinite_and_finite_reports_replay_without_nan_equality_trap(tmp_path,left,right,state):
    a,b=pair(state);a.steps[0].loss=left;b.steps[0].loss=right
    if state: b.steps[0].model_fingerprint='v2:a'
    report=compare_trajectories(a,b,rtol=0,atol=0)
    path=tmp_path/'bundle';save_comparison_artifacts(path,a,b,report,rtol=0,atol=0)
    replay=replay_comparison_artifacts(path)
    assert json.dumps(asdict(replay),sort_keys=True)==json.dumps(asdict(report),sort_keys=True)
    assert json.loads((path/'manifest.json').read_text())['schema_version']==(2 if state else 1)


def test_wrong_typed_pass_flag_cannot_match_boolean_by_python_equality(tmp_path):
    a,b=pair();b.steps[0].loss=1.
    report=replace(compare_trajectories(a,b,rtol=0,atol=0),passed=1)
    with pytest.raises(ValueError,match='report'):
        save_comparison_artifacts(tmp_path/'bundle',a,b,report,rtol=0,atol=0)
    assert not (tmp_path/'bundle').exists()


def test_stale_summary_replaced_with_saved_settings_without_mutating_caller(tmp_path):
    a,b=pair();b.steps[0].loss=1.
    report=replace(compare_trajectories(a,b),summary='PASS: stale tolerance wording')
    path=tmp_path/'bundle';save_comparison_artifacts(path,a,b,report,rtol=0,atol=0)
    saved=json.loads((path/'report.json').read_text())
    assert saved['summary']==compare_trajectories(a,b,rtol=0,atol=0).summary
    assert report.summary=='PASS: stale tolerance wording'
