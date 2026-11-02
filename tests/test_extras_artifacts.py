"""Persist the actual selected numeric evidence, not a weaker replay."""
import json
from dataclasses import asdict
import pytest
from resumecheck.artifacts import save_comparison_artifacts, replay_comparison_artifacts
from resumecheck.compare import compare_trajectories
from resumecheck.recorder import TrajectoryRecorder, fingerprint_tensor


def pair(diverge=False, state=False):
    a,b=TrajectoryRecorder('full'),TrajectoryRecorder('resumed')
    for i in range(1,4):
        a.record(i,loss=1.,lr=.1,momentum=.9)
        b.record(i,loss=1.,lr=.2 if diverge and i==2 else .1,momentum=.9)
        if state:
            a.trajectory.steps[-1].model_fingerprint=b.trajectory.steps[-1].model_fingerprint=fingerprint_tensor({'weight':[1.,2.]})
    return a.trajectory,b.trajectory


def save(tmp_path, diverge=False, state=False):
    a,b=pair(diverge,state)
    report=compare_trajectories(a,b,check_extras=['lr','momentum'])
    path=tmp_path/'bundle'
    save_comparison_artifacts(path,a,b,report,rtol=1e-5,atol=1e-7,check_extras=['lr','momentum'])
    return path,report


@pytest.mark.parametrize('diverge',[False,True])
@pytest.mark.parametrize('state',[False,True])
def test_replay_preserves_selected_scope(tmp_path,diverge,state):
    path,report=save(tmp_path,diverge,state)
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['schema_version']==3
    assert manifest['comparison']['check_extras']==['lr','momentum']
    assert manifest['comparison']['check_model_state'] is state
    replay=replay_comparison_artifacts(path)
    assert asdict(replay)==asdict(report)
    assert replay.passed is not diverge
    if diverge: assert replay.first_divergence.field=='extras.lr'


@pytest.mark.parametrize('names',['lr',[1],[''],['lr','lr'],{},False])
def test_invalid_selection_never_creates_destination(tmp_path,names):
    a,b=pair()
    with pytest.raises((TypeError,ValueError)):
        save_comparison_artifacts(tmp_path/'bundle',a,b,compare_trajectories(a,b),rtol=0,atol=0,check_extras=names)
    assert not (tmp_path/'bundle').exists()


@pytest.mark.parametrize('names',[None,[]])
def test_empty_selection_preserves_legacy_schema(tmp_path,names):
    a,b=pair(True)
    report=compare_trajectories(a,b)
    path=tmp_path/'bundle'
    save_comparison_artifacts(path,a,b,report,rtol=0,atol=0,check_extras=names)
    m=json.loads((path/'manifest.json').read_text())
    assert m['schema_version']==1 and 'check_extras' not in m['comparison']
    assert replay_comparison_artifacts(path).passed


@pytest.mark.parametrize('names',[[],['lr','lr'],[''],['lr',False],'lr',None,{}])
def test_bad_saved_selection_rejected(tmp_path,names):
    path,_=save(tmp_path)
    m=json.loads((path/'manifest.json').read_text());m['comparison']['check_extras']=names
    (path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError): replay_comparison_artifacts(path)


def test_report_mismatched_scope_rejected_before_writing(tmp_path):
    a,b=pair(True)
    with pytest.raises(ValueError,match='report'):
        save_comparison_artifacts(tmp_path/'bundle',a,b,compare_trajectories(a,b),rtol=1e-5,atol=1e-7,check_extras=['lr'])
    assert not (tmp_path/'bundle').exists()


@pytest.mark.parametrize('missing',['lr','momentum'])
def test_missing_selected_evidence_fails_replay(tmp_path,missing):
    path,_=save(tmp_path)
    content=json.loads((path/'resumed.json').read_text())
    del content['steps'][1]['extras'][missing]
    (path/'resumed.json').write_text(json.dumps(content))
    assert not replay_comparison_artifacts(path).passed


def test_cli_replay_uses_selected_fields(tmp_path,capsys):
    from resumecheck.cli import main
    path,_=save(tmp_path,True)
    assert main(['replay',str(path)])==1
    assert 'extras.lr' in capsys.readouterr().out


@pytest.mark.parametrize('flag',[None,1,'true',False])
def test_model_scope_cannot_be_dropped(tmp_path,flag):
    path,_=save(tmp_path,state=True)
    m=json.loads((path/'manifest.json').read_text());m['comparison']['check_model_state']=flag
    (path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError): replay_comparison_artifacts(path)


def test_saved_pass_never_overrides_extra_failure(tmp_path):
    path,_=save(tmp_path,True)
    report=json.loads((path/'report.json').read_text());report['passed']=True
    (path/'report.json').write_text(json.dumps(report))
    assert not replay_comparison_artifacts(path).passed


def test_extra_scope_round_trips_exact_integer_values(tmp_path):
    a,b=pair()
    a.steps[0].extras['lr']=9007199254740993
    b.steps[0].extras['lr']=9007199254740992.0
    report=compare_trajectories(a,b,check_extras=['lr'],rtol=0,atol=0)
    path=tmp_path/'bundle'
    save_comparison_artifacts(path,a,b,report,rtol=0,atol=0,check_extras=['lr'])
    assert asdict(replay_comparison_artifacts(path))==asdict(report)


def test_actual_scheduler_failure_replays_after_restart(tmp_path):
    import torch
    from copy import deepcopy

    def setup():
        weight = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.SGD([weight], lr=.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.1)
        return weight, optimizer, scheduler

    def step(weight, optimizer, scheduler, recorder, index):
        # Zero gradient leaves the model and loss unchanged. The LR is still
        # scheduled and is the only observable that diagnoses the lost state.
        optimizer.zero_grad()
        loss = weight * 0 + 1
        loss.backward()
        optimizer.step()
        scheduler.step()
        recorder.record(index, loss=float(loss.detach()), lr=optimizer.param_groups[0]['lr'])

    weight, optimizer, scheduler = setup()
    baseline = TrajectoryRecorder('baseline')
    step(weight, optimizer, scheduler, baseline, 0)
    checkpoint = deepcopy((optimizer.state_dict(), scheduler.state_dict()))
    step(weight, optimizer, scheduler, baseline, 1)

    for restore_scheduler in [False, True]:
        weight2, optimizer2, scheduler2 = setup()
        optimizer2.load_state_dict(checkpoint[0])
        if restore_scheduler:
            scheduler2.load_state_dict(checkpoint[1])
        resumed = TrajectoryRecorder('resumed')
        resumed.trajectory.steps = deepcopy(baseline.trajectory.steps[:1])
        step(weight2, optimizer2, scheduler2, resumed, 1)
        assert compare_trajectories(baseline.trajectory, resumed.trajectory).passed
        r = compare_trajectories(baseline.trajectory, resumed.trajectory, check_extras=['lr'])
        assert r.passed is restore_scheduler
        if not restore_scheduler:
            assert (r.first_divergence.step, r.first_divergence.field) == (1, 'extras.lr')

        path = tmp_path / ('restored' if restore_scheduler else 'omitted')
        save_comparison_artifacts(path, baseline.trajectory, resumed.trajectory, r,
                                  rtol=1e-5, atol=1e-7, check_extras=['lr'])
        assert asdict(replay_comparison_artifacts(path)) == asdict(r)
