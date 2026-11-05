"""Recorded loss/gradient evidence must retain JSON numeric identity."""
import json

import pytest

from resumecheck.compare import compare_trajectories
from resumecheck.recorder import StepRecord, Trajectory


def runs(field, left, right):
    return (Trajectory('a', [StepRecord(0, **{field:left})]),
            Trajectory('b', [StepRecord(0, **{field:right})]))


@pytest.mark.parametrize('field', ['loss','grad_norm'])
@pytest.mark.parametrize('left,right', [(True,1), (False,0), (True,True), ('1','1'), ([],[]), ({},{})])
def test_malformed_numeric_evidence_cannot_verify(field,left,right):
    report = compare_trajectories(*runs(field,left,right))
    assert not report.passed
    assert any(field in entry and 'number' in entry for entry in report.missing_fields)


@pytest.mark.parametrize('field', ['loss','grad_norm'])
@pytest.mark.parametrize('left,right', [
    (2**53+1,float(2**53)), (2**53+1,2**53),
    (10**400,10**400+1), (10**400,1.),
])
def test_large_integer_evidence_does_not_round_or_overflow(field,left,right):
    report = compare_trajectories(*runs(field,left,right),rtol=0,atol=0)
    assert not report.passed and report.first_divergence.field == field


@pytest.mark.parametrize('field', ['loss','grad_norm'])
@pytest.mark.parametrize('value', [0,1,2**53+1,10**400,1.,float('nan'),float('inf'),-float('inf')])
def test_matching_numeric_values_remain_supported(field,value):
    assert compare_trajectories(*runs(field,value,value),rtol=0,atol=0).passed


@pytest.mark.parametrize('field', ['loss','grad_norm'])
def test_exact_integer_difference_uses_requested_tolerance(field):
    a,b = runs(field,2**53+1,float(2**53))
    assert compare_trajectories(a,b,rtol=0,atol=1).passed
    assert not compare_trajectories(a,b,rtol=0,atol=.5).passed


def test_cli_boolean_loss_is_failed_evidence_not_pass(tmp_path,capsys):
    from resumecheck.cli import main
    a,b = runs('loss',True,1)
    left,right=tmp_path/'a.json',tmp_path/'b.json'
    a.save(left);b.save(right)
    assert main(['compare',str(left),str(right),'--no-batch','--json']) == 1
    data=json.loads(capsys.readouterr().out)
    assert not data['passed'] and data['status']=='fail'


def test_cli_large_integer_loss_retains_exact_json_value(tmp_path,capsys):
    from resumecheck.cli import main
    a,b = runs('loss',2**53+1,float(2**53))
    left,right=tmp_path/'a.json',tmp_path/'b.json'
    a.save(left);b.save(right)
    assert main(['compare',str(left),str(right),'--no-batch','--rtol','0','--atol','0','--json']) == 1
    data=json.loads(capsys.readouterr().out)
    assert data['report']['first_divergence']['value_a'] == 2**53+1
    assert data['report']['first_divergence']['value_b'] == float(2**53)


def test_disabled_numeric_field_stays_outside_explicit_scope():
    a,b = runs('loss','unobserved',{})
    a.steps[0].grad_norm=b.steps[0].grad_norm=1.
    assert compare_trajectories(a,b,check_loss=False).passed
