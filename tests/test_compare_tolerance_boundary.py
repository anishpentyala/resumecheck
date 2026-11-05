"""Finite arithmetic rounding cannot establish a false tolerance PASS."""
import json
import math
import random
from fractions import Fraction

import pytest

from resumecheck.compare import _floats_close, compare_trajectories
from resumecheck.recorder import StepRecord, Trajectory


CASES = [
    (1e-323, 5e-324, .75, 0),
    (-1e-323, -5e-324, .75, 0),
    (-1e-300, 1e300, 1., 0),
    (1e-300, -1e300, 1., 0),
    (7.651281824299423e34, 2.6320502816557916e50, .9999999999999997, 0),
    (-3.9204861326296634e56, 1.3754584565330638e-163, 2.850312282430917e219, 0),
]


@pytest.mark.parametrize('field',['loss','grad_norm'])
@pytest.mark.parametrize('a,b,rtol,atol',CASES)
def test_finite_rounded_boundaries_do_not_verify(field,a,b,rtol,atol):
    assert abs(Fraction(a)-Fraction(b)) > Fraction(atol)+Fraction(rtol)*abs(Fraction(b))
    left=Trajectory('a',[StepRecord(0,**{field:a})])
    right=Trajectory('b',[StepRecord(0,**{field:b})])
    report=compare_trajectories(left,right,rtol=rtol,atol=atol)
    assert not report.passed
    assert report.first_divergence.field==field
    assert report.first_divergence.step==0


@pytest.mark.parametrize('a,b,rtol,atol',[
    (1e-323,5e-324,1.,0), (1e-323,5e-324,0,5e-324),
    (1.,2.,.5,0), (1.,2.,0,1.), (0.,-0.,0,0),
    (1e308,-1e308,2.,0), (1e-300,-1e300,math.nextafter(1.,math.inf),0),
])
def test_exact_boundary_and_above_still_pass(a,b,rtol,atol):
    assert _floats_close(a,b,rtol,atol)


@pytest.mark.parametrize('field',['loss','grad_norm'])
def test_earliest_boundary_divergence_and_disabled_scope(field):
    a=Trajectory('a',[StepRecord(0,loss=1.),StepRecord(1,loss=1.,**({} if field=='loss' else {field:1e-323}))])
    b=Trajectory('b',[StepRecord(0,loss=1.),StepRecord(1,loss=1.,**({} if field=='loss' else {field:5e-324}))])
    if field=='loss':
        a.steps[1].loss=1e-323;b.steps[1].loss=5e-324
        a.steps[0].grad_norm=b.steps[0].grad_norm=1.
        a.steps[1].grad_norm=b.steps[1].grad_norm=1.
    result=compare_trajectories(a,b,rtol=.75,atol=0)
    assert not result.passed and result.first_divergence.step==1
    assert compare_trajectories(a,b,rtol=.75,atol=0,
                                **{'check_'+field:False}).passed


@pytest.mark.parametrize('mode',['text','json'])
def test_cli_subnormal_false_pass_is_divergence(tmp_path,capsys,mode):
    from resumecheck.cli import main
    paths=[]
    for label,value in [('a',1e-323),('b',5e-324)]:
        p=tmp_path/(label+'.json')
        Trajectory(label,[StepRecord(0,loss=value)]).save(p)
        paths.append(str(p))
    args=['compare',*paths,'--rtol','.75','--atol','0']
    if mode=='json':args.append('--json')
    assert main(args)==1
    out=capsys.readouterr()
    assert not out.err
    if mode=='json':
        data=json.loads(out.out)
        assert data['status']=='fail' and not data['passed']
    else:
        assert 'DIVERGED' in out.out


def test_deterministic_exact_oracle_across_exponent_scales():
    rng=random.Random(11050937)
    for _ in range(1000):
        a=math.ldexp(rng.uniform(-1,1),rng.randrange(-1070,1000))
        b=math.ldexp(rng.uniform(-1,1),rng.randrange(-1070,1000))
        if b==0:continue
        ratio=abs(Fraction(a)-Fraction(b))/abs(Fraction(b))
        try:boundary=float(ratio)
        except OverflowError:continue
        if not math.isfinite(boundary):continue
        for rtol in [boundary,math.nextafter(boundary,0),math.nextafter(boundary,math.inf)]:
            if not math.isfinite(rtol):continue
            expected=abs(Fraction(a)-Fraction(b))<=Fraction(rtol)*abs(Fraction(b))
            assert _floats_close(a,b,rtol,0) is expected
