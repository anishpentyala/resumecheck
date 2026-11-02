"""Execute the offline example and replay every positive and negative control."""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import pytest

pytest.importorskip('torch')
pytest.importorskip('accelerate')
pytest.importorskip('torchdata')
from resumecheck.artifacts import replay_comparison_artifacts

@pytest.fixture
def example():
    path=Path(__file__).resolve().parents[1]/'examples'/'accelerate_dropout_resume.py'
    spec=importlib.util.spec_from_file_location('accelerate_dropout_example',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('seed',[42,123])
@pytest.mark.parametrize('path',['stateful','boundary','skip'])
@pytest.mark.parametrize('isolate',[False,True])
def test_example_exact_state_parity_and_negative_control(example,tmp_path,seed,path,isolate):
    dest=tmp_path/'evidence'
    args=['--path',path,'--seed',str(seed),'--artifacts-dir',str(dest)]
    if not isolate:args+=['--without-loader-isolation']
    assert example.main(args)==(0 if isolate else 1)
    report=replay_comparison_artifacts(dest)
    assert report.passed is isolate
    assert asdict(report)==json.loads((dest/'report.json').read_text())
    if isolate:assert report.steps_compared==19
    else:assert report.first_divergence.field=='model_state'

@pytest.mark.parametrize('path',['boundary','skip'])
def test_example_sampler_negative_control(example,tmp_path,path):
    dest=tmp_path/'evidence'
    assert example.main(['--path',path,'--without-sampler-workaround','--artifacts-dir',str(dest)])==1
    report=replay_comparison_artifacts(dest)
    assert report.first_divergence.field=='batch'
    assert report.first_divergence.step==9

@pytest.mark.parametrize('args',[['--path','invalid'],['--path','stateful','--without-sampler-workaround']])
def test_invalid_scope_does_not_train(example,monkeypatch,args):
    def forbidden(*a,**kw):pytest.fail('must not train')
    monkeypatch.setattr(example,'check_accelerate',forbidden)
    monkeypatch.setattr(example,'check_accelerate_skip_path',forbidden)
    with pytest.raises(SystemExit) as error:example.main(args)
    assert error.value.code==2
