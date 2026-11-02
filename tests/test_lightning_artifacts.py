"""Actual Lightning evidence replays parity and plain-loader divergence."""
import json
from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('lightning.pytorch')

from resumecheck.lightning import check_lightning
from resumecheck.compare import compare_trajectories
from resumecheck.recorder import Trajectory


def lightning_fixtures():
    pytest.importorskip('lightning.pytorch')
    spec = importlib.util.spec_from_file_location('adapter_evidence_fixtures', Path(__file__).with_name('test_lightning_parity.py'))
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    return fixtures


def replay(path, report):
    manifest = json.loads((path / 'manifest.json').read_text())
    full = Trajectory.load(path / 'uninterrupted.json')
    resumed = Trajectory.load(path / 'resumed.json')
    assert len(full.steps) == len(resumed.steps) == 6
    assert asdict(compare_trajectories(full, resumed, **manifest['comparison'])) == asdict(report)
    assert json.loads((path / 'report.json').read_text()) == asdict(report)


@pytest.mark.parametrize('stateful', [True, False])
def test_lightning_bundle_replays_pass_and_known_plain_loader_divergence(tmp_path, stateful):
    fixtures = lightning_fixtures()
    path = tmp_path / 'evidence'
    report = check_lightning(fixtures.TinyModel, fixtures.StatefulDM if stateful else fixtures.TinyDM,
                             max_steps=6, resume_at=3, seed=42, rtol=.0003, atol=.0002,
                             artifacts_dir=path)
    replay(path, report)
    assert report.passed is stateful
    if not stateful:
        assert report.first_divergence.field == 'batch'
        assert report.first_divergence.step == 4


@pytest.mark.parametrize('kind', ['directory', 'file', 'missing_parent'])
def test_invalid_destination_precedes_model_factory(tmp_path, kind):
    path = tmp_path / 'evidence'
    if kind == 'directory':
        path.mkdir()
    elif kind == 'file':
        path.write_text('keep')
    else:
        path = path / 'child'
    def forbidden():
        pytest.fail('training must not start')
    fixtures = lightning_fixtures()
    kwargs = {'max_steps': 6, 'resume_at': 3, 'artifacts_dir': path}
    with pytest.raises((FileExistsError, FileNotFoundError)):
        check_lightning(forbidden, fixtures.StatefulDM, **kwargs)


def test_model_error_leaves_no_bundle(tmp_path):
    path = tmp_path / 'evidence'
    def broken():
        raise RuntimeError('intentional factory error')
    fixtures = lightning_fixtures()
    kwargs = {'max_steps': 6, 'resume_at': 3, 'artifacts_dir': path}
    with pytest.raises(RuntimeError, match='factory error'):
        check_lightning(broken, fixtures.StatefulDM, **kwargs)
    assert not path.exists()


def test_default_does_not_export_evidence(monkeypatch):
    import resumecheck.artifacts
    def forbidden(*args, **kwargs):
        pytest.fail('default must not export')
    monkeypatch.setattr(resumecheck.artifacts, 'save_comparison_artifacts', forbidden)
    fixtures = lightning_fixtures()
    assert check_lightning(fixtures.TinyModel, fixtures.StatefulDM, max_steps=6, resume_at=3, seed=42).passed


def test_destination_created_during_fit_is_preserved(tmp_path):
    path = tmp_path / 'evidence'
    fixtures = lightning_fixtures()
    def factory():
        if not path.exists():
            path.mkdir()
            (path / 'keep').write_text('other owner')
        return fixtures.TinyModel()
    with pytest.raises(FileExistsError):
        check_lightning(factory, fixtures.StatefulDM, max_steps=6, resume_at=3, seed=42, artifacts_dir=path)
    assert [p.name for p in path.iterdir()] == ['keep']
