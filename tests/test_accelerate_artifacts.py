"""Both Accelerate resume paths retain replayable local evidence."""
import json
from dataclasses import asdict

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('accelerate')
pytest.importorskip('torchdata')

from resumecheck.accelerate import check_accelerate, check_accelerate_skip_path
from resumecheck.compare import compare_trajectories
from resumecheck.recorder import Trajectory




class Tokens(torch.utils.data.Dataset):
    def __len__(self):
        return 16
    def __getitem__(self, index):
        return {'x': torch.tensor([float(index) / 16]), 'y': torch.tensor([1.])}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
    def forward(self, x, y):
        return {'loss': (self.layer(x) - y).square().mean()}


def replay(path, report):
    manifest = json.loads((path / 'manifest.json').read_text())
    full = Trajectory.load(path / 'uninterrupted.json')
    resumed = Trajectory.load(path / 'resumed.json')
    assert len(full.steps) == len(resumed.steps) == 6
    assert asdict(compare_trajectories(full, resumed, **manifest['comparison'])) == asdict(report)
    assert json.loads((path / 'report.json').read_text()) == asdict(report)


@pytest.mark.parametrize('skip', [False, True])
def test_both_accelerate_paths_export_completed_evidence(tmp_path, skip):
    path = tmp_path / 'evidence'
    check = check_accelerate_skip_path if skip else check_accelerate
    report = check(Model, Tokens(), max_steps=6, resume_at=3, seed=42,
                   artifacts_dir=path, rtol=.0003, atol=.0002)
    replay(path, report)
    assert report.passed, report.summary


@pytest.mark.parametrize('adapter', ['accelerate', 'skip'])
@pytest.mark.parametrize('kind', ['directory', 'file', 'missing_parent'])
def test_invalid_destination_precedes_model_factory(tmp_path, adapter, kind):
    path = tmp_path / 'evidence'
    if kind == 'directory':
        path.mkdir()
    elif kind == 'file':
        path.write_text('keep')
    else:
        path = path / 'child'
    def forbidden():
        pytest.fail('training must not start')
    kwargs = {'max_steps': 6, 'resume_at': 3, 'artifacts_dir': path}
    with pytest.raises((FileExistsError, FileNotFoundError)):
        (check_accelerate_skip_path if adapter == 'skip' else check_accelerate)(forbidden, Tokens(), **kwargs)


@pytest.mark.parametrize('adapter', ['accelerate', 'skip'])
def test_model_error_leaves_no_bundle(tmp_path, adapter):
    path = tmp_path / 'evidence'
    def broken():
        raise RuntimeError('intentional factory error')
    kwargs = {'max_steps': 6, 'resume_at': 3, 'artifacts_dir': path}
    with pytest.raises(RuntimeError, match='factory error'):
        (check_accelerate_skip_path if adapter == 'skip' else check_accelerate)(broken, Tokens(), **kwargs)
    assert not path.exists()


@pytest.mark.parametrize('skip', [False, True])
def test_default_does_not_export_evidence(monkeypatch, skip):
    import resumecheck.artifacts
    def forbidden(*args, **kwargs):
        pytest.fail('default must not export')
    monkeypatch.setattr(resumecheck.artifacts, 'save_comparison_artifacts', forbidden)
    check = check_accelerate_skip_path if skip else check_accelerate
    assert check(Model, Tokens(), max_steps=6, resume_at=3, seed=42).passed


@pytest.mark.parametrize('skip', [False, True])
def test_destination_created_during_training_is_preserved(tmp_path, skip):
    path = tmp_path / 'evidence'
    def factory():
        if not path.exists():
            path.mkdir()
            (path / 'keep').write_text('other owner')
        return Model()
    check = check_accelerate_skip_path if skip else check_accelerate
    with pytest.raises(FileExistsError):
        check(factory, Tokens(), max_steps=6, resume_at=3, seed=42, artifacts_dir=path)
    assert [p.name for p in path.iterdir()] == ['keep']
