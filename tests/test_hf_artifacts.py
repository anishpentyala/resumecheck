"""Actual Trainer output can be retained and replayed without training again."""
import json
from dataclasses import asdict

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from resumecheck.compare import compare_trajectories
from resumecheck.hf import check_hf
from resumecheck.recorder import Trajectory


class Dataset(torch.utils.data.Dataset):
    calls = 0
    changing = False

    def __len__(self):
        return 16

    def __getitem__(self, index):
        type(self).calls += 1
        value = type(self).calls if self.changing else index
        return {'x': torch.tensor([float(value)])}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return {'loss': self.weight.square()}


@pytest.mark.parametrize('changing', [False, True])
@pytest.mark.parametrize('accumulation', [1, 3])
def test_actual_hf_bundle_replays_pass_and_divergence(tmp_path, monkeypatch, changing, accumulation):
    monkeypatch.setattr(Dataset, 'changing', changing)
    Dataset.calls = 0
    directory = tmp_path / 'evidence'
    report = check_hf(Model, Dataset(), max_steps=4, resume_at=2,
                      artifacts_dir=directory, rtol=.0003, atol=.0002,
                      training_args_kwargs={'gradient_accumulation_steps': accumulation})
    manifest = json.loads((directory / 'manifest.json').read_text())
    full = Trajectory.load(directory / 'uninterrupted.json')
    resumed = Trajectory.load(directory / 'resumed.json')
    assert [record.step for record in full.steps] == [1, 2, 3, 4]
    assert [record.step for record in resumed.steps] == [1, 2, 3, 4]
    assert asdict(compare_trajectories(full, resumed, **manifest['comparison'])) == asdict(report)
    assert report.passed is not changing
    if changing:
        assert report.first_divergence.field == 'batch'


@pytest.mark.parametrize('kind', ['directory', 'file', 'missing_parent'])
def test_invalid_path_rejected_before_factory(tmp_path, kind):
    path = tmp_path / 'evidence'
    if kind == 'directory':
        path.mkdir()
    elif kind == 'file':
        path.write_text('retain')
    else:
        path = path / 'child'
    def forbidden():
        pytest.fail('training must not start')
    with pytest.raises((FileExistsError, FileNotFoundError)):
        check_hf(forbidden, Dataset(), max_steps=4, resume_at=2, artifacts_dir=path)


def test_training_error_propagates_without_bundle(tmp_path):
    def broken():
        raise RuntimeError('training factory failed')
    path = tmp_path / 'evidence'
    with pytest.raises(RuntimeError, match='factory failed'):
        check_hf(broken, Dataset(), max_steps=4, resume_at=2, artifacts_dir=path)
    assert not path.exists()


def test_default_does_not_write_bundle(tmp_path, monkeypatch):
    import resumecheck.artifacts
    def forbidden(*args, **kwargs):
        pytest.fail('default must not export artifacts')
    monkeypatch.setattr(resumecheck.artifacts, 'save_comparison_artifacts', forbidden)
    monkeypatch.chdir(tmp_path)
    report = check_hf(Model, Dataset(), max_steps=4, resume_at=2)
    assert report.passed
    assert not (tmp_path / 'manifest.json').exists()


def test_path_created_during_training_is_not_overwritten(tmp_path):
    path = tmp_path / 'evidence'
    calls = []
    def factory():
        calls.append(True)
        if not path.exists():
            path.mkdir()
            (path / 'keep').write_text('created while training ran')
        return Model()
    with pytest.raises(FileExistsError):
        check_hf(factory, Dataset(), max_steps=4, resume_at=2, artifacts_dir=path)
    assert len(calls) == 3  # full, pre-crash, resumed all finished
    assert [entry.name for entry in path.iterdir()] == ['keep']
    assert (path / 'keep').read_text() == 'created while training ran'
