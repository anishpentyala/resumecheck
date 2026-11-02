"""Executable offline example: actual attention/dropout, no model downloads."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('transformers')


@pytest.fixture
def example(monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    path = Path(__file__).resolve().parents[1] / 'examples' / 'accumulated_gpt2.py'
    spec = importlib.util.spec_from_file_location('resumecheck_offline_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('accumulation', [2, 3])
def test_offline_gpt2_accumulated_resume(example, accumulation):
    assert example.main(['--accumulation', str(accumulation)]) == 0


@pytest.mark.parametrize('accumulation', ['0', '-1'])
def test_example_rejects_invalid_accumulation(example, accumulation):
    with pytest.raises(SystemExit) as error:
        example.main(['--accumulation', accumulation])
    assert error.value.code == 2


def test_example_returns_failure_when_parity_is_not_verified(example, monkeypatch):
    from resumecheck.compare import CompareReport
    monkeypatch.setattr(example, 'check_hf', lambda *a, **k: CompareReport(False, 0, summary='FAIL: deliberate control'))
    assert example.main([]) == 1


def test_offline_example_writes_replayable_artifacts(example, tmp_path):
    import json
    from dataclasses import asdict
    from resumecheck.compare import compare_trajectories
    from resumecheck.recorder import Trajectory
    directory = tmp_path / 'gpt2-evidence'
    assert example.main(['--accumulation', '3', '--artifacts-dir', str(directory)]) == 0
    manifest = json.loads((directory / 'manifest.json').read_text())
    report = compare_trajectories(Trajectory.load(directory / 'uninterrupted.json'),
                                  Trajectory.load(directory / 'resumed.json'), **manifest['comparison'])
    assert report.passed
    assert asdict(report) == json.loads((directory / 'report.json').read_text())
