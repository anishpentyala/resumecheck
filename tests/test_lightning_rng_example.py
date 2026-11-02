"""Execute the CPU-only caller-owned RNG example, including its negative control."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning.pytorch")
from resumecheck.artifacts import replay_comparison_artifacts


@pytest.fixture
def example():
    path = (
        Path(__file__).resolve().parents[1] / "examples" / "lightning_dropout_resume.py"
    )
    spec = importlib.util.spec_from_file_location(
        "resumecheck_lightning_rng_example", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("accumulation", [1, 2, 3, 4])
@pytest.mark.parametrize("restore", [False, True])
def test_actual_dropout_rng_control_and_replay(
    example, tmp_path, accumulation, restore
):
    path = tmp_path / "evidence"
    args = ["--accumulation", str(accumulation), "--artifacts-dir", str(path)]
    if not restore:
        args += ["--without-rng-restore"]
    assert example.main(args) == (0 if restore else 1)
    report = replay_comparison_artifacts(path)
    assert report.passed is restore
    assert asdict(report) == json.loads((path / "report.json").read_text())
    if not restore:
        assert report.first_divergence.field == "model_state"
        assert report.first_divergence.step == 3


@pytest.mark.parametrize("value", ["0", "-1", "5"])
def test_invalid_accumulation_precedes_training(example, monkeypatch, value):
    def forbidden(*args, **kwargs):
        pytest.fail("training must not start")

    monkeypatch.setattr(example, "check_lightning", forbidden)
    with pytest.raises(SystemExit) as error:
        example.main(["--accumulation", value])
    assert error.value.code == 2
