"""An application-owned manual checkpoint with independently omitted state."""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import pytest

pytest.importorskip("torch")
from resumecheck.artifacts import replay_comparison_artifacts


@pytest.fixture
def example():
    path = Path(__file__).resolve().parents[1] / "examples" / "manual_checkpoint.py"
    spec = importlib.util.spec_from_file_location("manual_checkpoint_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", [42, 123])
@pytest.mark.parametrize("omit", [None, "model", "optimizer", "scheduler", "rng", "sampler"])
def test_checkpoint_state_and_independent_omission_controls(example, tmp_path, seed, omit):
    path = tmp_path / "evidence"
    args = ["--seed", str(seed), "--artifacts-dir", str(path)]
    if omit:
        args += ["--omit", omit]
    assert example.main(args) == (1 if omit else 0)
    report = replay_comparison_artifacts(path)
    assert asdict(report) == json.loads((path / "report.json").read_text())
    assert report.steps_compared == (6 if omit is None else 3 if omit == "scheduler" else 2)
    assert report.passed == (omit is None)
    full = json.loads((path / "uninterrupted.json").read_text())["steps"]
    resumed = json.loads((path / "resumed.json").read_text())["steps"]
    assert len(full) == len(resumed) == 6
    assert [step["step"] for step in resumed] == list(range(1, 7))
    assert all("model_fingerprint" in step for step in resumed)
    if omit:
        assert report.first_divergence.field == ("batch" if omit == "sampler" else "model_state")
        assert report.first_divergence.step == (4 if omit == "scheduler" else 3)
    else:
        assert full == resumed


@pytest.mark.parametrize("kind", ["existing", "missing-parent"])
def test_invalid_destination_precedes_training(example, tmp_path, monkeypatch, kind):
    path = tmp_path / "evidence"
    if kind == "existing":
        path.mkdir()
    else:
        path = path / "child"
    def forbidden(*args, **kwargs):
        pytest.fail("must not begin training")
    monkeypatch.setattr(example, "make_training", forbidden)
    with pytest.raises((FileExistsError, FileNotFoundError)):
        example.check_manual(artifacts_dir=path)


def test_bad_omission_rejected_before_training(example, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("must not begin training")
    monkeypatch.setattr(example, "make_training", forbidden)
    with pytest.raises(ValueError, match="unsupported"):
        example.check_manual(omit="typo")
    with pytest.raises(SystemExit) as error:
        example.main(["--omit", "typo"])
    assert error.value.code == 2
