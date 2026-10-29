from resumecheck.compare import compare_trajectories
from resumecheck.recorder import StepRecord, Trajectory


def traj(name, losses, fps=None, steps=None):
    steps = steps if steps is not None else list(range(len(losses)))
    return Trajectory(
        name=name,
        steps=[
            StepRecord(
                step=s,
                loss=l,
                batch_fingerprint=(fps[i] if fps else f"fp{i:04d}"),
            )
            for i, (s, l) in enumerate(zip(steps, losses))
        ],
    )


def test_identical_passes():
    r = compare_trajectories(traj("a", [1.0, 0.9, 0.8]), traj("b", [1.0, 0.9, 0.8]))
    assert r.passed and r.steps_compared == 3
    assert r.first_divergence is None


def test_loss_divergence_reports_first_step():
    r = compare_trajectories(traj("a", [1.0, 0.9, 0.8]), traj("b", [1.0, 0.91, 0.8]))
    assert not r.passed
    assert r.first_divergence.step == 1
    assert r.first_divergence.field == "loss"


def test_tolerance_allows_tiny_drift():
    r = compare_trajectories(
        traj("a", [1.0]), traj("b", [1.0 + 1e-9]), rtol=1e-5, atol=1e-7
    )
    assert r.passed


def test_batch_mismatch_reports_dataloader_hint():
    a = traj("a", [1.0, 0.9])
    b = traj("b", [1.0, 0.9], fps=["fp0000", "fp9999"])
    r = compare_trajectories(a, b)
    assert not r.passed
    assert r.first_divergence.field == "batch"
    assert r.first_divergence.step == 1
    assert "dataloader" in r.first_divergence.detail


def test_misaligned_steps_fail():
    a = traj("a", [1.0, 0.9], steps=[0, 1])
    b = traj("b", [1.0, 0.9], steps=[0, 2])
    r = compare_trajectories(a, b)
    assert not r.passed and r.first_divergence.field == "step"


def test_unequal_lengths_warn_but_compare_prefix():
    r = compare_trajectories(traj("a", [1.0, 0.9, 0.8]), traj("b", [1.0, 0.9]))
    assert r.passed
    assert r.steps_compared == 2
    assert any("step count" in m for m in r.missing_fields)


def test_nan_equal_nan_passes():
    r = compare_trajectories(traj("a", [float("nan")]), traj("b", [float("nan")]))
    assert r.passed
