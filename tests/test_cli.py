from resumecheck.cli import main
from resumecheck.recorder import TrajectoryRecorder


def _write(tmp_path, name, order):
    rec = TrajectoryRecorder(name)
    for i, b in enumerate(order):
        rec.record(i, loss=1.0 - 0.1 * i, batch=[b, b + 1])
    p = tmp_path / f"{name}.json"
    rec.trajectory.save(p)
    return str(p)


def test_cli_compare_pass(tmp_path, capsys):
    a = _write(tmp_path, "a", [0, 1, 2])
    b = _write(tmp_path, "b", [0, 1, 2])
    assert main(["compare", a, b]) == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_compare_divergence(tmp_path, capsys):
    a = _write(tmp_path, "a", [0, 1, 2])
    b = _write(tmp_path, "b", [0, 1, 0])
    assert main(["compare", a, b]) == 1
    out = capsys.readouterr().out
    assert "DIVERGED at step 2" in out
    assert "field=batch" in out


def test_cli_requires_subcommand():
    import pytest

    with pytest.raises(SystemExit):
        main([])
