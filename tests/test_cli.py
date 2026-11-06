import json
import pytest

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


def test_cli_nested_batch_structure_diverges(tmp_path, capsys):
    paths = []
    for name, batch in [("a", [[1], 2]), ("b", [[1, 2]])]:
        rec = TrajectoryRecorder(name)
        rec.record(1, loss=1.0, batch=batch)
        path = tmp_path / f"{name}.json"
        rec.trajectory.save(path)
        paths.append(str(path))
    assert main(["compare", *paths]) == 1
    assert "field=batch" in capsys.readouterr().out


def _deeply_nested():
    import json

    # The C decoder has a different nesting limit on newer Python versions.
    # Find a bounded real decoder failure instead of assuming depth=2000.
    for depth in (1000, 2000, 4000, 8000, 16000, 32000):
        text = '{"a":' * depth + '1' + '}' * depth
        try:
            json.loads(text)
        except RecursionError:
            return text
    raise AssertionError("decoder recursion probe exceeded bounded depth")


@pytest.mark.parametrize("json_output", [False, True])
def test_deeply_nested_trajectory_json_is_a_clean_error(tmp_path, capsys, json_output):
    # Past the interpreter recursion limit the decoder raises RecursionError;
    # that must stay a clean exit-2 error, never a traceback or a verdict code.
    deep = _deeply_nested()
    a = tmp_path / "a.json"
    a.write_text('{"name": "a", "meta": ' + deep + ', "steps": []}')
    assert main(["compare", str(a), str(a)] + (["--json"] if json_output else [])) == 2
    captured = capsys.readouterr()
    assert "could not verify" in captured.err
    if json_output:
        payload = json.loads(captured.out)
        assert payload["status"] == "error" and payload["passed"] is False
        assert payload["report"] is None and payload["error"]
    else:
        assert captured.out == ""


@pytest.mark.parametrize("json_output", [False, True])
def test_deeply_nested_bundle_member_is_a_clean_error(tmp_path, capsys, json_output):
    from resumecheck.artifacts import save_comparison_artifacts
    from resumecheck.compare import compare_trajectories
    from resumecheck.recorder import TrajectoryRecorder
    left, right = TrajectoryRecorder("full"), TrajectoryRecorder("resumed")
    for step in range(1, 3):
        left.record(step, loss=1.0, batch=[step])
        right.record(step, loss=1.0, batch=[step])
    report = compare_trajectories(left.trajectory, right.trajectory)
    bundle = tmp_path / "evidence"
    save_comparison_artifacts(bundle, left.trajectory, right.trajectory, report, rtol=0, atol=0)
    member = bundle / "resumed.json"
    data = member.read_text()
    member.write_text(data.replace('"meta": {}', '"meta": ' + _deeply_nested()))
    assert main(["replay", str(bundle)] + (["--json"] if json_output else [])) == 2
    captured = capsys.readouterr()
    assert "could not verify" in captured.err
    if json_output:
        payload = json.loads(captured.out)
        assert payload["status"] == "error" and payload["passed"] is False
        assert payload["report"] is None and payload["error"]
    else:
        assert captured.out == ""
