"""A new test module cannot silently fall outside every dependency-isolated leg."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('ci_assignment', ROOT / 'scripts/check_ci_test_coverage.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def fixture(tmp_path, workflow, files):
    (tmp_path / '.github/workflows').mkdir(parents=True)
    (tmp_path / '.github/workflows/ci.yml').write_text(workflow)
    (tmp_path / 'tests').mkdir()
    for file in files:
        (tmp_path / 'tests' / file).write_text('')
    return tmp_path


def test_actual_repository_has_no_unassigned_test_files():
    assert CHECK.unassigned_tests(ROOT) == []


def test_missing_module_is_reported_even_if_named_in_comment(tmp_path):
    root = fixture(tmp_path, '# tests/test_orphan.py\n - run: python -m pytest tests/test_a.py -q\n',
                   ['test_a.py','test_orphan.py'])
    assert CHECK.unassigned_tests(root) == ['tests/test_orphan.py']


def test_nodes_and_shared_files_count_as_assignment_not_test_execution(tmp_path):
    root = fixture(tmp_path, '- run: python -m pytest tests/test_a.py::test_one -k "subset" -q\n', ['test_a.py'])
    assert CHECK.unassigned_tests(root) == []


@pytest.mark.parametrize('workflow', ['# no pytest', '- run: python -m pytest tests/missing.py -q'])
def test_unusable_workflow_target_fails_cleanly(tmp_path, workflow):
    root = fixture(tmp_path, workflow, ['test_a.py'])
    with pytest.raises(ValueError): CHECK.unassigned_tests(root)
