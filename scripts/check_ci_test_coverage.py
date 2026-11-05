"""Check explicit pytest file assignment, not actual collection or test results.

The workflow deliberately uses one-line explicit test paths for dependency
isolation. A new file must be assigned to a leg. This check cannot prove the
file's tests are selected by -k, run successfully or have their extras installed.
"""
import re
import shlex
from pathlib import Path


RUN = re.compile(r'^\s*(?:-\s*)?run:\s*python -m pytest\s+(.+)$', re.MULTILINE)


def unassigned_tests(root):
    root = Path(root)
    workflow = (root / '.github/workflows/ci.yml').read_text()
    commands = RUN.findall(workflow)
    if not commands:
        raise ValueError('no explicit one-line pytest commands found in CI workflow')
    assigned = set()
    for command in commands:
        for argument in shlex.split(command, comments=True):
            if argument.startswith('tests/'):
                file = argument.split('::', 1)[0]
                if not file.endswith('.py') or not (root / file).is_file():
                    raise ValueError('CI pytest target is not an existing test file: ' + file)
                assigned.add(file)
    return sorted(str(path.relative_to(root)) for path in (root / 'tests').glob('test_*.py')
                  if str(path.relative_to(root)) not in assigned)


def main():
    root = Path(__file__).resolve().parents[1]
    try:
        missing = unassigned_tests(root)
    except (ValueError, OSError) as error:
        print('CI test assignment error: ' + str(error))
        return 1
    if missing:
        print('Test modules with no explicit pytest CI assignment:')
        print('\n'.join(missing))
        return 1
    print('Every test module has an explicit pytest CI assignment; selection and results require CI logs.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
