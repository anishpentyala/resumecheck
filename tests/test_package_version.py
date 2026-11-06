"""The importable package version must match the released package metadata.

A stale ``resumecheck.__version__`` makes user bug reports and support
diagnostics cite the wrong release. Parse pyproject.toml as text so the
check also runs on the 3.9 stdlib-only floor (no tomllib before 3.11).
"""
import re
from pathlib import Path

import resumecheck


def test_init_version_matches_pyproject():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert match, "pyproject.toml must declare a project version"
    assert resumecheck.__version__ == match.group(1)
