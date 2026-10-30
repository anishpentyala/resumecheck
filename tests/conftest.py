"""Shared pytest configuration.

The adapter test modules guard their imports with ``pytest.importorskip`` so
local contributors can run the suite with any subset of extras installed.
That is the wrong default for CI: a leg that lost an extra would skip the
whole adapter module and report green. CI adapter legs therefore set
``RESUMECHECK_REQUIRE_NO_SKIPS=1``, which converts any skip in the run into
a session failure. The guard is opt-in; local runs are unaffected.
"""

import os

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    if not os.environ.get("RESUMECHECK_REQUIRE_NO_SKIPS"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if skipped:
        reporter.write_line(
            "RESUMECHECK_REQUIRE_NO_SKIPS: {} skip(s) occurred - a required "
            "extra is missing from this environment".format(len(skipped))
        )
        session.exitstatus = 1
