"""Command line: compare two recorded trajectory files."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import sys

from .compare import compare_trajectories
from .recorder import Trajectory


def _json_safe(value):
    """Strict JSON has no nonfinite numeric literals. Preserve them as labels."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="resumecheck",
        description="Verify checkpoint+resume training matches uninterrupted training.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser(
        "compare", help="compare an uninterrupted trajectory against a resumed one"
    )
    c.add_argument("uninterrupted", help="trajectory JSON from the full run")
    c.add_argument("resumed", help="trajectory JSON from the checkpoint+resume run")
    c.add_argument("--rtol", type=float, default=1e-5)
    c.add_argument("--atol", type=float, default=1e-7)
    c.add_argument("--no-batch", action="store_true", help="skip batch fingerprint check")
    c.add_argument("--extra", action="append", default=None, metavar="NAME",
                   help="also compare this recorded numeric extra on every step (repeatable)")
    c.add_argument("--json", action="store_true", help="emit a versioned strict JSON report")
    replay = sub.add_parser("replay", help="recompute a completed adapter evidence bundle with saved settings")
    replay.add_argument("directory", help="local evidence directory containing the final manifest")
    replay.add_argument("--json", action="store_true", help="emit a versioned strict JSON report")
    args = p.parse_args(argv)

    try:
        if args.command == "replay":
            from .artifacts import replay_comparison_artifacts

            report = replay_comparison_artifacts(args.directory)
        else:
            a = Trajectory.load(args.uninterrupted)
            b = Trajectory.load(args.resumed)
            report = compare_trajectories(
                a, b, rtol=args.rtol, atol=args.atol, check_batch=not args.no_batch,
                check_extras=args.extra
            )
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        if args.json:
            print(json.dumps({"schema_version": 1, "status": "error", "passed": False,
                              "error": str(error), "report": None}, allow_nan=False))
        print(f"resumecheck: could not verify comparison: {error}", file=sys.stderr)
        return 2
    if args.json:
        payload = {"schema_version": 1, "status": "pass" if report.passed else "fail",
                   "passed": report.passed, "error": None, "report": asdict(report)}
        print(json.dumps(_json_safe(payload), allow_nan=False))
        return 0 if report.passed else 1
    print(report.summary)
    if report.first_divergence:
        d = report.first_divergence
        print(f"  step={d.step} field={d.field}: {d.detail}")
    for w in report.missing_fields:
        print(f"  warning: {w}")
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
