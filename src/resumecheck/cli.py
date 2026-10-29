"""Command line: compare two recorded trajectory files."""
from __future__ import annotations

import argparse
import sys

from .compare import compare_trajectories
from .recorder import Trajectory


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
    args = p.parse_args(argv)

    a = Trajectory.load(args.uninterrupted)
    b = Trajectory.load(args.resumed)
    report = compare_trajectories(
        a, b, rtol=args.rtol, atol=args.atol, check_batch=not args.no_batch
    )
    print(report.summary)
    if report.first_divergence:
        d = report.first_divergence
        print(f"  step={d.step} field={d.field}: {d.detail}")
    for w in report.missing_fields:
        print(f"  warning: {w}")
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
