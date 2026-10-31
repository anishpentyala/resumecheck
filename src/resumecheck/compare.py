"""Compare two trajectories and locate the first divergence."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .recorder import Trajectory


@dataclass
class Divergence:
    step: int
    field: str
    value_a: Optional[float]
    value_b: Optional[float]
    detail: str


@dataclass
class CompareReport:
    """Result of comparing an uninterrupted run against a resumed run."""

    passed: bool
    steps_compared: int
    first_divergence: Optional[Divergence] = None
    missing_fields: List[str] = field(default_factory=list)
    summary: str = ""

    def __str__(self) -> str:
        return self.summary


def _floats_close(a: float, b: float, rtol: float, atol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= atol + rtol * abs(b)


def compare_trajectories(
    uninterrupted: Trajectory,
    resumed: Trajectory,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    check_loss: bool = True,
    check_batch: bool = True,
    check_grad_norm: bool = True,
    allow_partial: bool = False,
) -> CompareReport:
    """Compare step-by-step. The first field to exceed tolerance at the
    earliest step is reported as the divergence - that step is where resume
    state restoration went wrong (or where nondeterminism begins).

    Steps are matched by their ordinal index (0, 1, 2, ... of the full
    training run), so the resumed trajectory must be the concatenation of the
    pre-crash and post-resume phases in training order.

    Fails closed: a step-count mismatch, fields recorded in only one run, or
    a step with no shared enabled observable, or legacy batch fingerprints
    means the runs were NOT verified, so the report fails (the condition is listed in ``missing_fields``). Pass
    ``allow_partial=True`` to explicitly scope the check to the common
    prefix: a step-count mismatch then demotes to a warning, while having
    nothing comparable or one-sided fields still fails.
    """
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    for label, trajectory in (("uninterrupted", uninterrupted), ("resumed", resumed)):
        if any(right.step <= left.step for left, right in zip(trajectory.steps, trajectory.steps[1:])):
            detail = f"{label} step indices must be strictly increasing in recorded order"
            return CompareReport(passed=False, steps_compared=0, missing_fields=[detail], summary=f"FAIL: {detail}")
    a = uninterrupted.steps
    b = resumed.steps
    n = min(len(a), len(b))
    missing: List[str] = []
    if len(a) != len(b):
        missing.append(
            f"step count differs: {len(a)} vs {len(b)} (compared first {n})"
        )

    one_sided: set = set()
    if not any((check_loss, check_batch, check_grad_norm)):
        missing.append("no checks enabled")

    for i in range(n):
        sa, sb = a[i], b[i]
        if sa.step != sb.step:
            return CompareReport(
                passed=False,
                steps_compared=i,
                first_divergence=Divergence(
                    step=min(sa.step, sb.step),
                    field="step",
                    value_a=float(sa.step),
                    value_b=float(sb.step),
                    detail="step indices misaligned between runs",
                ),
                missing_fields=missing,
                summary=f"DIVERGED at record {i}: step indices misaligned",
            )
        enabled_fields = (
            ("loss", check_loss),
            ("batch_fingerprint", check_batch),
            ("grad_norm", check_grad_norm),
        )
        if not any(
            enabled and getattr(sa, name) is not None and getattr(sb, name) is not None
            for name, enabled in enabled_fields
        ):
            missing.append(f"no comparable enabled fields at step {sa.step}")
        if check_batch:
            if any(fp is not None and not fp.startswith("v2:") for fp in (sa.batch_fingerprint, sb.batch_fingerprint)):
                if "legacy_fingerprint" not in one_sided:
                    one_sided.add("legacy_fingerprint")
                    missing.append(
                        "unversioned batch fingerprints cannot verify structure; "
                        "record both runs with v2 fingerprints"
                    )
            if (sa.batch_fingerprint is None) != (sb.batch_fingerprint is None):
                if "batch" not in one_sided:
                    one_sided.add("batch")
                    missing.append("batch_fingerprint recorded in only one run")
            elif (
                sa.batch_fingerprint is not None
                and sa.batch_fingerprint != sb.batch_fingerprint
            ):
                return CompareReport(
                    passed=False,
                    steps_compared=i,
                    first_divergence=Divergence(
                        step=sa.step,
                        field="batch",
                        value_a=None,
                        value_b=None,
                        detail=(
                            f"different data seen at step {sa.step} "
                            f"({sa.batch_fingerprint} != {sb.batch_fingerprint}) - "
                            "dataloader position or shuffling state was not restored"
                        ),
                    ),
                    missing_fields=missing,
                    summary=f"DIVERGED at step {sa.step}: different training data consumed",
                )
        for fname, sa_v, sb_v in (
            ("loss", sa.loss if check_loss else None, sb.loss if check_loss else None),
            (
                "grad_norm",
                sa.grad_norm if check_grad_norm else None,
                sb.grad_norm if check_grad_norm else None,
            ),
        ):
            if sa_v is None or sb_v is None:
                if (sa_v is None) != (sb_v is None) and fname not in one_sided:
                    one_sided.add(fname)
                    missing.append(f"{fname} recorded in only one run")
                continue
            if not _floats_close(sa_v, sb_v, rtol, atol):
                return CompareReport(
                    passed=False,
                    steps_compared=i,
                    first_divergence=Divergence(
                        step=sa.step,
                        field=fname,
                        value_a=sa_v,
                        value_b=sb_v,
                        detail=(
                            f"{fname} differs at step {sa.step}: {sa_v} vs {sb_v} "
                            f"(rtol={rtol}, atol={atol})"
                        ),
                    ),
                    missing_fields=missing,
                    summary=f"DIVERGED at step {sa.step}: {fname} {sa_v} vs {sb_v}",
                )

    if n == 0:
        missing.append("no comparable fields recorded")
    if allow_partial:
        hard_fail = [m for m in missing if not m.startswith("step count differs")]
    else:
        hard_fail = missing
    passed = not hard_fail
    if passed:
        summary = (
            f"PASS: {n} steps identical within rtol={rtol}, atol={atol}"
            + (f" (warnings: {'; '.join(missing)})" if missing else "")
        )
    else:
        summary = (
            f"FAIL: {'; '.join(hard_fail)}"
            + (
                f" (compared {n} steps, no divergence within compared prefix)"
                if n
                else ""
            )
        )
    return CompareReport(
        passed=passed,
        steps_compared=n,
        missing_fields=missing,
        summary=summary,
    )
