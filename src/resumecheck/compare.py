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
    return abs(a - b) <= atol + rtol * abs(b)


def compare_trajectories(
    uninterrupted: Trajectory,
    resumed: Trajectory,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    check_loss: bool = True,
    check_batch: bool = True,
    check_grad_norm: bool = True,
) -> CompareReport:
    """Compare step-by-step. The first field to exceed tolerance at the
    earliest step is reported as the divergence - that step is where resume
    state restoration went wrong (or where nondeterminism begins).

    Steps are matched by their ordinal index (0, 1, 2, ... of the full
    training run), so the resumed trajectory must be the concatenation of the
    pre-crash and post-resume phases in training order.
    """
    a = sorted(uninterrupted.steps, key=lambda s: s.step)
    b = sorted(resumed.steps, key=lambda s: s.step)
    n = min(len(a), len(b))
    missing: List[str] = []
    if len(a) != len(b):
        missing.append(
            f"step count differs: {len(a)} vs {len(b)} (compared first {n})"
        )

    checked = {"loss": check_loss, "batch": check_batch, "grad": check_grad_norm}
    if not any(
        getattr(s, f) is not None
        for s in (a[:n] or [])
        for f in ("loss", "batch_fingerprint", "grad_norm")
    ):
        missing.append("no comparable fields recorded")

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
        if check_batch and sa.batch_fingerprint and sb.batch_fingerprint:
            if sa.batch_fingerprint != sb.batch_fingerprint:
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
                if (sa_v is None) != (sb_v is None) and fname not in missing:
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

    return CompareReport(
        passed=not missing or True,
        steps_compared=n,
        missing_fields=missing,
        summary=(
            f"PASS: {n} steps identical within rtol={rtol}, atol={atol}"
            + (f" (warnings: {'; '.join(missing)})" if missing else "")
        ),
  )
