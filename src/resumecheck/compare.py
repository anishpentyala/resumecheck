"""Compare two trajectories and locate the first divergence."""
from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

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
    # JSON integers may exceed float precision or range. Keep their exact
    # value, including when only one side is an integer. Nonfinite float
    # behavior below remains the documented matching-NaN/infinity policy.
    if isinstance(a, int) or isinstance(b, int):
        from fractions import Fraction
        if any(isinstance(v, float) and not math.isfinite(v) for v in (a, b)):
            return False
        return abs(Fraction(a) - Fraction(b)) <= Fraction(atol) + Fraction(rtol) * abs(Fraction(b))
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    # Rounded finite subtraction/product/addition can cross the tolerance
    # boundary too (not only overflow). Compare represented values exactly,
    # including subnormal differences and underflowed tolerance products.
    from fractions import Fraction

    return abs(Fraction(a) - Fraction(b)) <= Fraction(atol) + Fraction(rtol) * abs(Fraction(b))


def compare_trajectories(
    uninterrupted: Trajectory,
    resumed: Trajectory,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    check_loss: bool = True,
    check_batch: bool = True,
    check_grad_norm: bool = True,
    allow_partial: bool = False,
    check_model_state: bool = True,
    check_extras: Optional[Sequence[str]] = None,
    check_optimizer_state: bool = True,
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
    ``check_extras`` optionally names recorded numeric extras such as ``lr``.
    Every selected field must be present and finite on both sides at each
    compared step. They use the same rtol/atol; unselected extras remain
    outside the comparison. This does not verify full scheduler/optimizer state.
    Recorded optimizer fingerprints are exact-bit observations when enabled.
    Missing or malformed fingerprints fail closed; absent evidence in both
    runs preserves the existing comparison scope. This is not a checkpoint
    loader or a guarantee about unrecorded optimizer/application state.
    The boolean scope flags require actual booleans;
    configuration strings, numbers and containers raise TypeError.
    """
    # Scope is an explicit policy, not Python truthiness. In particular a
    # configuration string "false" must not opt into partial verification.
    for name, value in (
        ("check_loss", check_loss),
        ("check_batch", check_batch),
        ("check_grad_norm", check_grad_norm),
        ("allow_partial", allow_partial),
        ("check_model_state", check_model_state),
        ("check_optimizer_state", check_optimizer_state),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a boolean")
    # Named extras are opt-in. Missing/malformed evidence must never be
    # silently ignored, even when other enabled observables match.
    if isinstance(check_extras, (str, bytes)):
        raise TypeError("check_extras must be a sequence of nonempty field names")
    extra_names = tuple(check_extras) if check_extras is not None else ()
    if any(not isinstance(name, str) or not name for name in extra_names):
        raise ValueError("check_extras field names must be nonempty strings")
    extra_names = tuple(dict.fromkeys(extra_names))
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    normalized = []
    for label, trajectory in (("uninterrupted", uninterrupted), ("resumed", resumed)):
        records = []
        for position, record in enumerate(trajectory.steps):
            try:
                if isinstance(record.step, bool):
                    raise TypeError("bool is not a step index")
                step = operator.index(record.step)
            except TypeError:
                detail = f"{label} record {position}: step must be an integer without coercion or truncation"
                return CompareReport(passed=False, steps_compared=0, missing_fields=[detail], summary=f"FAIL: {detail}")
            records.append(replace(record, step=step))
        if any(right.step <= left.step for left, right in zip(records, records[1:])):
            detail = f"{label} step indices must be strictly increasing in recorded order"
            return CompareReport(passed=False, steps_compared=0, missing_fields=[detail], summary=f"FAIL: {detail}")
        normalized.append(records)
    a, b = normalized
    n = min(len(a), len(b))
    missing: List[str] = []
    if len(a) != len(b):
        missing.append(
            f"step count differs: {len(a)} vs {len(b)} (compared first {n})"
        )

    state_expected = check_model_state and any(record.model_fingerprint is not None for record in a + b)
    optimizer_expected = check_optimizer_state and any(record.optimizer_fingerprint is not None for record in a + b)
    one_sided: set = set()
    if not any((check_loss, check_batch, check_grad_norm)) and not (
        state_expected or optimizer_expected or extra_names
    ):
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
                    value_a=sa.step,
                    value_b=sb.step,
                    detail="step indices misaligned between runs",
                ),
                missing_fields=missing,
                summary=f"DIVERGED at record {i}: step indices misaligned",
            )
        enabled_fields = (
            ("loss", check_loss),
            ("batch_fingerprint", check_batch),
            ("grad_norm", check_grad_norm),
            ("model_fingerprint", check_model_state),
            ("optimizer_fingerprint", check_optimizer_state),
        )
        if not any(
            enabled and getattr(sa, name) is not None and getattr(sb, name) is not None
            for name, enabled in enabled_fields
        ) and not extra_names:
            missing.append(f"no comparable enabled fields at step {sa.step}")
        for name in extra_names:
            values = []
            for label, record in (("uninterrupted", sa), ("resumed", sb)):
                if not isinstance(record.extras, dict) or name not in record.extras:
                    missing.append(f"extra {name!r} missing in {label} at step {sa.step}")
                    values.append(None)
                    continue
                value = record.extras[name]
                try:
                    valid = (not isinstance(value, bool)
                             and isinstance(value, (int, float))
                             and math.isfinite(value))
                except (OverflowError, ValueError, TypeError):
                    valid = False
                if not valid:
                    missing.append(f"extra {name!r} must be a finite number in {label} at step {sa.step}")
                    values.append(None)
                else:
                    values.append(value)
            if all(value is not None for value in values):
                va, vb = values
                # Accepted JSON integers may exceed float's exact range.
                # Mixed subtraction would round before testing, and a large
                # integer difference may overflow math.isfinite. Compare the
                # represented values exactly against the same tolerance rule.
                from fractions import Fraction

                close = abs(Fraction(va) - Fraction(vb)) <= (
                    Fraction(atol) + Fraction(rtol) * abs(Fraction(vb)))
                if not close:
                    return CompareReport(
                        passed=False, steps_compared=i,
                        first_divergence=Divergence(
                            step=sa.step, field=f"extras.{name}",
                            value_a=va, value_b=vb,
                            detail=f"recorded extra {name!r} differs at step {sa.step}: {va} vs {vb} (rtol={rtol}, atol={atol})",
                        ), missing_fields=missing,
                        summary=f"DIVERGED at step {sa.step}: extra {name!r} {va} vs {vb}",
                    )
        if check_batch:
            if any(fp is not None and (not isinstance(fp, str) or not fp.startswith("v2:")) for fp in (sa.batch_fingerprint, sb.batch_fingerprint)):
                if "legacy_fingerprint" not in one_sided:
                    one_sided.add("legacy_fingerprint")
                    missing.append(
                        "unversioned batch fingerprints or invalid batch_fingerprint types cannot verify structure; "
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
        if check_model_state:
            fingerprints = (sa.model_fingerprint, sb.model_fingerprint)
            if state_expected and all(fp is None for fp in fingerprints):
                missing.append(f"model_fingerprint missing in both runs at step {sa.step}")
            if any(fp is not None and (not isinstance(fp, str) or not fp.startswith("v2:")) for fp in fingerprints):
                if "model_fingerprint" not in one_sided:
                    one_sided.add("model_fingerprint")
                    missing.append("model_fingerprint must use the supported v2 content encoding")
            if (sa.model_fingerprint is None) != (sb.model_fingerprint is None):
                missing.append(f"model_fingerprint recorded in only one run at step {sa.step}")
            elif sa.model_fingerprint is not None and sa.model_fingerprint != sb.model_fingerprint:
                return CompareReport(
                    passed=False, steps_compared=i,
                    first_divergence=Divergence(
                        step=sa.step, field="model_state", value_a=None, value_b=None,
                        detail="model-state content fingerprints differ (exact-bit comparison)",
                    ), missing_fields=missing,
                    summary=f"DIVERGED at step {sa.step}: model state differs",
                )
        if check_optimizer_state:
            fingerprints = (sa.optimizer_fingerprint, sb.optimizer_fingerprint)
            if optimizer_expected and all(fp is None for fp in fingerprints):
                missing.append(f"optimizer_fingerprint missing in both runs at step {sa.step}")
            if any(fp is not None and (not isinstance(fp, str) or re.fullmatch(r"v2:[0-9a-f]{16}", fp) is None) for fp in fingerprints):
                if "optimizer_fingerprint" not in one_sided:
                    one_sided.add("optimizer_fingerprint")
                    missing.append("optimizer_fingerprint must use the supported v2 content encoding")
            if (sa.optimizer_fingerprint is None) != (sb.optimizer_fingerprint is None):
                missing.append(f"optimizer_fingerprint recorded in only one run at step {sa.step}")
            elif (sa.optimizer_fingerprint is not None
                  and all(isinstance(fp, str) for fp in fingerprints)
                  and sa.optimizer_fingerprint != sb.optimizer_fingerprint):
                return CompareReport(
                    passed=False, steps_compared=i,
                    first_divergence=Divergence(
                        step=sa.step, field="optimizer_state", value_a=None, value_b=None,
                        detail="optimizer-state content fingerprints differ (exact-bit comparison)",
                    ), missing_fields=missing,
                    summary=f"DIVERGED at step {sa.step}: optimizer state differs",
                )
        for fname, sa_v, sb_v in (
            ("loss", sa.loss if check_loss else None, sb.loss if check_loss else None),
            (
                "grad_norm",
                sa.grad_norm if check_grad_norm else None,
                sb.grad_norm if check_grad_norm else None,
            ),
        ):
            invalid = False
            for label, value in (("uninterrupted", sa_v), ("resumed", sb_v)):
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    missing.append(f"{fname} must be a number in {label} at step {sa.step}")
                    invalid = True
            if invalid:
                continue
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
            + ("; model-state fingerprints match exactly" if state_expected else "")
            + ("; optimizer-state fingerprints match exactly" if optimizer_expected else "")
            + (f"; compared extras {', '.join(repr(name) for name in extra_names)}" if extra_names else "")
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
