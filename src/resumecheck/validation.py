"""Shared input contracts for finite adapter experiments."""
import operator


def validate_step_limits(max_steps, resume_at):
    """Accept integer-index values, never coerce fractions or booleans."""
    values = []
    for name, value in (("max_steps", max_steps), ("resume_at", resume_at)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, not bool")
        try:
            values.append(int(operator.index(value)))
        except TypeError as error:
            raise TypeError(f"{name} must be an integer without coercion or truncation") from error
    max_steps, resume_at = values
    if not 0 < resume_at < max_steps:
        raise ValueError("resume_at must be in (0, max_steps)")
    return max_steps, resume_at
