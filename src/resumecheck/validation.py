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


def validate_loader_shape(dataset, batch_size):
    """Normalize a positive batch index and require a nonempty sized dataset."""
    if isinstance(batch_size, bool):
        raise TypeError("batch_size must be a positive integer, not bool")
    try:
        batch_size = operator.index(batch_size)
    except TypeError as error:
        raise TypeError("batch_size must be a positive integer without coercion") from error
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    size = len(dataset)
    if size < 1:
        raise ValueError("Accelerate requires a nonempty sized training dataset")
    return size, batch_size
