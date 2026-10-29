"""resumecheck - verify that checkpoint-and-resume matches uninterrupted training.

The question this library answers: "If my training job crashes at step k and
I resume from the checkpoint, do I get the same model I would have gotten by
training straight through?" Silent state-restoration bugs (lost dataloader
position, wrong RNG, stale optimizer internals) corrupt long training runs
without ever raising an error.
"""

from .recorder import TrajectoryRecorder, Trajectory, StepRecord
from .compare import CompareReport, Divergence, compare_trajectories

__version__ = "0.2.0"

__all__ = [
    "TrajectoryRecorder",
    "Trajectory",
    "StepRecord",
    "CompareReport",
    "Divergence",
    "compare_trajectories",
    "__version__",
]
