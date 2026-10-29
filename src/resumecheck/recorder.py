"""Record per-step training state into a comparable trajectory."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def fingerprint_tensor(obj: Any) -> str:
    """Stable content fingerprint of a batch or tensor-like object.

    Handles tensors/arrays (via .tobytes()), scalars, and nested
    lists/tuples/dicts. Intended for test-scale data, not production
    throughput - hashing large batches every step is expensive.
    """
    parts: List[bytes] = []

    def walk(x: Any) -> None:
        if hasattr(x, "detach"):  # torch.Tensor
            parts.append(x.detach().cpu().contiguous().numpy().tobytes())
        elif hasattr(x, "tobytes") and not isinstance(x, (bytes, str)):
            try:
                parts.append(x.tobytes())
                return
            except Exception:
                pass
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                walk(k)
                walk(x[k])
            return
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
            return
        elif isinstance(x, (bytes, str)):
            parts.append(str(x).encode())
            return
        else:
            parts.append(repr(x).encode())
            return

    walk(obj)
    return hashlib.sha256(b"|".join(parts)).hexdigest()[:16]


@dataclass
class StepRecord:
    """One training step's observable state."""

    step: int
    loss: Optional[float] = None
    batch_fingerprint: Optional[str] = None
    grad_norm: Optional[float] = None
    extras: Dict[str, float] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Ordered step records plus run metadata."""

    name: str
    steps: List[StepRecord] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "name": self.name,
                    "meta": self.meta,
                    "steps": [asdict(s) for s in self.steps],
                },
                indent=1,
            )
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Trajectory":
        d = json.loads(Path(path).read_text())
        return cls(
            name=d["name"],
            meta=d.get("meta", {}),
            steps=[StepRecord(**s) for s in d["steps"]],
        )


class TrajectoryRecorder:
    """Append-only recorder; one per training run.

    Usage in a manual loop:
        rec = TrajectoryRecorder("run-a")
        for step, batch in enumerate(loader):
            loss = train_step(batch)
            rec.record(step, loss=float(loss), batch=batch)
        rec.trajectory.save("run-a.json")
    """

    def __init__(self, name: str = "run", **meta: Any) -> None:
        self.trajectory = Trajectory(name=name, meta=dict(meta))

    def record(
        self,
        step: int,
        loss: Optional[float] = None,
        batch: Any = None,
        grad_norm: Optional[float] = None,
        **extras: float,
    ) -> None:
        fp = fingerprint_tensor(batch) if batch is not None else None
        self.trajectory.steps.append(
            StepRecord(
                step=int(step),
                loss=None if loss is None else float(loss),
                batch_fingerprint=fp,
                grad_norm=None if grad_norm is None else float(grad_norm),
                extras={k: float(v) for k, v in extras.items()},
            )
        )
