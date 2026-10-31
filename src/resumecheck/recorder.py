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

    The encoding is unambiguous: every part is length-prefixed and tagged
    with its structural role; containers carry size and type, and tensor
    parts carry dtype and shape, so
    supported containers, dtypes, and shapes do not alias merely because
    their payload bytes match. This is a diagnostic fingerprint, not a
    collision-proof attestation or a serializer for arbitrary objects.
    """
    parts: List[bytes] = []

    def emit(tag: bytes, payload: bytes) -> None:
        parts.append(tag + str(len(payload)).encode() + b":" + payload)

    def walk(x: Any) -> None:
        if hasattr(x, "detach"):  # torch.Tensor
            arr = x.detach().cpu().contiguous().numpy()
            emit(b"tensor", f"{arr.dtype}|{arr.shape}|".encode() + arr.tobytes())
            return
        if hasattr(x, "tobytes") and not isinstance(x, (bytes, str)):
            try:
                dtype = getattr(x, "dtype", None)
                shape = getattr(x, "shape", None)
                header = f"{dtype}|{shape}|".encode()
                emit(b"array", header + x.tobytes())
                return
            except Exception:
                pass
        if isinstance(x, dict):
            emit(b"dict", str(len(x)).encode())
            for k in sorted(x, key=str):
                walk(k)
                walk(x[k])
            return
        if isinstance(x, (list, tuple)):
            emit(b"list" if isinstance(x, list) else b"tuple", str(len(x)).encode())
            for v in x:
                walk(v)
            return
        if isinstance(x, bytes):
            emit(b"bytes", x)
            return
        if isinstance(x, str):
            emit(b"str", x.encode())
            return
        emit(b"repr", repr(x).encode())

    walk(obj)
    return "v2:" + hashlib.sha256(b"".join(parts)).hexdigest()[:16]


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
