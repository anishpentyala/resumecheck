"""Record per-step training state into a comparable trajectory."""
from __future__ import annotations

import hashlib
import json
import operator
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
            tensor = x.detach().cpu().contiguous()
            try:
                arr = tensor.numpy()
            except TypeError:
                # NumPy cannot represent bfloat16/float8. Keep their exact
                # bits, dtype and logical shape, without a lossy float cast.
                # A clone owns compact storage: views may otherwise include
                # unrelated bytes before/after the logical tensor.
                dtype = str(tensor.dtype)
                if dtype != "torch.bfloat16" and not dtype.startswith("torch.float8_"):
                    raise
                payload = bytes(tensor.clone().untyped_storage())
                emit(b"tensor-raw", f"{dtype}|{tuple(tensor.shape)}|".encode() + payload)
            else:
                # Preserve existing fingerprints for NumPy-supported dtypes.
                emit(b"tensor", f"{arr.dtype}|{arr.shape}|".encode() + arr.tobytes())
            return
        if hasattr(x, "tobytes") and not isinstance(x, (bytes, str)):
            if hasattr(x, "mask") and hasattr(x, "filled"):
                # MaskedArray.tobytes substitutes fill values and discards the
                # mask and hidden data. Even nomask instances carry semantics
                # that the plain-array encoding cannot represent safely.
                raise TypeError(
                    "masked arrays cannot be fingerprinted by raw bytes; "
                    "convert explicitly to a supported container containing "
                    "the data, mask and any relevant fill value"
                )
            dtype = getattr(x, "dtype", None)
            shape = getattr(x, "shape", None)
            if dtype is None or shape is None:
                # array.array and memoryview expose bytes, but their element
                # type/shape live in different attributes. Never silently
                # hash these as None metadata and verify unequal inputs.
                raise TypeError(
                    f"unsupported batch value type {type(x).__name__}; "
                    "raw-byte arrays require dtype and shape metadata; "
                    "convert explicitly to a supported array or container"
                )
            if getattr(dtype, "hasobject", False):
                # Object-array bytes are Python object addresses, not content.
                # The referent can change without changing a single byte here.
                raise TypeError(
                    "object-containing array dtypes cannot be fingerprinted by content; "
                    "convert the batch to supported tensors, arrays or containers"
                )
            try:
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
        if type(x) in (type(None), bool, int, float, complex):
            # Keep scalar fingerprints stable, but never mistake an opaque
            # object's display text for its content. Scalar subclasses can
            # override repr or carry additional state, so require exact types.
            emit(b"repr", repr(x).encode())
            return
        raise TypeError(
            f"unsupported batch value type {type(x).__name__}; "
            "convert the batch to supported tensors, arrays, scalars or containers"
        )

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
        if isinstance(step, bool):
            raise TypeError("step must be an integer, not bool")
        try:
            step = operator.index(step)
        except TypeError as error:
            raise TypeError("step must be an integer without coercion or truncation") from error
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
