"""Memory-safety helpers (``training.memory``) — torch-free unit tests.

These exercise the OOM-recovery logic with injected fakes, so they run in CI
without torch (the module imports torch lazily). The real CUDA path can only be
exercised on a GPU, but the batch-shrinking control flow is validated here.
"""

from __future__ import annotations

import pytest
from training import memory


def test_slice_bounds_partitions_exactly() -> None:
    assert memory.slice_bounds(4, 1) == [(0, 4)]
    assert memory.slice_bounds(4, 2) == [(0, 2), (2, 4)]
    assert memory.slice_bounds(4, 4) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    # Near-equal split when it doesn't divide evenly; spans cover [0, n) once.
    bounds = memory.slice_bounds(5, 2)
    assert bounds[0][0] == 0 and bounds[-1][1] == 5
    assert all(lo < hi for lo, hi in bounds)
    # chunks are clamped to the batch size.
    assert memory.slice_bounds(3, 99) == [(0, 1), (1, 2), (2, 3)]


def test_is_oom_error_recognises_oom() -> None:
    assert memory.is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate ..."))
    assert memory.is_oom_error(MemoryError("host OOM"))
    assert not memory.is_oom_error(RuntimeError("shape mismatch"))
    assert not memory.is_oom_error(ValueError("nope"))


def test_safe_num_workers_is_bounded() -> None:
    import os

    cpu = os.cpu_count() or 1
    assert memory.safe_num_workers(0) == 0
    assert memory.safe_num_workers(-5) == 0
    # Never exceeds the request or the CPU count, never negative.
    for req in (1, 2, 8, 64):
        got = memory.safe_num_workers(req)
        assert 0 <= got <= min(req, cpu)


class _FakeOptimizer:
    def __init__(self) -> None:
        self.steps = 0
        self.zeroed = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zeroed += 1

    def step(self) -> None:
        self.steps += 1


class _FakeLoss:
    """A scalar-ish stand-in supporting ``* weight``, ``.backward``, ``.item``."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __mul__(self, weight: float) -> _FakeLoss:
        return _FakeLoss(self.value * weight)

    def backward(self) -> None:
        return None

    def detach(self) -> _FakeLoss:
        return self

    def item(self) -> float:
        return self.value


class _FakeScaler:
    def scale(self, loss: _FakeLoss) -> _FakeLoss:
        return loss

    def step(self, optimizer: _FakeOptimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


class _FakeTensor:
    """Minimal batch tensor: has ``.shape`` and slice indexing."""

    def __init__(self, size: int) -> None:
        self.shape = (size,)

    def __getitem__(self, key: slice) -> _FakeTensor:
        lo, hi, _ = key.indices(self.shape[0])
        return _FakeTensor(hi - lo)


def test_adaptive_step_shrinks_microbatch_on_oom() -> None:
    opt, scaler = _FakeOptimizer(), _FakeScaler()
    step = memory.AdaptiveTrainStep(opt, scaler, max_microbatches=4, verbose=False)

    # Forward OOMs whenever a slice holds more than one sample; succeeds at size 1.
    def forward_fn(images: _FakeTensor) -> _FakeLoss:
        if images.shape[0] > 1:
            raise RuntimeError("CUDA out of memory")
        return _FakeLoss(1.0)

    loss = step.run((_FakeTensor(4),), forward_fn)
    # It ratcheted 1 -> 2 -> 4 micro-batches and finally succeeded.
    assert step.micro == 4
    assert opt.steps == 1
    # Weighted sum of four size-1 slices (each weight 1/4, loss 1.0) == 1.0.
    assert loss == pytest.approx(1.0)


def test_adaptive_step_raises_when_oom_persists() -> None:
    opt, scaler = _FakeOptimizer(), _FakeScaler()
    step = memory.AdaptiveTrainStep(opt, scaler, max_microbatches=4, verbose=False)

    def always_oom(images: _FakeTensor) -> _FakeLoss:
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(MemoryError):
        step.run((_FakeTensor(2),), always_oom)
    assert opt.steps == 0  # never stepped the optimizer
