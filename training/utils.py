"""Training utilities: seeding, device/AMP, schedules, EMA, checkpoints, config.

These helpers are deliberately framework-thin so the trainer scripts stay
readable. Everything is typed and side-effect-free unless documented otherwise.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve ``'auto' | 'cpu' | 'cuda' | 'cuda:N'`` to a ``torch.device``."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Total number of (trainable) parameters in ``model``."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def cosine_warmup_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Linear warmup then cosine decay to ``min_lr_ratio`` of the base LR.

    Stepped once per optimizer step (not per epoch). ``total_steps`` is the total
    number of optimizer updates over the whole run.
    """
    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps + 1)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


@dataclass
class AverageMeter:
    """Running mean of a scalar stream."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


class ModelEma:
    """Exponential-moving-average copy of a model (the mean-teacher stabiliser).

    The EMA weights are updated after each optimizer step and are used only for
    evaluation / consistency targets; they are never optimized directly. This is
    the ``EMA-teacher`` in the methods, distinct from the rule ``teacher``.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.module = deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.decay
        for ema_p, p in zip(self.module.parameters(), model.parameters(), strict=True):
            ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for ema_b, b in zip(self.module.buffers(), model.buffers(), strict=True):
            ema_b.copy_(b)


class CheckpointManager:
    """Save training state and track the best checkpoint by a monitored metric."""

    def __init__(self, out_dir: Path, *, mode: str = "max") -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_metric = -math.inf if mode == "max" else math.inf
        self.best_path = self.out_dir / "best.pt"
        self.last_path = self.out_dir / "last.pt"

    def _is_better(self, metric: float) -> bool:
        if self.mode == "max":
            return metric > self.best_metric
        return metric < self.best_metric

    def save(self, state: Mapping[str, Any], *, metric: float) -> bool:
        """Write ``last.pt`` always; overwrite ``best.pt`` on improvement.

        Returns ``True`` when this checkpoint became the new best.
        """
        payload = dict(state)
        payload["metric"] = float(metric)
        torch.save(payload, self.last_path)
        if self._is_better(metric):
            self.best_metric = float(metric)
            torch.save(payload, self.best_path)
            return True
        return False


def make_summary_writer(log_dir: Path) -> Any:
    """Return a TensorBoard ``SummaryWriter``, or a no-op stub if unavailable.

    The stub exposes ``add_scalar`` / ``add_scalars`` / ``flush`` / ``close`` so
    call sites never need to branch on whether TensorBoard is installed.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return _NullWriter()
    return SummaryWriter(log_dir=str(log_dir))


class _NullWriter:
    """A do-nothing stand-in for ``SummaryWriter``."""

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add_scalars(self, *args: Any, **kwargs: Any) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict (empty file → empty dict)."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping, got {type(data).__name__}")
    return data


def merge_cli_over_config(
    config: Mapping[str, Any],
    cli: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay CLI values (non-``None``) on top of a config mapping."""
    merged = dict(config)
    for key, value in cli.items():
        if value is not None:
            merged[key] = value
    return merged


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def autocast_context(device: torch.device, *, enabled: bool) -> torch.autocast:
    """AMP autocast context, active only on CUDA (fp16) when ``enabled``."""
    use = enabled and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use)


def make_grad_scaler(device: torch.device, *, enabled: bool) -> torch.amp.GradScaler:
    """A ``GradScaler`` that is a no-op unless AMP runs on CUDA."""
    return torch.amp.GradScaler(device.type, enabled=enabled and device.type == "cuda")


def iter_batches(n_items: int, batch_size: int) -> Iterator[range]:
    """Yield contiguous index ranges of at most ``batch_size`` items."""
    for start in range(0, n_items, batch_size):
        yield range(start, min(start + batch_size, n_items))
