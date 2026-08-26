"""Memory-safety helpers: avoid RAM/VRAM blow-ups and survive CUDA OOM.

The training scripts run on a shared H200 **MIG slice** (~40 GB), where an
out-of-memory event would otherwise crash the whole run. This module provides
three guards, all framework-thin and torch-imported lazily so the file imports
without torch (matching the rest of ``training``):

1. :func:`configure` — reduce fragmentation-driven OOM (``expandable_segments``)
   and optionally cap the process to a fraction of VRAM so it fails *early and
   catchably* instead of taking the device down.
2. :class:`AdaptiveTrainStep` — run one optimizer step; on a CUDA OOM, empty the
   cache and **retry the batch split into more micro-batches** with gradient
   accumulation. The effective update is unchanged; only peak activation memory
   shrinks. The working split is remembered so later steps don't re-hit OOM.
3. :func:`safe_infer` — the same shrink-on-OOM idea for a forward-only pass
   (validation), returning the concatenated logits.

:func:`safe_num_workers` keeps the DataLoader from exhausting system RAM.
"""

from __future__ import annotations

import gc
import os
import shutil
from collections.abc import Callable, Sequence
from typing import Any

# NOTE: torch is imported lazily inside the functions that need it so this
# module imports on a CPU-only, torch-less machine (CI, lint) just like
# ``training.datasets``.

_ALLOC_CONF_KEY = "PYTORCH_CUDA_ALLOC_CONF"


def configure(device: Any, *, vram_fraction: float | None = None, verbose: bool = True) -> None:
    """Set memory-friendly CUDA defaults and (optionally) cap VRAM.

    ``expandable_segments:True`` markedly reduces fragmentation OOMs; it is only
    honoured if set before the CUDA allocator initialises, so we also export it
    from ``run_tmux.sh``. Setting it here is a best-effort backstop. When
    ``vram_fraction`` is given (0–1], the process is capped to that fraction of
    the slice's memory, so it hits a *catchable* OOM (handled by
    :class:`AdaptiveTrainStep`) well before starving the device.
    """
    os.environ.setdefault(_ALLOC_CONF_KEY, "expandable_segments:True")
    tune_dataloader_sharing(verbose=verbose)
    if getattr(device, "type", None) != "cuda":
        return
    import torch

    if vram_fraction is not None:
        frac = float(vram_fraction)
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"vram_fraction must be in (0, 1], got {frac}")
        index = getattr(device, "index", None) or 0
        torch.cuda.set_per_process_memory_fraction(frac, index)
    if verbose:
        log_memory(device, tag="configure")


def shm_free_bytes() -> int | None:
    """Free bytes on ``/dev/shm``; ``None`` if it can't be read."""
    try:
        return shutil.disk_usage("/dev/shm").free
    except OSError:
        return None


def tune_dataloader_sharing(*, min_shm_mb: int = 512, verbose: bool = True) -> None:
    """Avoid the container ``/dev/shm`` DataLoader crash.

    Containers (Docker, k8s) often cap ``/dev/shm`` at 64 MB. PyTorch's default
    ``file_descriptor`` tensor-sharing passes worker batches through ``/dev/shm``,
    so ``num_workers > 0`` dies with ``unable to allocate shared memory``. When
    ``/dev/shm`` is small we switch to ``file_system`` sharing, which uses regular
    temp files instead — workers keep running, no crash. No-op if torch is absent.
    """
    free = shm_free_bytes()
    if free is None or free >= min_shm_mb * 1024 * 1024:
        return
    try:
        import torch.multiprocessing as mp

        mp.set_sharing_strategy("file_system")
    except Exception:  # torch absent or strategy unavailable
        return
    if verbose:
        print(
            f"[mem] /dev/shm is small ({free / 1e6:.0f} MB) — using file_system tensor "
            "sharing so DataLoader workers don't crash on shared memory"
        )


def is_oom_error(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA/CPU out-of-memory error we can recover from."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        text = str(exc).lower()
        return (
            "out of memory" in text
            or "cuda oom" in text
            or ("cublas" in text and "alloc" in text)
        )
    return False


def empty_cache() -> None:
    """Release cached allocations back to the driver and collect garbage."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _available_ram_gb() -> float | None:
    """Available system RAM in GiB from ``/proc/meminfo``; ``None`` if unknown."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    kib = float(line.split()[1])
                    return kib / (1024.0 * 1024.0)
    except OSError:
        return None
    return None


def _cuda_mem_gb(device: Any) -> tuple[float, float, float] | None:
    """``(allocated, reserved, total)`` GiB for ``device``; ``None`` off-CUDA."""
    if getattr(device, "type", None) != "cuda":
        return None
    try:
        import torch

        index = getattr(device, "index", None) or 0
        gib = 1024.0**3
        allocated = torch.cuda.memory_allocated(index) / gib
        reserved = torch.cuda.memory_reserved(index) / gib
        total = torch.cuda.get_device_properties(index).total_memory / gib
        return allocated, reserved, total
    except Exception:
        return None


def log_memory(device: Any, *, tag: str = "") -> None:
    """Print a one-line RAM/VRAM snapshot (best effort)."""
    prefix = f"[mem:{tag}]" if tag else "[mem]"
    ram = _available_ram_gb()
    ram_str = f"RAM avail {ram:.1f} GiB" if ram is not None else "RAM avail n/a"
    cuda = _cuda_mem_gb(device)
    if cuda is not None:
        alloc, reserved, total = cuda
        print(f"{prefix} {ram_str} | VRAM {alloc:.1f}/{reserved:.1f} reserved / {total:.1f} GiB")
    else:
        print(f"{prefix} {ram_str} | VRAM n/a (cpu)")


def safe_num_workers(
    requested: int,
    *,
    per_worker_gb: float = 1.5,
    reserve_gb: float = 2.0,
    min_shm_mb: int = 512,
) -> int:
    """Cap DataLoader workers by available RAM, CPU count, and ``/dev/shm``.

    Each worker holds a copy of the dataset's Python state and prefetch buffers;
    on a busy shared box, honouring a large ``--num-workers`` can OOM the host.
    We budget ``per_worker_gb`` per worker out of (available RAM − ``reserve_gb``)
    and never exceed the CPU count. Returns at least 0 (0 = load in the main
    process, always safe).

    Crucially, worker processes pass batches through ``/dev/shm``; containers cap
    it at ~64 MB, which makes *any* ``num_workers > 0`` crash with "unable to
    allocate shared memory" regardless of the tensor-sharing strategy. When
    ``/dev/shm`` is that small we force 0 workers — the only reliable fix short of
    resizing the container's shared memory.
    """
    requested = max(0, int(requested))
    shm = shm_free_bytes()
    if requested > 0 and shm is not None and shm < min_shm_mb * 1024 * 1024:
        print(
            f"[mem] /dev/shm is only {shm / 1e6:.0f} MB — forcing --num-workers 0 "
            "(DataLoader workers can't share memory here; increase the container's "
            "--shm-size to re-enable them)"
        )
        return 0
    cpu = os.cpu_count() or 1
    cap = min(requested, cpu)
    ram = _available_ram_gb()
    if ram is not None:
        budget = max(0.0, ram - reserve_gb)
        cap = min(cap, int(budget // max(0.1, per_worker_gb)))
    return max(0, cap)


def slice_bounds(batch_size: int, chunks: int) -> list[tuple[int, int]]:
    """Split ``range(batch_size)`` into ``chunks`` near-equal ``(lo, hi)`` spans."""
    batch_size = int(batch_size)
    chunks = max(1, min(int(chunks), batch_size))
    base, extra = divmod(batch_size, chunks)
    bounds: list[tuple[int, int]] = []
    lo = 0
    for i in range(chunks):
        hi = lo + base + (1 if i < extra else 0)
        if hi > lo:
            bounds.append((lo, hi))
        lo = hi
    return bounds


class AdaptiveTrainStep:
    """One optimizer step that shrinks its micro-batch on CUDA OOM.

    Give it the ``optimizer`` and AMP ``scaler``; call :meth:`run` with the batch
    tensors and a ``forward_fn`` that maps a slice of those tensors to a scalar
    (mean) loss. The step accumulates gradients over ``micro`` micro-batches,
    each loss weighted by its share of the batch so the accumulated gradient
    equals the full-batch mean. On OOM it empties the cache, doubles ``micro``
    (halving activation memory), and retries the *whole* step — gradients are
    re-zeroed first, so the update stays correct. ``micro`` only ratchets up, so
    subsequent steps skip straight to the working split.
    """

    def __init__(
        self, optimizer: Any, scaler: Any, *, max_microbatches: int, verbose: bool = True
    ) -> None:
        self.optimizer = optimizer
        self.scaler = scaler
        self.max_microbatches = max(1, int(max_microbatches))
        self.micro = 1
        self.verbose = verbose

    def run(
        self,
        tensors: Sequence[Any],
        forward_fn: Callable[..., Any],
    ) -> float:
        """Do one step over ``tensors``; return the batch mean loss (float).

        ``forward_fn(*slices)`` receives one slice per tensor and must return the
        mean loss for that slice (autocast applied inside the closure). Raises
        :class:`MemoryError` only if a single-sample micro-batch still OOMs.
        """
        batch_size = int(tensors[0].shape[0])
        if batch_size == 0:
            return 0.0
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            micro = min(self.micro, batch_size)
            try:
                total = 0.0
                for lo, hi in slice_bounds(batch_size, micro):
                    weight = (hi - lo) / batch_size
                    loss = forward_fn(*[t[lo:hi] for t in tensors])
                    self.scaler.scale(loss * weight).backward()
                    total += float(loss.detach().item()) * weight
                self.scaler.step(self.optimizer)
                self.scaler.update()
                return total
            except RuntimeError as exc:
                if not is_oom_error(exc):
                    raise
                empty_cache()
                if micro >= batch_size or micro >= self.max_microbatches:
                    raise MemoryError(
                        f"CUDA OOM persists at micro-batch {micro} of {batch_size}; "
                        "lower --input-size or --batch-size, or set --vram-fraction."
                    ) from exc
                self.micro = min(self.max_microbatches, batch_size, micro * 2)
                if self.verbose:
                    print(
                        f"[mem] CUDA OOM — retrying batch in {self.micro} micro-batches "
                        "(gradient accumulation; effective batch unchanged)"
                    )


def safe_infer(model: Any, images: Any, *, max_chunks: int | None = None) -> Any:
    """Forward pass that halves its chunk on OOM; returns concatenated logits.

    For validation/inference (no autograd graph to accumulate). Call inside a
    ``torch.no_grad()`` context as usual.
    """
    import torch

    batch_size = int(images.shape[0])
    if batch_size == 0:
        return model(images)
    limit = min(int(max_chunks), batch_size) if max_chunks else batch_size
    chunks = 1
    while True:
        try:
            outputs = [model(images[lo:hi]) for lo, hi in slice_bounds(batch_size, chunks)]
            return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            empty_cache()
            if chunks >= limit:
                raise MemoryError(
                    f"CUDA OOM in forward pass at {chunks} chunks of {batch_size}; "
                    "lower --input-size or --batch-size."
                ) from exc
            chunks = min(limit, chunks * 2)
