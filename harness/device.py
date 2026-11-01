"""The only module that names concrete devices. Everything else asks here for devices, dtypes and memory info."""
from __future__ import annotations

import gc
from typing import Optional

import torch

DTYPES = {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
          "float16": torch.float16, "fp16": torch.float16}
GiB = 2 ** 30


def dtype_of(name) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    try:
        return DTYPES[str(name).replace("torch.", "")]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}") from None


def dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def resolve_device(spec: Optional[str] = "auto") -> torch.device:
    """'auto' -> first available of CUDA, MPS, CPU. An explicit accelerator that is unavailable is an error."""
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())      # indexed: some CUDA APIs reject "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"profile asks for {spec} but CUDA is not available")
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if dev.type == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError(f"profile asks for {spec} but MPS is not available")
    return dev


def cpu() -> torch.device:
    return torch.device("cpu")


def is_accel(device: torch.device) -> bool:
    return torch.device(device).type != "cpu"


def is_cuda(device: torch.device) -> bool:
    return torch.device(device).type == "cuda"


def cuda_available() -> bool:
    return torch.cuda.is_available()


def device_info(device: torch.device) -> dict:
    device = torch.device(device)
    out = {"device": str(device), "torch": torch.__version__, "torch_cuda": torch.version.cuda}
    if is_cuda(device):
        p = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        out.update(gpu_name=p.name, gpu_total_gb=round(total / GiB, 2), gpu_free_gb=round(free / GiB, 2),
                   capability=f"{p.major}.{p.minor}", cudnn=torch.backends.cudnn.version())
    return out


def free_bytes(device: torch.device) -> Optional[int]:
    """Free accelerator memory in bytes; None on CPU (callers use RAM checks instead)."""
    return torch.cuda.mem_get_info(device)[0] if is_cuda(device) else None


def total_bytes(device: torch.device) -> Optional[int]:
    return torch.cuda.mem_get_info(device)[1] if is_cuda(device) else None


def cap_memory(device: torch.device, cap_gb: Optional[float]) -> None:
    """Limit this process's CUDA allocator to cap_gb (no-op on CPU / without a cap)."""
    if cap_gb and is_cuda(device):
        device = torch.device(device)
        # set_per_process_memory_fraction needs an index: torch.device("cuda") raises ValueError (RUN, 2026-11-01)
        idx = device.index if device.index is not None else torch.cuda.current_device()
        total = torch.cuda.get_device_properties(idx).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb * GiB / total), idx)


def empty_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_peak(device: torch.device) -> None:
    if is_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)


def peak_bytes(device: torch.device) -> int:
    return torch.cuda.max_memory_allocated(device) if is_cuda(device) else 0


def is_oom(e: BaseException) -> bool:
    """Accelerator out-of-memory (incl. cuBLAS allocation failures), or our test double FakeOOM."""
    if isinstance(e, FakeOOM) or isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    msg = str(e)
    return isinstance(e, RuntimeError) and ("out of memory" in msg or "CUBLAS_STATUS_ALLOC_FAILED" in msg)


class FakeOOM(RuntimeError):
    """Raised by tests to exercise OOM backoff on machines without a GPU."""


def sync(device: torch.device) -> None:
    """Wait for queued accelerator work (for timing)."""
    if is_cuda(device):
        torch.cuda.synchronize(device)


def allocated_bytes(device: torch.device) -> int:
    return torch.cuda.memory_allocated(device) if is_cuda(device) else 0
