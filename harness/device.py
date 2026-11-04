"""The only module that names concrete devices. Everything else asks here for devices, dtypes and memory info."""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

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


# --------------------------------------------------------------------------- machine probe (plan step 3)
def _meminfo() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return out


def _physical_cores() -> Optional[int]:
    """Distinct (physical id, core id) pairs in /proc/cpuinfo; None if unknown."""
    try:
        pairs, phys, core = set(), None, None
        for line in Path("/proc/cpuinfo").read_text().splitlines() + [""]:
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
            elif not line.strip():
                if core is not None:
                    pairs.add((phys, core))
                phys, core = None, None
        return len(pairs) or None
    except OSError:
        return None


def visible_gpu_ids() -> Optional[List[str]]:
    """CUDA_VISIBLE_DEVICES as a list (None = not set; [] = none visible)."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is None:
        return None
    return [x.strip() for x in v.split(",") if x.strip() and x.strip() != "-1"]


def _nvidia_smi_gpus() -> List[dict]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,compute_cap,driver_version",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    gpus = []
    for line in r.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 6:
            continue
        gpus.append({"index": int(f[0]), "uuid": f[1], "name": f[2], "total_bytes": int(float(f[3])) * 2 ** 20,
                     "free_bytes": int(float(f[4])) * 2 ** 20, "compute_cap": f[5],
                     "driver": f[6] if len(f) > 6 else None})
    vis = visible_gpu_ids()
    if vis is not None:                      # keep the visible ones, in CUDA_VISIBLE_DEVICES order
        by = {str(g["index"]): g for g in gpus} | {g["uuid"]: g for g in gpus}
        gpus = [by[v] for v in vis if v in by]
    return gpus


def probe_machine(cache_root: Optional[Path] = None) -> dict:
    """What this machine has, measured without initialising CUDA (nvidia-smi), so a scheduler can call it.
    HL_FAKE_PROBE=<json file> replaces the measurement (tests, dry runs on a machine without the GPUs)."""
    fake = os.environ.get("HL_FAKE_PROBE")
    if fake:
        d = json.loads(Path(fake).read_text())
        d.setdefault("source", f"fake:{fake}")
        return d
    mi = _meminfo()
    logical = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = logical
    du = shutil.disk_usage(cache_root) if cache_root and Path(cache_root).exists() else None
    return {"source": "nvidia-smi + /proc", "gpus": _nvidia_smi_gpus(),
            "ram_total_bytes": mi.get("MemTotal"), "ram_available_bytes": mi.get("MemAvailable"),
            "cpu_logical": logical, "cpu_affinity": affinity, "cpu_physical": _physical_cores() or max(1, affinity // 2),
            "disk_free_bytes": du.free if du else None, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


MIN_DRIVER_CU13 = 580      # CUDA 13 wheels (requirements/harness.lock) need driver branch >= 580


def gpu_compat(index: int, n: int = 2048) -> dict:
    """Can the locked torch build run the harness on GPU `index`? Kernels built for its architecture, a bf16 GEMM
    that agrees with fp32, and the bf16 / fp32 GEMM speed ratio (GPUs before compute capability 8.0, e.g. Turing
    RTX 20xx, have no bf16 tensor cores: bf16 runs but is emulated and slow)."""
    import time
    dev = torch.device(f"cuda:{index}")
    cap = torch.cuda.get_device_capability(dev)
    arch = torch.cuda.get_arch_list()
    sm = cap[0] * 10 + cap[1]
    sms = [int(a.split("_")[1]) for a in arch if a.startswith("sm_")]
    ptx = [int(a.split("_")[1]) for a in arch if a.startswith("compute_")]
    kernels = sm in sms or any(p <= sm for p in ptx)
    out = {"index": index, "name": torch.cuda.get_device_name(dev), "compute_cap": f"{cap[0]}.{cap[1]}",
           "torch_arch_list": arch, "kernels_for_arch": kernels, "native_bf16": cap[0] >= 8}
    if not kernels:
        out["ok"] = False
        out["error"] = f"this torch build has no kernels for sm_{sm} (built for {arch})"
        return out
    try:
        g = torch.Generator(device=dev).manual_seed(0)
        a = torch.randn(n, n, device=dev, generator=g)
        b = torch.randn(n, n, device=dev, generator=g)
        ref = a @ b
        lo = (a.bfloat16() @ b.bfloat16()).float()
        out["bf16_rel_err"] = float((lo - ref).norm() / ref.norm())

        def speed(x, y, reps=10):
            x @ y
            torch.cuda.synchronize(dev)
            t = time.perf_counter()
            for _ in range(reps):
                x @ y
            torch.cuda.synchronize(dev)
            return reps * 2 * n ** 3 / (time.perf_counter() - t) / 1e12
        out["fp32_tflops"] = round(speed(a, b), 2)
        out["bf16_tflops"] = round(speed(a.bfloat16(), b.bfloat16()), 2)
        out["ok"] = out["bf16_rel_err"] < 2e-2
        if not out["ok"]:
            out["error"] = f"bf16 GEMM disagrees with fp32 (relative error {out['bf16_rel_err']:.3g})"
    except RuntimeError as e:
        out["ok"], out["error"] = False, f"{type(e).__name__}: {e}"
    finally:
        torch.cuda.empty_cache()
    return out


def device_memory(device: torch.device) -> Optional[dict]:
    """Measured free / total bytes of the device this process computes on (None on CPU)."""
    if not is_cuda(device):
        return None
    free, total = torch.cuda.mem_get_info(device)
    return {"free_bytes": int(free), "total_bytes": int(total), "name": torch.cuda.get_device_properties(device).name}


def set_threads(n: Optional[int]) -> int:
    """Intra-op CPU threads for this process (torch + the BLAS env read by child processes)."""
    if n:
        torch.set_num_threads(int(n))
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[v] = str(int(n))
    return torch.get_num_threads()
