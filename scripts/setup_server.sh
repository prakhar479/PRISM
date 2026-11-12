#!/usr/bin/env bash
# One-time setup of a run checkout (the `run` branch) on a GPU server. Safe to re-run.
#   bash scripts/setup_server.sh            # venv + locked packages + env check + golden fixtures + machine probe
# Needs: python 3.12, NVIDIA driver >= 560 and GPUs of compute capability >= 5.0 (Pascal GTX 10xx and older need
# the CUDA 12.6 build, chosen automatically; Turing RTX 20xx and newer can use either),
# ~70 GB free disk, HF_TOKEN later. The torch build follows the driver (both are torch 2.14.1, same other pins):
#   driver >= 580    requirements/harness.lock        (CUDA 13.0 build, torch 2.14.1+cu130)
#   driver 560-579   requirements/harness-cu126.lock  (CUDA 12.6 build, torch 2.14.1+cu126)
#   any GPU < cc 7.5 requirements/harness-cu126.lock  (CUDA 13 has no Pascal / Volta / Maxwell kernels)
# Override with LOCK=requirements/<file>.lock TORCH_INDEX=https://download.pytorch.org/whl/<cuXYZ>.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv-harness}
PYVER=${PYVER:-3.12}

echo "== 0/5 NVIDIA driver and GPUs"
DRV=0
MINCC=9.9
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv,noheader
  MINCC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -n | head -1)
  TOOOLD=$(nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader | awk -F', ' '$3+0 < 5.0')
  if [ -n "$TOOOLD" ]; then
    echo "ERROR: these GPUs are older than compute capability 5.0; no locked torch build has kernels for them:"
    echo "$TOOOLD"
    exit 1
  fi
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
else
  echo "WARNING: nvidia-smi not found"
fi
if [ -z "${LOCK:-}" ]; then
  PRE_TURING=$(awk -v c="$MINCC" 'BEGIN { print (c + 0 < 7.5) ? 1 : 0 }')
  if [ "$PRE_TURING" = 1 ] && [ "$DRV" -lt 560 ]; then
    echo "ERROR: GPUs of compute capability $MINCC need the CUDA 12.6 build, which needs driver >= 560 (have $DRV)."
    exit 1
  elif [ "$PRE_TURING" = 1 ]; then
    # CUDA 13 dropped Maxwell / Pascal / Volta; the CUDA 12.6 build has sm_50 sm_60 sm_70 kernels (any driver >= 560)
    LOCK=requirements/harness-cu126.lock; TORCH_INDEX=https://download.pytorch.org/whl/cu126
  elif [ "$DRV" -ge 580 ] || [ "$DRV" -eq 0 ]; then
    LOCK=requirements/harness.lock; TORCH_INDEX=https://download.pytorch.org/whl/cu130
  elif [ "$DRV" -ge 560 ]; then
    LOCK=requirements/harness-cu126.lock; TORCH_INDEX=https://download.pytorch.org/whl/cu126
  else
    echo "ERROR: driver $DRV is older than 560: no locked torch build runs on it. Ask the admin for driver >= 560."
    exit 1
  fi
fi
TORCH_INDEX=${TORCH_INDEX:?set TORCH_INDEX together with LOCK}
echo "driver $DRV, lowest compute capability $MINCC -> $LOCK ($TORCH_INDEX)"

echo "== 1/5 uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found: installing it for this user (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "== 2/5 venv $VENV (python $PYVER) + $LOCK"
[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python "$PYVER"
uv pip sync --python "$VENV/bin/python" "$LOCK" \
  --index-url https://pypi.org/simple --extra-index-url "$TORCH_INDEX" \
  --index-strategy unsafe-best-match
H="$VENV/bin/python"

echo "== 3/5 environment vs lockfile"
"$H" scripts/check_env.py --lock "$LOCK"

echo "== 4/5 golden fixtures (preflight check 5)"
if ls golden_parts/R2.tar.part-* >/dev/null 2>&1; then
  if [ -f golden/R2/manifest.json ]; then
    echo "golden/R2 already unpacked"
  else
    mkdir -p golden
    cat golden_parts/R2.tar.part-* > golden/R2.tar
    (cd golden && sha256sum -c ../golden_parts/R2.tar.sha256)
    tar xf golden/R2.tar -C golden
    rm golden/R2.tar
    echo "golden/R2 unpacked and verified"
  fi
else
  echo "WARNING: no golden_parts/ in this checkout: preflight check 5 will FAIL and the queue will stop."
fi

echo "== 5/5 machine probe"
"$H" - <<'EOF'
from harness.device import probe_machine
p = probe_machine(".")
gb = lambda b: f"{(b or 0) / 2**30:.0f} GiB"
print(f"GPUs: {len(p['gpus'])}")
for g in p["gpus"]:
    print(f"  {g['index']}: {g['name']}  {gb(g['total_bytes'])} total, {gb(g['free_bytes'])} free, cc {g['compute_cap']}")
print(f"RAM {gb(p['ram_available_bytes'])} free of {gb(p['ram_total_bytes'])}; CPU {p['cpu_physical']} physical / "
      f"{p['cpu_logical']} logical cores; disk {gb(p['disk_free_bytes'])} free")
if not p["gpus"]:
    print("WARNING: nvidia-smi found no GPU")
import json, os, torch
from harness.device import gpu_compat
res = [gpu_compat(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
os.makedirs("runs", exist_ok=True)
json.dump({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
           "torch_arch_list": torch.cuda.get_arch_list() if torch.backends.cuda.is_built() else [],
           "nvidia_smi": p["gpus"], "gpus": res}, open("runs/gpu_compat.json", "w"), indent=1)
print("GPU compatibility (runs/gpu_compat.json):")
for r in res:
    print(f"  {r['index']}: {r['name']} cc {r['compute_cap']}: {'OK' if r.get('ok') else 'FAIL ' + r.get('error', '')}"
          + (f"; GEMM fp32 {r['fp32_tflops']} / bf16 {r['bf16_tflops']} TFLOPS" if 'bf16_tflops' in r else "")
          + ("" if r["native_bf16"] else " (no native bf16: emulated, slow; numbers stay bf16 like the pilot)"))
bad = [r for r in res if not r.get("ok")]
if p["gpus"] and not res:
    print("ERROR: torch cannot see the GPUs (torch.cuda.is_available() is False)"); raise SystemExit(1)
if bad:
    print(f"ERROR: {len(bad)} GPU(s) cannot run this torch build; preflight would drop them"); raise SystemExit(1)
if (p["disk_free_bytes"] or 0) < 70 * 2**30:
    print("WARNING: release 2 needs ~35 GiB of SAE downloads plus caches: keep >= 70 GiB free")
EOF
echo
echo "Setup done. Next: export HL_PROFILE=server HF_TOKEN=hf_...; see README.md"
