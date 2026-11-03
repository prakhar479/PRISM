#!/usr/bin/env bash
# One-time setup of a run checkout (the `run` branch) on a GPU server. Safe to re-run.
#   bash scripts/setup_server.sh            # venv + locked packages + env check + golden fixtures + machine probe
# Needs: python 3.12, an NVIDIA driver new enough for CUDA 13 (torch 2.14.1+cu130), ~70 GB free disk, HF_TOKEN later.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv-harness}
PYVER=${PYVER:-3.12}

echo "== 1/5 uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found: installing it for this user (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "== 2/5 venv $VENV (python $PYVER) + requirements/harness.lock (CUDA 13 torch wheels)"
[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python "$PYVER"
uv pip install --python "$VENV/bin/python" -r requirements/harness.lock \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
H="$VENV/bin/python"

echo "== 3/5 environment vs lockfile"
"$H" scripts/check_env.py

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
if (p["disk_free_bytes"] or 0) < 70 * 2**30:
    print("WARNING: release 2 needs ~35 GiB of SAE downloads plus caches: keep >= 70 GiB free")
EOF
echo
echo "Setup done. Next: export HL_PROFILE=server HF_TOKEN=hf_...; see README.md"
