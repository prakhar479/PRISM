#!/usr/bin/env bash
# Overnight PRISM run on one GPU, in priority tiers. Each tier is its own process and writes metrics, figures and
# SUMMARY.md before the next starts, so a crash or the deadline still leaves usable results.
#
#   HF_TOKEN=hf_xxx bash scripts/run_server_gpu.sh                 # background (nohup), 6.5 h budget
#   bash scripts/run_server_gpu.sh --foreground --hours 4
#   bash scripts/run_server_gpu.sh --resume results/overnight_XXXX # continue a crashed run (finished units skipped)
#   bash scripts/run_server_gpu.sh --mock --foreground             # CPU dry run of the whole schedule (~2 min)
#   bash scripts/run_server_gpu.sh --tiers tier1_core,tier2_widths
#
# Stages: env checks -> GPU smoke test -> preflight (prompt format / few-shot per task) -> prefetch SAEs
#         -> tier1_core -> tier2_layers -> tier2_widths -> tier3_expansions -> INDEX.md
set -euo pipefail
cd "$(dirname "$0")/.."

FOREGROUND=0; SKIP_SMOKE=0; MOCK=0; HOURS=6.5; OUT=""; RESUME=0
TIERS="tier1_core,tier2_layers,tier2_widths,tier3_expansions"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) FOREGROUND=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --mock) MOCK=1; SKIP_SMOKE=1 ;;
    --hours) HOURS="$2"; shift ;;
    --tiers) TIERS="$2"; shift ;;
    --resume) OUT="$2"; RESUME=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done

PY="${PYTHON:-python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-results/overnight_${STAMP}}"
mkdir -p "$OUT" logs
LOG="logs/prism_${STAMP}.log"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ $MOCK -eq 0 ]]; then
  echo "== environment checks =="
  command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found (no NVIDIA driver?)" >&2; exit 1; }
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
  : "${HF_TOKEN:?ERROR: HF_TOKEN is not set (gemma-2-2b is a gated model)}"
  export HF_TOKEN
  df -h . | tail -1 | awk '{print "disk free here: "$4}'
  $PY - <<'PYEOF'
import os, sys, torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
free, total = torch.cuda.mem_get_info()
print(f"torch {torch.__version__}, CUDA ok ({torch.cuda.get_device_name()}), free VRAM {free/2**30:.1f}/{total/2**30:.1f} GiB")
for m in ("transformer_lens", "sae_lens", "transformers"):
    mod = __import__(m); print(f"  {m} {getattr(mod, '__version__', '?')}")
from huggingface_hub import HfApi
try:
    HfApi().model_info("google/gemma-2-2b", token=os.environ["HF_TOKEN"])
    print("HF access to google/gemma-2-2b: ok")
except Exception as e:
    sys.exit(f"ERROR: cannot access google/gemma-2-2b with HF_TOKEN ({type(e).__name__}: {e})")
PYEOF
fi

run_all() {
  set -uo pipefail
  local T0 DEADLINE MOCKF EXTRA status
  T0=$(date +%s)
  DEADLINE=$($PY -c "print(int($T0 + $HOURS * 3600))")
  MOCKF=""; EXTRA=()
  if [[ $MOCK -eq 1 ]]; then MOCKF="--mock"; EXTRA=(--n-train 64 --n-test 32 --set min_pairs=8 --set bootstrap=200); fi
  echo "== PRISM overnight run -> $OUT (deadline in ${HOURS} h) =="
  : > "$OUT/STATUS"

  if [[ $SKIP_SMOKE -eq 0 ]]; then
    echo "== [0/4] GPU smoke test =="
    if ! $PY scripts/run_smoke_test.py --out "$OUT/smoke"; then
      echo "smoke: FAILED (continuing: preflight decides whether tasks are usable)" | tee -a "$OUT/STATUS"
    else
      echo "smoke: ok" >> "$OUT/STATUS"
    fi
  fi

  echo "== [0/4] preflight: prompt format and few-shot per task =="
  if [[ $RESUME -eq 1 && -f "$OUT/preflight/preflight.json" ]]; then
    echo "preflight: reused" >> "$OUT/STATUS"
  elif ! $PY scripts/preflight.py $MOCKF --out "$OUT/preflight"; then
    echo "preflight: NO usable task -> abort" | tee -a "$OUT/STATUS"; return 1
  else
    echo "preflight: ok" >> "$OUT/STATUS"
  fi

  IFS=',' read -ra TL <<< "$TIERS"
  if [[ $MOCK -eq 0 ]]; then
    echo "== prefetch model + SAEs =="
    local cfgs=(); for t in "${TL[@]}"; do cfgs+=("configs/$t.yaml"); done
    $PY scripts/prefetch.py "${cfgs[@]}" --out "$OUT/prefetch" || echo "prefetch: errors (see $OUT/prefetch)" | tee -a "$OUT/STATUS"
  fi

  for t in "${TL[@]}"; do
    if (( $(date +%s) > DEADLINE )); then echo "$t: skipped (deadline)" | tee -a "$OUT/STATUS"; continue; fi
    echo "== tier $t ($(( ($(date +%s) - T0) / 60 )) min elapsed) =="
    $PY scripts/run_full_suite.py --config "configs/$t.yaml" $MOCKF "${EXTRA[@]}" \
        --overrides "$OUT/preflight/preflight.json" --out "$OUT/$t" --resume --deadline "$DEADLINE"
    status=$?
    echo "$t: exit $status after $(( ($(date +%s) - T0) / 60 )) min" | tee -a "$OUT/STATUS"
  done

  {
    echo "# PRISM overnight run $(basename "$OUT")"; echo; echo '```'; cat "$OUT/STATUS"; echo '```'; echo
    for t in "${TL[@]}"; do [[ -f "$OUT/$t/SUMMARY.md" ]] && echo "- [$t]($t/SUMMARY.md)"; done
  } > "$OUT/INDEX.md"
  echo "== DONE: $OUT/INDEX.md =="
}

if [[ $FOREGROUND -eq 1 ]]; then
  run_all 2>&1 | tee "$LOG"
  exit "${PIPESTATUS[0]}"
else
  export -f run_all; export PY OUT HOURS MOCK SKIP_SMOKE TIERS RESUME
  nohup bash -c 'run_all' > "$LOG" 2>&1 &
  echo "started in background (pid $!), results: $OUT"
  echo "follow with: tail -f $LOG     (status: cat $OUT/STATUS)"
fi
