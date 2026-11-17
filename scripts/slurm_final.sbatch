#!/usr/bin/env bash
# PRISM: every remaining experiment in ONE queue (queue/complete-server.yaml: release 3 E12/E13 + extensions, final pass E9, E10,
# E11, E14, exploratory E9b, E10b, E2r, E16), in one Slurm job. Resumable: resubmitting skips finished units.
#
#   sbatch scripts/slurm_final.sbatch                                   # the complete queue
#   sbatch --export=ALL,QUEUES=queue/extra-1m-server.yaml scripts/slurm_final.sbatch  # optional E4_1m (needs ~40 GiB for downloads)
#   bash   scripts/slurm_final.sbatch --check                           # pre-flight checks only, no GPU work (login node is fine)
#   bash   scripts/slurm_final.sbatch --clean-full                      # AFTER results are pushed: delete the activation caches
#
# Read docs/FINAL_PASS_RUNBOOK.md first (before / during / after). The node matters: the release-2 activation caches are on
# gnode074's node-local /ssd_scratch, so the final pass reuses them there.
#SBATCH --job-name=prism-final
#SBATCH --nodelist=gnode084
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
set -uo pipefail

REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }
export HL_PROFILE=server CUDA_DEVICE_ORDER=PCI_BUS_ID
WORK=${PRISM_WORK:-/ssd_scratch/${USER:-$(id -un)}/prism}
export PRISM_WORK=$WORK
QUEUES=${QUEUES:-"queue/complete-server.yaml"}
MIN_FREE_GIB=${MIN_FREE_GIB:-25}            # free GiB over all cache tiers required before GPU work starts
H=$REPO/.venv-harness/bin/python
MODE=${1:-run}

say() { echo "== $*"; }
gib() { df -Pk "$1" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}'; }

clean_light() {   # temp files and package caches only; activation caches and downloads are kept
  case "$WORK" in /ssd_scratch/?*/*|/*/prism*) ;; *) echo "clean: refusing odd WORK=$WORK"; return 1;; esac
  rm -rf "$WORK/tmp" && mkdir -p "$WORK/tmp"
  rm -rf "$WORK/uv-cache" "$HOME/.cache/pip" "$HOME/.cache/uv" "$HOME/.triton" 2>/dev/null; mkdir -p "$WORK/uv-cache"
  find "$REPO" -name __pycache__ -type d -not -path "$REPO/.venv-harness/*" -prune -exec rm -rf {} + 2>/dev/null
  find "$WORK/hf" -name '*.incomplete' -delete 2>/dev/null
  find "$WORK/cache" ${PRISM_BULK:+"$PRISM_BULK/cache"} -name '*.tmp.*' -type f -delete 2>/dev/null   # half-written files of killed writers (no writer runs now)
  say "light cleanup done: scratch $(gib "$WORK") GiB free, home $(gib "$REPO") GiB free"
}

checks() {   # everything that would otherwise fail hours in; exits non-zero on the first hard problem
  local ok=1
  say "checks on $(hostname) at $(date -Is); repo $REPO"
  local ex; ex="$(git rev-parse --git-common-dir)/info/exclude"; mkdir -p "$(dirname "$ex")"
  grep -qx 'slurm-\*.out' "$ex" 2>/dev/null || echo 'slurm-*.out' >> "$ex"          # job logs never dirty the tree
  local tag; tag=$(git describe --tags --exact-match 2>/dev/null || true)
  if [ -n "$tag" ]; then say "checkout: $tag ($(git rev-parse --short HEAD))"; else echo "WARNING: HEAD is not a run-* tag ($(git describe --tags --always)); results will not ingest without --allow-untagged"; fi
  local dirty; dirty=$(git status --porcelain)
  if [ -n "$dirty" ]; then echo "ERROR: working tree is dirty (the harness refuses to run):"; echo "$dirty" | head -20; ok=0; fi
  if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.hf_token" ]; then HF_TOKEN=$(<"$HOME/.hf_token"); fi
  if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; say "HF token: found"; else echo "ERROR: HF_TOKEN not set and ~/.hf_token missing (Gemma-2 is gated)"; ok=0; fi
  for q in $QUEUES; do [ -f "$q" ] || { echo "ERROR: queue $q not in this checkout"; ok=0; }; done
  [ -f data/lang/signoff.json ] && say "template sign-offs: present" || echo "WARNING: data/lang/signoff.json missing: E13/E14 rows will say validated=false"
  if [ -d golden/R2 ] || ls golden_parts/R2.tar.part-* >/dev/null 2>&1; then say "golden fixtures: present"
  else echo "ERROR: no golden/R2 and no golden_parts/ (preflight check 5 would fail every queue)"; ok=0; fi
  say "disk: scratch $WORK $(gib "$WORK") GiB free, home $(gib "$REPO") GiB free"
  [ "$(gib "$WORK" 2>/dev/null || echo 0)" -ge 15 ] || echo "WARNING: < 15 GiB free on the scratch disk; the queue pauses below its floor (see storage_report.py)"
  [ $ok = 1 ]
}

if [ "$MODE" = "--clean-full" ]; then
  say "full cleanup: activation caches under $WORK/cache (the Hugging Face download is kept)"
  case "$WORK" in /ssd_scratch/?*/*|/*/prism*) rm -rf "$WORK/cache" && mkdir -p "$WORK/cache";; *) echo "refusing odd WORK=$WORK"; exit 1;; esac
  clean_light; exit 0
fi
checks || exit 1
if [ "$MODE" = "--check" ]; then say "checks passed (no GPU work started)"; exit 0; fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || { echo "ERROR: no GPUs visible"; exit 1; }
QPID=
on_signal() { say "signal at $(date -Is): stopping the queue (resubmit to continue)"; [ -n "$QPID" ] && kill -TERM "$QPID" 2>/dev/null && wait "$QPID" 2>/dev/null; clean_light; exit 143; }
trap on_signal TERM INT

bash scripts/setup_server.sh || { echo "ERROR: setup failed"; exit 1; }      # idempotent: venv, lock, env check, golden, probe
$H scripts/storage_report.py || true
# fail in seconds, not after hours paused: the new caches need ~16 GiB (E14 ~9, E11 ~5, rest small) beyond what is cached
FREE=$($H -c "from harness.profile import load_profile; from harness.cache import free_gib; p=load_profile('server'); \
roots=[p.root('cache_root'), *p.spill_roots()]; import os; seen={}; [seen.setdefault(os.stat(r if r.exists() else r.parent).st_dev, free_gib(r)) for r in roots]; print(int(sum(seen.values())))" 2>/dev/null || echo 0)
say "free over the cache tiers: ${FREE} GiB (need ${MIN_FREE_GIB})"
if [ "${FREE:-0}" -lt "$MIN_FREE_GIB" ]; then
  echo "ERROR: only ${FREE} GiB free for caches. Free space on $WORK, or add a bulk disk: export PRISM_BULK=/scratch/\$USER/prism"
  echo "       (and PRISM_MIGRATE_HF=1 to move the model download there) before sbatch; MIN_FREE_GIB=<n> overrides this check."
  exit 1
fi

declare -A RC
for q in $QUEUES; do
  tag=$($H -c "import yaml; print(yaml.safe_load(open('$q'))['tag'])")
  rm -f "runs/_queue/$tag/bundle.done" "runs/_queue/$tag/publish.done"     # always re-bundle / re-publish what exists now
  say "queue $q: plan"
  $H scripts/plan_runs.py --queue "$q" --profile server 2>&1 | grep -v "stage 2:" || true
  say "queue $q: start $(date -Is)"
  $H scripts/run_queue.py --queue "$q" --gpus auto --retry-failed &
  QPID=$!; wait "$QPID"; RC[$q]=$?; QPID=
  say "queue $q: exit ${RC[$q]} at $(date -Is)"          # a failed queue does not stop the next one; each is resumable
done

say "summary"
for q in $QUEUES; do
  tag=$($H -c "import yaml,sys; print(yaml.safe_load(open('$q'))['tag'])")
  pub="results_server/$tag/PUBLISHED.json"
  echo "-- $q (tag $tag): exit ${RC[$q]}; status: runs/_queue/$tag/queue_status.md"
  if [ -f "$pub" ]; then
    $H -c "import json; d=json.load(open('$pub')); [print(f\"   {r['exp_id']:5s} {r['run_hash']}  {r['status']}{'  SMOKE' if r.get('smoke') else ''}\") for r in d['runs']]"
  else
    echo "   not published (see the queue status)"
  fi
done
clean_light
cat <<EOF
== next (see docs/FINAL_PASS_RUNBOOK.md, 'After'):
   git switch -c results/final-\$(git describe --tags --always)
   git add -f results_server && git commit -m "final pass results (\$(hostname))"
   git push -u origin HEAD
   # if any *_fin job failed: also  git add -f runs/E9 runs/E9b runs/E10 runs/E10b runs/E11 runs/E14 runs/E16 runs/E2r runs/E12 runs/E13 runs/_queue
EOF
for q in $QUEUES; do [ "${RC[$q]}" = 0 ] || exit 1; done
exit 0
