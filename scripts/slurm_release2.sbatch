#!/usr/bin/env bash
# Release 2 (E3 gate, E4, E5, E8) on the 4-GPU node gnode077 under Slurm.
#
#   cd ~/prism-run                       # the run checkout, on the run-YYYYMMDD-N tag
#   echo 'hf_xxx' > ~/.hf_token && chmod 600 ~/.hf_token     # once; or export HF_TOKEN before sbatch
#   sbatch scripts/slurm_release2.sbatch                      # run (resumes whatever is already done)
#   bash   scripts/slurm_release2.sbatch --clean-only         # just free space, no job (run it on gnode077)
#   sbatch --export=ALL,PURGE_HF=1 scripts/slurm_release2.sbatch   # also drop the 49 GB model download at the end
#
# Adjust the #SBATCH lines to the cluster (partition, account, memory, time). --nodelist matters: /ssd_scratch is
# node-local, so the venv, downloads and caches that setup_server.sh put there exist only on gnode077.
#SBATCH --job-name=prism-r2
#SBATCH --nodelist=gnode077
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
# Slurm sends SIGTERM at the time limit and on scancel; the queue is resumable, so just resubmit.
set -uo pipefail

# Slurm runs a copy of this file from its spool; the checkout is where it was submitted from.
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

export HL_PROFILE=server
export CUDA_DEVICE_ORDER=PCI_BUS_ID
WORK=${PRISM_WORK:-/ssd_scratch/${USER:-$(id -un)}/prism}
export PRISM_WORK=$WORK
QUEUE=${QUEUE:-queue/release2-server.yaml}
PURGE_HF=${PURGE_HF:-0}                       # 1: also delete the Hugging Face download when everything succeeded
CACHE_ON_SUCCESS=${CLEAN_CACHE_ON_SUCCESS:-1} # 1: delete activation caches once results are published
H=$REPO/.venv-harness/bin/python

gib() { df -Pk "$1" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}'; }
size() { du -sh "$1" 2>/dev/null | cut -f1; }

# Only ever deletes inside $WORK or the checkout, and only things that are rebuilt on demand.
clean() {   # clean light | clean full
  local mode=${1:-light}
  case "$WORK" in /ssd_scratch/?*/*|/*/prism*) ;; *) echo "clean: refusing odd WORK=$WORK"; return 1;; esac
  echo "== cleanup ($mode): scratch $(gib "$WORK" 2>/dev/null || echo ?) GiB free, home $(gib "$REPO") GiB free"
  # always safe: temp files, package-manager caches, bytecode, half-downloaded model files
  rm -rf "$WORK/tmp" && mkdir -p "$WORK/tmp"
  [ -d "$WORK/uv-cache" ] && { echo "  uv cache $(size "$WORK/uv-cache")"; rm -rf "$WORK/uv-cache" && mkdir -p "$WORK/uv-cache"; }
  rm -rf "$HOME/.cache/pip" "$HOME/.cache/uv" "$HOME/.cache/torch/kernels" "$HOME/.triton" 2>/dev/null
  find "$REPO" -name __pycache__ -type d -not -path "$REPO/.venv-harness/*" -prune -exec rm -rf {} + 2>/dev/null
  find "$WORK/hf" -name '*.incomplete' -delete 2>/dev/null
  find "$WORK/hf" -name '*.lock' -path '*/.locks/*' -delete 2>/dev/null
  local BULK=${PRISM_BULK:-/scratch/${USER:-$(id -un)}/prism}
  if [ "$mode" = full ] && [ -d "$BULK/cache" ]; then echo "  bulk-disk caches $(size "$BULK/cache")"; rm -rf "$BULK/cache" && mkdir -p "$BULK/cache"; fi
  if [ "$mode" = full ]; then
    # results are published: the activation caches are only a speed-up and are rebuilt if a rerun needs them
    [ -d "$WORK/cache" ] && { echo "  activation caches $(size "$WORK/cache")"; rm -rf "$WORK/cache" && mkdir -p "$WORK/cache"; }
    if [ "$PURGE_HF" = 1 ]; then
      for hf in "$WORK/hf" "$BULK/hf"; do
        [ -d "$hf" ] && { echo "  Hugging Face download $(size "$hf") (PURGE_HF=1; the next run downloads it again)"; rm -rf "$hf" && mkdir -p "$hf"; }
      done
    fi
  fi
  echo "== after cleanup: scratch $(gib "$WORK") GiB free, home $(gib "$REPO") GiB free"
}

if [ "${1:-}" = "--clean-only" ]; then clean full; exit $?; fi

echo "== job ${SLURM_JOB_ID:-local} on $(hostname), start $(date -Is)"
echo "   repo $REPO ($(git -C "$REPO" describe --tags --always 2>/dev/null)); work $WORK"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || { echo "no GPUs visible"; exit 1; }

if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.hf_token" ]; then HF_TOKEN=$(<"$HOME/.hf_token"); fi
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN not set and ~/.hf_token missing (Gemma-2 is gated)"; exit 1; }
export HF_TOKEN

STAMP=$(mktemp -p "${TMPDIR:-/tmp}" prism-stamp.XXXXXX)
trap 'rm -f "$STAMP"' EXIT
finish() {   # SIGTERM: stop the queue, free the cheap stuff, keep caches so the resubmit continues
  echo "== signal received at $(date -Is): stopping queue"
  [ -n "${QPID:-}" ] && kill -TERM "$QPID" 2>/dev/null && wait "$QPID" 2>/dev/null
  clean light; exit 143
}
trap finish TERM INT

# fail in seconds, not after the 30-minute preflight, if the checkout is dirty (a stray file counts: every run records it)
DIRTY=$(git status --porcelain 2>/dev/null)
if [ -n "$DIRTY" ]; then
  echo "ERROR: working tree is dirty; the harness refuses to run. Offending files:"; echo "$DIRTY" | head -20
  echo "Remove them, or for log files: echo 'slurm-*.out' >> .git/info/exclude"; exit 1
fi

# idempotent: venv, locked packages, env check, golden fixtures, profiles/local.yaml, gpu_compat.json
bash scripts/setup_server.sh || { echo "setup failed"; exit 1; }

$H scripts/storage_report.py || true
$H scripts/plan_runs.py --queue "$QUEUE" --profile server || true

# --retry-failed: a resubmit picks up units that were running when the last job ended
$H scripts/run_queue.py --queue "$QUEUE" --gpus auto --retry-failed &
QPID=$!
wait "$QPID"; RC=$?
QPID=

PUBLISHED=$(find results_server -name PUBLISHED.json -newer "$STAMP" 2>/dev/null | head -1)
if [ "$RC" -eq 0 ] && [ -n "$PUBLISHED" ] && [ "$CACHE_ON_SUCCESS" = 1 ]; then
  echo "== queue finished, results in $(dirname "$PUBLISHED")"
  clean full
else
  echo "== queue exit $RC (published: ${PUBLISHED:-none}); keeping activation caches so a resubmit continues"
  clean light
fi
echo "== done $(date -Is) exit $RC. Commit the results:"
echo "   git switch -c results/\$(git describe --tags --always) && git add -f results_server/* && git commit -m 'release 2 results' && git push -u origin HEAD"
exit "$RC"
