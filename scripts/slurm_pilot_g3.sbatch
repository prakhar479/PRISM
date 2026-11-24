#!/usr/bin/env bash
# PRISM pilot, second set (branch pilot-g3, docs/PILOT_G3_PLAN.md) in ONE Slurm job: queue/pilot-g3-server.yaml.
#   P12  dictionary families on Gemma 3 1B (residual SAEs, transcoders, CLTs, MLP / attention SAEs, crosscoders; Gemma Scope 2)
#   P13  refusal state in Gemma 3 1B-IT (features vs error; difference-of-means direction vs DAS)
# Runs next to the breadth pilot (another account / node); exploratory. Resumable: resubmitting skips finished units.
#
#   bash   scripts/slurm_pilot_g3.sbatch --check                           # pre-flight checks only (login node is fine)
#   sbatch --time=04:00:00 scripts/slurm_pilot_g3.sbatch --smoke           # every path on the real models, tiny sizes (~1-2 h)
#   sbatch scripts/slurm_pilot_g3.sbatch                                   # the whole set (4 GPUs, 96 h)
#   sbatch --nodelist=<node> scripts/slurm_pilot_g3.sbatch                  # pin a node (none pinned: any node with 4 GPUs)
#   sbatch --export=ALL,ONLY=P13 scripts/slurm_pilot_g3.sbatch             # a subset (pilot ids or job ids + dependencies)
#   bash   scripts/slurm_pilot_g3.sbatch --record                          # after the job: results branch + commit (no push)
#   bash   scripts/slurm_pilot_g3.sbatch --clean-full                      # AFTER results are pushed: delete the activation caches
#
# Read docs/PILOT_G3_PLAN.md ("Run") first. Needs the Gemma 3 licences (google/gemma-3-1b-pt AND google/gemma-3-1b-it) on the
# token's account. Downloads ~200 GB of Gemma Scope 2 weights (composites read every layer's dictionary): keep HF on /scratch.
#SBATCH --job-name=prism-g3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
set -uo pipefail

# the repo is the script's own checkout when run with `bash` (also inside an interactive Slurm session, where
# SLURM_SUBMIT_DIR points elsewhere); under sbatch the script runs from Slurm's spool copy, so the submit directory is used
SELF_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)
if [ -n "$SELF_REPO" ] && [ -f "$SELF_REPO/scripts/slurm_pilot_g3.sbatch" ]; then REPO=$SELF_REPO; else REPO=${SLURM_SUBMIT_DIR:-$PWD}; fi
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }
export HL_PROFILE=server CUDA_DEVICE_ORDER=PCI_BUS_ID
WORK=${PRISM_WORK:-/ssd_scratch/${USER:-$(id -un)}/prism}
export PRISM_WORK=$WORK
QUEUES=${QUEUES:-"queue/pilot-g3-server.yaml"}
MIN_FREE_GIB=${MIN_FREE_GIB:-300}           # free GiB over all cache tiers: ~200 GB Gemma Scope 2 downloads (CLTs, 262k transcoders,
                                            # 1m crosscoders) + ~40 GB stage-1 sublayer spans and stage-2 caches; see docs/PILOT_G3_PLAN.md
ONLY=${ONLY:-}
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
  for f in configs/dict_registry.yaml configs/pilot/P12.yaml configs/pilot/P13.yaml data/prompt_sets/add_chain_p8_g3.jsonl.gz \
           data/prompt_sets/people_cities_p1_g3.jsonl.gz data/prompt_sets/refusal_p13_g3it.jsonl.gz; do
    [ -f "$f" ] || { echo "ERROR: $f missing (pilot-g3 inputs are committed; wrong checkout?)"; ok=0; }
  done
  local ans; ans=$(python3 -c "import json; print(json.load(open('data/prompt_sets/refusal_p13_g3it.meta.json'))['settings']['answers'])" 2>/dev/null)
  if [ "$ans" = "model" ]; then say "refusal set: answers are the IT model's own first tokens"
  elif [[ ",${ONLY}," == *",P12,"* ]] && [[ ",${ONLY}," != *",P13"* ]]; then say "refusal set is a placeholder ($ans); P13 not selected (ONLY=$ONLY)"
  else echo "ERROR: data/prompt_sets/refusal_p13_g3it is a placeholder (answers: ${ans:-?}); rebuild it with the IT model"
       echo "       (python scripts/build_refusal_set.py, needs the google/gemma-3-1b-it licence) or run only P12: --export=ALL,ONLY=P12"; ok=0; fi
  if [ -x "$H" ]; then
    "$H" scripts/make_pilot_configs.py --check >/dev/null 2>&1 && say "pilot configs: match scripts/make_pilot_configs.py" \
      || echo "WARNING: configs/pilot/*.yaml differ from scripts/make_pilot_configs.py (or the venv is not set up yet)"
  fi
  if [ -d golden/R2 ] || ls golden_parts/R2.tar.part-* >/dev/null 2>&1; then say "golden fixtures: present"
  else echo "ERROR: no golden/R2 and no golden_parts/ (preflight check 5 would fail every queue)"; ok=0; fi
  say "disk: scratch $WORK $(gib "$WORK") GiB free, home $(gib "$REPO") GiB free"
  [ "$(gib "$WORK" 2>/dev/null || echo 0)" -ge 15 ] || echo "WARNING: < 15 GiB free on the scratch disk; the queue pauses below its floor (see storage_report.py)"
  [ $ok = 1 ]
}

if [ "$MODE" = "--record" ]; then          # after the job (login node): commit the published results on a results branch
  for q in $QUEUES; do
    tag=$(python3 -c "import yaml; print(yaml.safe_load(open('$q'))['tag'])" 2>/dev/null || echo pilot-g3-server)
    [ -f "results_server/$tag/PUBLISHED.json" ] || { echo "ERROR: results_server/$tag/PUBLISHED.json missing (did publish run?)"; exit 1; }
  done
  br="results/pilot-g3-$(git describe --tags --always)"
  git switch -c "$br" || exit 1
  git add -f results_server && git commit -m "pilot-g3 results ($(hostname), $(date -Is))" || exit 1
  say "committed on $br; push with: git push -u origin $br   (then tell DEV the branch name)"
  exit 0
fi
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
# fail in seconds, not after hours paused: the pilot's downloads and stage-2 caches need ~110 GiB (MIN_FREE_GIB)
FREE=$($H -c "from harness.profile import load_profile; from harness.cache import free_gib; p=load_profile('server'); \
roots=[p.root('cache_root'), *p.spill_roots()]; import os; seen={}; [seen.setdefault(os.stat(r if r.exists() else r.parent).st_dev, free_gib(r)) for r in roots]; print(int(sum(seen.values())))" 2>/dev/null || echo 0)
say "free over the cache tiers: ${FREE} GiB (need ${MIN_FREE_GIB})"
if [ "${FREE:-0}" -lt "$MIN_FREE_GIB" ]; then
  echo "ERROR: only ${FREE} GiB free for caches. Free space on $WORK, or add a bulk disk: export PRISM_BULK=/scratch/\$USER/prism"
  echo "       (and PRISM_MIGRATE_HF=1 to move the model download there) before sbatch; MIN_FREE_GIB=<n> overrides this check."
  exit 1
fi

if [ -n "$ONLY" ]; then     # pilot ids (P0,P11) or job ids -> those jobs plus everything they depend on (transitively)
  ONLY=$($H - "$ONLY" ${QUEUES%% *} <<'PY'
import sys, yaml
want, q = set(sys.argv[1].split(",")), yaml.safe_load(open(sys.argv[2]))
jobs = {j["id"]: j for j in q["jobs"]}
sel = {i for i in jobs if i in want or i.split("_")[0] in want or (i.startswith("PF_") and i[3:] in want)}
todo = list(sel)
while todo:
    for d in jobs[todo.pop()].get("depends_on", []):
        if d not in sel and d not in ("bundle", "publish"):
            sel.add(d); todo.append(d)
sel |= {"bundle", "publish"}
print(",".join(i for i in jobs if i in sel))
PY
)
  say "ONLY expands to: $ONLY"
fi

if [ "$MODE" = "--smoke" ]; then   # every pilot path on the REAL model, tiny sizes, isolated caches; ~1-2 h on 4 GPUs
  SMK="$WORK/smoke"; rm -rf "$SMK"; mkdir -p "$SMK/logs"
  export HL_CACHE_ROOT="$SMK/cache" HL_RUNS_ROOT="$SMK/runs"     # trained / cached smoke objects never reach the real run
  say "smoke: prefetch (real downloads, shared with the full run)"
  $H scripts/run_prefetch.py --queue queue/pilot-g3-server.yaml || { echo "ERROR: prefetch failed"; exit 1; }
  lane() {   # lane <gpu> <cmd...>: run commands one after another on one GPU, each logged; a failure is recorded, not fatal
    local g=$1; shift
    for c in "$@"; do
      local log="$SMK/logs/$(echo "$c" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-90).log"
      if CUDA_VISIBLE_DEVICES=$g bash -c "$c" > "$log" 2>&1; then echo "SMOKE OK   gpu$g  $c"
      else echo "SMOKE FAIL gpu$g  $c  (log: $log)"; echo "$c" > "$log.FAILED"; fi
    done
  }
  R="$H -m harness.run --profile server --smoke --config"
  # P12 is split by cell group so the lanes share the work (each lane runs its own config subset: --cells)
  lane 0 "$R configs/pilot/P13.yaml" &
  lane 1 "$R configs/pilot/P12.yaml --only-sae gs2_1b_L13 --only-sae gs2_1b_L17" &
  lane 2 "$R configs/pilot/P12.yaml --only-sae gs2_1bc_L13_tc --only-sae gs2_1bc_L13_mlpsae --only-sae gs2_1bc_L17" &
  lane 3 "$R configs/pilot/P12.yaml --only-sae gs2_1bc_L13_clt --only-sae gs2_1bc_L13_xc" &
  wait
  n_fail=$(ls "$SMK"/logs/*.FAILED 2>/dev/null | wc -l)
  say "smoke finished: $(ls "$SMK"/logs/*.log | wc -l) commands, $n_fail failed (logs kept in $SMK/logs)"
  rm -rf "$SMK/cache"                                             # caches and smoke runs are throwaway; logs stay
  [ "$n_fail" = 0 ] && say "smoke passed: launch the full job with  sbatch scripts/slurm_pilot_g3.sbatch" && exit 0
  exit 1
fi

declare -A RC
for q in $QUEUES; do
  tag=$($H -c "import yaml; print(yaml.safe_load(open('$q'))['tag'])")
  rm -f "runs/_queue/$tag/bundle.done" "runs/_queue/$tag/publish.done"     # always re-bundle / re-publish what exists now
  say "queue $q: plan"
  $H scripts/plan_runs.py --queue "$q" --profile server 2>&1 | grep -v "stage 2:" || true
  say "queue $q: start $(date -Is)"
  $H scripts/run_queue.py --queue "$q" --gpus auto --retry-failed ${ONLY:+--only "$ONLY"} &
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
== next (see docs/PILOT_G3_PLAN.md, 'Run'):
   bash scripts/slurm_pilot_g3.sbatch --record          # results branch results/pilot-g3-<tag> with results_server/ committed
   git push -u origin HEAD
   # if any *_fin job failed: also  git add -f runs/P* runs/_queue  (before pushing)
EOF
for q in $QUEUES; do [ "${RC[$q]}" = 0 ] || exit 1; done
exit 0
