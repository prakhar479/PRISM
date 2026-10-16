#!/usr/bin/env bash
# PRISM: the one script you need.  ./run.sh <command>     (full guide: docs/RUNBOOK.md)
#
#   setup      create .venv and install requirements
#   test       CPU unit tests + mock smoke test (no GPU, ~15 s)
#   dryrun     whole overnight schedule on CPU with the mock model (~90 s)
#   start      launch the overnight GPU run in the background   [--hours 6.5] [--tiers a,b]
#   status     progress of the latest run
#   resume     continue the latest (crashed / stopped) run, finished work is skipped
#   decide     judge H1-H3 for the latest run and print what to do next (writes DECISIONS.md)
#   results    print the latest run's INDEX, SUMMARY and DECISIONS paths
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || { [[ -x ../.venv/bin/python ]] && echo ../.venv/bin/python; } || echo python )}"
export PYTHON="$PY"
latest() { ls -d results/overnight_* 2>/dev/null | grep -v mock | tail -1; }
cmd="${1:-help}"; shift || true

case "$cmd" in
  setup)
    [[ -d .venv ]] || python3 -m venv .venv
    .venv/bin/pip install -U pip -q && .venv/bin/pip install -r requirements.txt
    echo "ok. Next: export HF_TOKEN=hf_... && ./run.sh test" ;;
  test)
    $PY -m pytest tests/test_pipeline_cpu.py -q
    $PY scripts/run_smoke_test.py --mock | tail -3 ;;
  dryrun)
    bash scripts/run_server_gpu.sh --mock --foreground "$@"
    d=$(ls -d results/overnight_* | tail -1); mv "$d" "${d}_mock"
    $PY scripts/decide.py "${d}_mock" >/dev/null; echo "dry run ok -> ${d}_mock (mock numbers are NOT real results)" ;;
  start)
    : "${HF_TOKEN:?set HF_TOKEN first: export HF_TOKEN=hf_...}"
    bash scripts/run_server_gpu.sh "$@" ;;
  status)
    d=$(latest); [[ -n "$d" ]] || { echo "no run yet"; exit 0; }
    echo "run: $d"; cat "$d/STATUS" 2>/dev/null || true
    log=$(ls -t logs/prism_*.log 2>/dev/null | head -1); [[ -n "$log" ]] && { echo "--- last log lines ($log)"; tail -5 "$log"; }
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true ;;
  resume)
    d=$(latest); [[ -n "$d" ]] || { echo "no run to resume"; exit 1; }
    bash scripts/run_server_gpu.sh --resume "$d" --skip-smoke "$@" ;;
  decide)
    $PY scripts/decide.py "${1:-$(latest)}" ;;
  results)
    d="${1:-$(latest)}"; echo "run: $d"
    for f in "$d/INDEX.md" "$d/DECISIONS.md" "$d"/*/SUMMARY.md "$d"/*/figures; do [[ -e "$f" ]] && echo "  $f"; done ;;
  *) sed -n '2,12p' "$0" ;;
esac
