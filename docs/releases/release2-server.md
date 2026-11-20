# Release 2 on a multi-GPU server (queue/release2-server.yaml)

Same experiments and code paths as release 2 (`releases/release2.md`): E3 (RQ1 gate), E4, E5, E8 fill. What
changes is how the work is placed on the machine; results are statistically, not bitwise, identical to a laptop run
(bf16 batch composition differs), and every number of one run uses one pinned token budget.

## Steps (on the server)

See `README.md` in the run checkout:
1. `bash scripts/setup_server.sh`
2. `export HL_PROFILE=server HF_TOKEN=hf_...`
3. `$H scripts/plan_runs.py --queue queue/release2-server.yaml --profile server`
4. `$H scripts/run_queue.py --queue queue/release2-server.yaml --gpus auto` (in tmux)

## Pipeline

| phase | jobs | GPUs |
|---|---|---|
| prefetch | model + 23 SAEs (~35 GiB new) | none |
| preflight | checks 1-8 on the first GPU; check 9 = every GPU in its own process (model, hooks, largest token budget that fits, throughput) -> `gpus.json` | all |
| stage 1 | E3 sets and the pilot sets (E8; covers E4/E5) filled once each | 2 in parallel |
| stage 2 | disjoint SAE groups per GPU (E3, E8, then E4/E5) | all |
| stage 3 | one claiming worker per GPU; E3 first (the gate), then E4, E5, E8 | all |
| finalize | per experiment, the only writer of `manifest.json` / `per_pair.parquet` | none |
| bundle | `bundles/release2-server_<stamp>.tar.gz`, runs even if an experiment failed | none |
| publish | `results_server/release2-server/`: the same finished files, readable, big files split; commit it with `git add -f` | none |

## Send back

`bundles/release2-server_<stamp>.tar.gz` + `.sha256`, or `results_server/release2-server/`, on a `results/<tag>` branch (README.md, last section), plus a
short note: GPU model and count (from `machine.json`), start and end time, anything in `queue_status.md` that is not
`done`.
