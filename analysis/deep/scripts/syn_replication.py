"""Replication of the two headline quantities across every run that contains them (exploratory synthesis).
RS L18: flip shortfall full_A - feat_A_meanPos and NLDR gap of feat_A_errB; OT L12: NLDR gap of feat_A_errB, flip gap.
Uses harness.final.run_table on each run's per_pair (same metric code as the tables)."""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/hw"); sys.path.insert(0, sys.argv[1] + "/hw/scripts")
import pandas as pd
from harness import final as FN
SP = Path(sys.argv[1]); rows = []
for man in sorted(glob.glob(str(SP / "ana_runs/E*/*/manifest.json"))):
    m = json.loads(open(man).read()); d = Path(man).parent
    if m.get("status") != "complete" or not (d / "per_pair.parquet").exists() or m.get("smoke"):
        continue
    pp = pd.read_parquet(d / "per_pair.parquet", columns=None)
    pp = pp[((pp.task == "add_chain") & (pp.layer == 18)) | ((pp.task == "entity_track") & (pp.layer == 12))]
    pp = pp[pp.sae.str.contains("_16k_") & pp.condition.isin(["full_A", "full_B", "clean_B", "feat_A_errB", "feat_A_meanPos"])]
    if pp.empty or "feat_A_errB" not in set(pp.condition):
        continue
    t = FN.run_table(m, pp, ["f_A_correct", "f_pilot"], 2000)
    if t.empty or "condition" not in t: continue
    t = t[t.condition.isin(["feat_A_errB", "feat_A_meanPos"])]
    for _, r in t.iterrows():
        rows.append({"exp": m["exp_id"], "run": m["run_hash"], "commit": m["git"]["commit"][:7], "task": r.task, "family": r.get("family"),
                     "fit": r.get("fit_families"), "filter": r["filter"], "cond": r.condition, "n": r.n_pairs,
                     "flip_gap": r.gap_flip_rate, "flip_gap_lo": r.gap_flip_rate_lo, "flip_gap_hi": r.gap_flip_rate_hi,
                     "nldr_gap": r.gap_nldr, "nldr_gap_lo": r.gap_nldr_lo, "nldr_gap_hi": r.gap_nldr_hi})
out = pd.DataFrame(rows); out.to_csv(SP / "deep/replication.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(out.round(3).to_string(index=False))
