"""Cross-run reproducibility of steering: E10 (node-C, 2026-11-17/18) vs E19 (node-D, 2026-11-19) on the shared keys
(task, condition steer_x1/x2/x4, estimator, r in {1, 4}, random seeds), same test pairs, same U (bitwise identical, see
steering_dirs.py). Reports per key: flip in each run, per-pair top-1 agreement, max |D_E19 - D_E10|.
Output: steering_out/repro_E10_E19.csv"""
import numpy as np
import pandas as pd
from steering_common import OUT

a = pd.read_parquet(OUT / "pp_E10.parquet")
b = pd.read_parquet(OUT / "pp_E19.parquet")
k = ["task", "pair_id", "condition", "estimator", "r", "seed"]
m = a[k + ["top1_id", "D", "isA", "elig", "f_pilot"]].merge(b[k + ["top1_id", "D", "isA"]], on=k, suffixes=("_10", "_19"))
m = m[m.condition.str.startswith("steer_") | m.condition.isin(["clean_B", "clean_A", "full_A"])]
rows = []
for (task, cond, est), g in m.groupby(["task", "condition", "estimator"]):
    e = g[g.elig & g.f_pilot]
    rows.append(dict(task=task, condition=cond, estimator=est, n_rows=len(g), top1_agree=(g.top1_id_10 == g.top1_id_19).mean(),
                     max_absdD=np.abs(g.D_10 - g.D_19).max(), med_absdD=np.abs(g.D_10 - g.D_19).median(),
                     flip10=e.isA_10.mean(), flip19=e.isA_19.mean()))
R = pd.DataFrame(rows)
R.to_csv(OUT / "repro_E10_E19.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False))
print("overall top1 agreement", (m.top1_id_10 == m.top1_id_19).mean(), "rows", len(m), "max |dD|", np.abs(m.D_10 - m.D_19).max())
