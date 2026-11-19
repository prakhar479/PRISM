"""Task 2 addendum: seed-dependent conditions (noise controls, resampled errors) in E2 (seed 0, laptop RTX 5060) vs E2r (seed 1,
gnode084) vs E3 (pilot families subset): per-pair agreement and run-level differences with paired bootstrap CIs. EXPLORATORY."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad/team/scripts")
from stats_common import OUT  # noqa: E402
from stats_engine import boot_W, ci  # noqa: E402
from stats_pp import host, load_cell, run_dirs  # noqa: E402

pd.set_option("display.width", 250)
CELLS = {"RS": ("add_chain", 18, "L18_16k_l074"), "OT": ("entity_track", 12, "L12_16k_l082")}
runs = {"E2": [d for d in run_dirs("E2") if "5ad7ada7f971" in d][0], "E2r": run_dirs("E2r")[0], "E3": run_dirs("E3")[0]}
out = []
for c, (task, L, sae) in CELLS.items():
    T = {k: load_cell(d, task, L, sae, fit=("fresh" if k == "E3" else "pilot"), family="pilot") for k, d in runs.items()}  # E3 evaluates pilot families only under fit "fresh"
    conds = sorted(set.intersection(*[set(t[t.estimator == ""].condition) for t in T.values()]) - {"clean_A", "clean_B", "full_B"})
    for a, b in (("E2", "E2r"), ("E2", "E3"), ("E2r", "E3")):
        for cond in conds:
            fa = T[a][(T[a].condition == cond) & (T[a].estimator == "")].drop_duplicates("pair_id").set_index("pair_id")
            fb = T[b][(T[b].condition == cond) & (T[b].estimator == "")].drop_duplicates("pair_id").set_index("pair_id")
            com = fa.index.intersection(fb.index).sort_values()
            fa, fb = fa.loc[com], fb.loc[com]
            el = ((fa.elig > 0) & (fb.elig > 0)).to_numpy()
            ha, hb = fa.hitA.to_numpy()[el], fb.hitA.to_numpy()[el]
            W = boot_W(int(el.sum()), 2000, 0)
            d = (W @ (ha - hb)) / W.sum(1)
            lo, hi = ci(d[1:])
            na = np.clip(fa.nldr_i.to_numpy(), -1, 2)
            nb = np.clip(fb.nldr_i.to_numpy(), -1, 2)
            out.append({"cell": c, "a": f"{a}@{host(runs[a])}", "b": f"{b}@{host(runs[b])}", "condition": cond,
                        "n_elig": int(el.sum()), "flip_a": ha.mean(), "flip_b": hb.mean(), "diff": d[0], "lo95": lo, "hi95": hi,
                        "per_pair_flip_agree": float(np.mean(ha == hb)), "corr_nldr_i": float(np.corrcoef(na, nb)[0, 1]),
                        "max_abs_dD": float(np.abs(fa.D - fb.D).max())})
R = pd.DataFrame(out)
R.to_csv(f"{OUT}/var_e2_seed_machine.csv", index=False)
print(R.round(3).to_string(index=False))
