"""Are the steering U of E19 the same subspaces as the patching U of E9 / E17 / E18 (and E10)? Principal cosines between
saved direction matrices (directions/*.npz). Also DAS meta (train interchange acc, loss) from the json files.
Output: steering_out/dirs_identity.csv, steering_out/das_meta.csv"""
import json
import numpy as np
import pandas as pd
from steering_common import run_dir, OUT

STEMS = {"add_chain": "add_chain_L18_L18_16k_l074", "entity_track": "entity_track_L12_L12_16k_l082",
         "people_cities": "people_cities_L12_L12_16k_l082__fit-pc_en"}


def pcos(U, V):
    qu, _ = np.linalg.qr(U.astype(np.float64))
    qv, _ = np.linalg.qr(V.astype(np.float64))
    return np.linalg.svd(qu.T @ qv, compute_uv=False)


rows, meta = [], []
for task, stem in STEMS.items():
    ref = np.load(run_dir("E19") / "directions" / f"{stem}.npz")
    for exp in ["E9", "E10", "E17", "E18"]:
        p = run_dir(exp) / "directions" / f"{stem}.npz"
        if not p.exists():
            continue
        o = np.load(p)
        for est in ref.files:
            if est not in o.files:
                continue
            for r in ([1, 4, 8] if not est.startswith("das") else [ref[est].shape[1]]):
                c = pcos(ref[est][:, :r], o[est][:, :r])
                rows.append(dict(task=task, est=est, r=r, other=exp, min_cos=c.min(), mean_cos=c.mean(),
                                 max_abs_diff=float(np.abs(ref[est][:, :r] - o[est][:, :r]).max())))
    for exp in ["E9", "E10", "E17", "E18", "E19"]:
        p = run_dir(exp) / "directions" / f"{stem}.json"
        if p.exists():
            j = json.load(open(p))
            for k, v in j.items():
                if k.startswith("das"):
                    m = v["meta"]
                    meta.append(dict(task=task, exp=exp, est=k, acc=m.get("train_interchange_acc"),
                                     loss_first=m.get("loss_first_epoch"), loss_last=m.get("loss_last_epoch"),
                                     epochs=m.get("epochs"), digest=v.get("fit_digest")))
d = pd.DataFrame(rows)
d.to_csv(OUT / "dirs_identity.csv", index=False)
m = pd.DataFrame(meta)
m.to_csv(OUT / "das_meta.csv", index=False)
pd.set_option("display.width", 200)
print(d.to_string())
print(m[m.est.isin(["das4", "das8", "das8_e10"])].to_string())
