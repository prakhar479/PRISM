#!/usr/bin/env python
"""tables/E12_fidelity_lang.csv from ingested E12 runs (runs/E12/*/fidelity.csv). DEV only; no model.

    python scripts/make_e12.py [--runs runs] [--out tables]

One row per (set, layer, SAE; E21 adds wider SAEs): the E12 measures with bootstrap CIs over windows (scripts/run_fidelity.py), plus the preregistered M1
comparison against English at the same layer and SAE, kept as plain facts (no verdict wording):
    recovered_mean_below_en_ci   the set's recovered_mean upper bound is below English's lower bound ("non-overlapping CIs")
    fvu_above_en_ci              the set's FVU lower bound is above English's upper bound
and, for the Dakshina pairs (same sentences in two scripts), `script_pair` / `script_partner` so the native and romanized rows can
be read side by side. The pilot parity row (en_pilot100) is listed under the English rows and is not used as the reference.
"""
import argparse
import csv
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from harness.profile import REPO_ROOT, load_profile

EN_REF = "en_pile"
PAIRS = {"hi_deva_dakshina": "hi_latn_dakshina", "hi_latn_dakshina": "hi_deva_dakshina", "ta_taml_dakshina": "ta_latn_dakshina",
         "ta_latn_dakshina": "ta_taml_dakshina", "bn_beng_dakshina": "bn_latn_dakshina", "bn_latn_dakshina": "bn_beng_dakshina"}


def gather(runs_root: Path):
    rows = {}
    for f in sorted(runs_root.glob("E12/*/fidelity.csv"), key=lambda p: p.stat().st_mtime):
        run = f.parent.name
        for r in csv.DictReader(f.open()):
            r["run_hash"] = run
            rows[(r["set"], int(r["layer"]), r.get("sae", ""))] = r   # a later run of the same (set, layer, SAE) replaces an earlier one
    return rows


def build(rows):
    out = []
    for (name, layer, sae), r in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        ref = rows.get((EN_REF, layer, sae))                 # English with the same SAE (layer, width, L0)
        row = dict(r)
        f = lambda k: float(r[k])
        if ref and name != EN_REF:
            row["recovered_mean_below_en_ci"] = f("recovered_mean_hi") < float(ref["recovered_mean_lo"])
            row["fvu_above_en_ci"] = f("fvu_lo") > float(ref["fvu_hi"])
            row["recovered_mean_minus_en"] = round(f("recovered_mean") - float(ref["recovered_mean"]), 5)
        else:
            row["recovered_mean_below_en_ci"] = row["fvu_above_en_ci"] = row["recovered_mean_minus_en"] = ""
        partner = PAIRS.get(name)
        row["script_partner"] = partner or ""
        row["recovered_mean_minus_script_partner"] = (round(f("recovered_mean") - float(rows[(partner, layer, sae)]["recovered_mean"]), 5)
                                                      if partner and (partner, layer, sae) in rows else "")
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "tables"))
    a = ap.parse_args()
    runs_root = Path(a.runs) if a.runs else load_profile("dev").root("runs_root")
    rows = gather(runs_root)
    if not rows:
        print(f"no E12 runs under {runs_root}/E12")
        return 1
    table = build(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in table for k in r))          # E23 rows add the noise-control columns
    with (out / "E12_fidelity_lang.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    print(f"{len(table)} rows -> {out / 'E12_fidelity_lang.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
