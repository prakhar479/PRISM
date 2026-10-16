"""Plain-text SUMMARY.md of a results directory: the numbers to look at first in the morning."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils import load_json


def _fmt(x) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"


def _ci(r: dict) -> str:
    if "nldr_lo" in r:
        return f"{r['nldr']:.2f} [{r['nldr_lo']:.2f},{r['nldr_hi']:.2f}]"
    return _fmt(r.get("nldr"))


def write_summary(results_dir: Path) -> Path:
    d = Path(results_dir)
    m = d / "metrics"
    L: List[str] = [f"# PRISM summary: `{d.name}`", ""]
    if (d / "run_info.json").exists():
        info = load_json(d / "run_info.json")
        L += [f"device `{info.get('device')}` dtype `{info.get('dtype')}` mock={info.get('mock')} "
              f"timings {info.get('timings')} total {info.get('total_seconds')}s", "", "## Tasks", "",
              "| task | n_train | n_test | skipped | n_shots | format | filter | dL clean | dL corrupt | clean top1 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for t, ti in (info.get("task_info") or {}).items():
            L.append(f"| {t} | {ti.get('n_train')} | {ti.get('n_test')} | {ti.get('skipped')} | {ti.get('n_shots')} | "
                     f"{ti.get('answer_format')} | {ti.get('filter_mode')} | {_fmt(ti.get('dl_clean'))} | "
                     f"{_fmt(ti.get('dl_corrupt'))} | {_fmt(ti.get('clean_top1'))} |")
        L.append("")
    if (m / "exp1_results.json").exists():
        rows = load_json(m / "exp1_results.json")
        ivs = ["clean_full", "sae_only", "error_only", "sae_splice", "noise_control",
               "error_resample_same_state", "error_resample_diff_state", "mean_error"]
        ivs = [i for i in ivs if any(r["intervention"] == i for r in rows)]
        L += ["## Exp1: causal split (NLDR, 95% bootstrap CI)", "",
              "| task | layer | width | pos | rel err | " + " | ".join(ivs) + " |", "|" + "---|" * (5 + len(ivs))]
        keys = sorted({(r["task"], r["layer"], r["width"], r.get("positions", "last")) for r in rows})
        for k in keys:
            g = {r["intervention"]: r for r in rows if (r["task"], r["layer"], r["width"], r.get("positions", "last")) == k}
            any_r = next(iter(g.values()))
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {_fmt(any_r.get('rel_recon_err'))} | "
                     + " | ".join(_ci(g[i]) if i in g else "-" for i in ivs) + " |")
        L.append("")
    if (m / "exp2_results.json").exists():
        pr = load_json(m / "exp2_results.json")["probes"]
        L += ["## Exp2: test probe accuracy (selectivity) / R^2 (circular for mod_sum)", "",
              "| task | layer | width | x | x_hat | eps | latents | chance |", "|---|---|---|---|---|---|---|---|"]
        for k in sorted({(r["task"], r["layer"], r["width"]) for r in pr}):
            g = {r["source"]: r for r in pr if (r["task"], r["layer"], r["width"]) == k}

            def cell(src):
                r = g.get(src)
                if not r:
                    return "-"
                r2 = r.get("r2_circular", r.get("r2_ordinal"))
                return f"{r['acc']:.2f} ({_fmt(r.get('selectivity'))}) / {_fmt(r2)}"

            ch = next(iter(g.values()))["chance_acc"]
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cell(s) for s in ("x", "x_hat", "eps", "latents"))
                     + f" | {ch:.2f} |")
        L.append("")
    if (m / "exp3_results.json").exists():
        rows = load_json(m / "exp3_results.json")["rows"]
        L += ["## Exp3: hybrid patch NLDR at each rank (method rows; reference = full error_only)", ""]
        ranks = sorted({r["rank"] for r in rows if r["rank"] >= 0})
        L += ["| task | layer | width | pos | method | " + " | ".join(f"r={r}" for r in ranks) + " | ref |",
              "|" + "---|" * (6 + len(ranks))]
        for k in sorted({(r["task"], r["layer"], r["width"], r.get("positions", "last")) for r in rows}):
            sel = [r for r in rows if (r["task"], r["layer"], r["width"], r.get("positions", "last")) == k]
            ref = next((r["nldr"] for r in sel if r["method"] == "reference"), None)
            for meth in sorted({r["method"] for r in sel if r["method"] != "reference"}):
                by = {r["rank"]: r["nldr"] for r in sel if r["method"] == meth and r["kind"] == "hybrid_patch"}
                L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {meth} | " + " | ".join(_fmt(by.get(r)) for r in ranks)
                         + f" | {_fmt(ref)} |")
        L.append("")
    f = m / "failures.jsonl"
    if f.exists():
        fails = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        L += [f"## Failures ({len(fails)})", ""] + [f"- {x['exp']} `{x['key']}`: {x['error']}" for x in fails] + [""]
    figs = sorted((d / "figures").glob("*.png")) if (d / "figures").exists() else []
    L += ["## Figures", ""] + [f"- figures/{p.name}" for p in figs]
    out = d / "SUMMARY.md"
    out.write_text("\n".join(L) + "\n")
    return out
