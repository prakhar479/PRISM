#!/usr/bin/env python
"""End-to-end smoke test. --mock: CPU, no weights (<30s). Without --mock: 1 layer, 16k SAE, 16 samples (GPU)."""
import argparse
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from src.data.synthetic_tasks import STATE_TASKS
from src.pipeline import run_suite
from src.utils import load_config, load_json, setup_logging


def check(cond: bool, msg: str, failures: list) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-d-model", type=int, default=None, help="256 (default) or 2304")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    over = dict(run_name="smoke_test")
    if a.mock:
        over.update(layers=[4, 12], geometry_layers=[4, 12], widths=["16k", "65k"], geometry_widths=["16k", "65k"],
                    n_train=64, n_test=32, noise_seeds=2, min_pairs=8)
    else:
        over.update(layers=[12], geometry_layers=[12], widths=["16k"], geometry_widths=["16k"], n_train=16, n_test=16,
                    noise_seeds=1, min_pairs=4, candidate_multiplier=6, ranks=[0, 1, 2, 4, 8])
    cfg = load_config(None, over)
    if a.mock and a.mock_d_model:
        cfg["mock"]["d_model"] = a.mock_d_model
    out = Path(a.out or f"results/smoke_test{'_mock' if a.mock else ''}")
    setup_logging(out / "logs" / "run.log")
    t0 = time.time()
    run_suite(cfg, a.mock, out)
    print(f"\nPipeline finished in {time.time() - t0:.1f}s; verifying outputs")

    f: list = []
    for name in ("fig1_causal_split", "fig2_error_manifolds", "fig3_spectrum_and_probes", "fig4_hybrid_pareto"):
        for ext in ("pdf", "png"):
            p = out / "figures" / f"{name}.{ext}"
            check(p.exists() and p.stat().st_size > 1000, f"{p.name} written", f)
    rows = load_json(out / "metrics" / "exp1_results.json")
    ok_rows = all(r["nldr"] == r["nldr"] for r in rows)
    check(ok_rows, "exp1 NLDR finite everywhere", f)
    if a.mock:
        g = lambda t, l, w, iv: next(r["nldr"] for r in rows if (r["task"], r["layer"], r["width"], r["intervention"]) == (t, l, w, iv))
        for t in STATE_TASKS:
            check(g(t, 12, "16k", "error_only") > 0.6 and g(t, 12, "16k", "sae_only") < 0.2,
                  f"{t}: error_only >> sae_only at L12 ({g(t, 12, '16k', 'error_only'):.2f} vs {g(t, 12, '16k', 'sae_only'):.2f})", f)
        check(g("ioi_control", 12, "16k", "sae_only") > 0.6 and g("ioi_control", 12, "16k", "error_only") < 0.3,
              f"ioi_control: sae_only >> error_only at L12 ({g('ioi_control', 12, '16k', 'sae_only'):.2f} vs {g('ioi_control', 12, '16k', 'error_only'):.2f})", f)
        e3 = load_json(out / "metrics" / "exp3_results.json")["rows"]
        h = lambda r: next(x["nldr"] for x in e3 if (x["task"], x["layer"], x["width"], x["method"], x["kind"], x["rank"]) == ("add_chain", 12, "16k", "supervised_centroid_svd", "hybrid_patch", r))
        check(h(0) < 0.2 and h(16) > 0.6, f"add_chain hybrid repair: r=0 {h(0):.2f} -> r=16 {h(16):.2f}", f)
        pr = load_json(out / "metrics" / "exp2_results.json")["probes"]
        c = lambda s: next(x["r2_ordinal"] for x in pr if (x["task"], x["layer"], x["width"], x["source"]) == ("add_chain", 12, "16k", s))
        check(c("eps") > 0.6 and c("x_hat") < 0.3, f"add_chain ordinal R2: eps {c('eps'):.2f} vs x_hat {c('x_hat'):.2f}", f)
    if not a.mock:   # real-model gates: hook correctness and SAE loading/encoding
        cf = [r["nldr"] for r in rows if r["intervention"] == "clean_full" and r.get("positions") == "from_diff"]
        check(bool(cf) and min(cf) > 0.9, f"full clean patch (from_diff) restores clean answer: {cf}", f)
        fid = Path(out / "metrics" / "sae_fidelity.json")
        if fid.exists():
            fr = load_json(fid)
            check(all(r["ce_recovered"] > 0.7 for r in fr), f"SAE CE recovered on generic text: {[round(r['ce_recovered'], 3) for r in fr]}", f)
        rel = [r["rel_recon_err"] for r in rows if r["intervention"] == "error_only"]
        check(bool(rel) and max(rel) < 0.6, f"SAE relative reconstruction error plausible: {[round(x, 3) for x in rel]}", f)
    print("\nSMOKE TEST " + ("FAILED" if f else "PASSED"))
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
