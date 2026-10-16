#!/usr/bin/env python
"""Task action decider: read an overnight results dir, judge each hypothesis, and say what to do next.

    python scripts/decide.py results/overnight_XXXX        # prints and writes <dir>/DECISIONS.md

Every rule here is listed (with its rationale) in docs/RUNBOOK.md, section "Decision rules".
"""
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

ALGO = ["mod_sum", "dyck_depth", "graph_reach"]
OUT: list = []


def say(level: str, msg: str, action: str = "") -> None:
    icon = {"OK": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}[level]
    OUT.append(f"- {icon} **{level}** {msg}" + (f"\n  - → *{action}*" if action else ""))


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def pick(rows, **kw):
    return [r for r in rows if all(r.get(k, "last" if k == "positions" else None) == v for k, v in kw.items())]


def check_setup(root: Path) -> None:
    OUT.append("\n## 0. Did the run work?")
    st = root / "STATUS"
    if st.exists():
        for line in st.read_text().splitlines():
            bad = "FAIL" in line or "abort" in line or ("exit" in line and "exit 0 " not in line)
            say("FAIL" if bad else "INFO", f"`{line}`", "open logs/prism_*.log and search for 'ERROR' / 'Traceback'" if bad else "")
    pf = load(root / "preflight" / "preflight.json")
    if pf:
        for t, o in pf["task_overrides"].items():
            r = o.get("expected_retention", 1)
            lvl = "OK" if r >= 0.3 and o["filter_mode"] == "strict" else "WARN"
            say(lvl, f"`{t}`: n_shots={o['n_shots']} format={o['answer_format']} filter={o['filter_mode']} retention={r:.2f}",
                "" if lvl == "OK" else "few pairs pass the filter: results for this task are noisier; consider easier prompts (fewer steps)")
        for t in pf.get("drop_tasks", []):
            say("FAIL", f"`{t}` dropped: gemma-2-2b cannot solve it under any prompt setting",
                "make the task easier (mod_sum: n_steps 4->2; graph_reach: 3 edges) in src/data/synthetic_tasks.py, rerun with --resume")
    for f in sorted(root.glob("*/metrics/sae_fidelity.json")):
        fr = json.loads(f.read_text())
        ri = load(f.parent.parent / "run_info.json") or {}
        if ri.get("mock"):
            say("INFO", f"`{f.parent.parent.name}`: mock run, SAE CE fidelity is meaningless (no real LM)")
            continue
        bad = [r for r in fr if not (r["ce_recovered"] > 0.7)]
        if fr and not bad:
            say("OK", f"`{f.parent.parent.name}`: all {len(fr)} SAEs recover >70% of CE on generic text "
                f"(min {min(r['ce_recovered'] for r in fr):.2f})")
        for r in bad:
            say("FAIL", f"`{f.parent.parent.name}` SAE L{r['layer']} {r['width']}: CE recovered {r['ce_recovered']:.2f} (delta {r['ce_delta']:.2f} nats)",
                "SAE loading/encoding is wrong for this SAE (expect >0.8): do not interpret its exp1-3 rows; check resolve_sae_id/RealSAE.encode")
    for f in sorted(root.glob("*/metrics/failures.jsonl")):
        n = len([x for x in f.read_text().splitlines() if x.strip()])
        say("WARN", f"{n} failed work units in `{f.parent.parent.name}`",
            f"see {f.relative_to(root)}; missing SAE widths are expected, OOM/other errors -> rerun with --resume")


def check_h1(root: Path) -> None:
    OUT.append("\n## H1. Does the SAE error carry the task state? (exp1)")
    rows = load(root / "tier1_core" / "metrics" / "exp1_results.json")
    if not rows:
        say("FAIL", "no tier1 exp1 results", "tier1 did not finish: check STATUS / failures, then --resume")
        return
    cf = [r["nldr"] for r in pick(rows, intervention="clean_full", positions="from_diff")]
    info = load(root / "tier1_core" / "run_info.json") or {}
    if cf and info.get("mock"):
        say("INFO", f"mock run: clean-patch sanity {np.mean(cf):.2f} is low by design (mock writes state after the patch layer)")
    elif cf:
        m = float(np.mean(cf))
        say("OK" if m > 0.9 else "FAIL", f"sanity: full clean patch (from_diff) recovers NLDR {m:.2f} (should be ~1)",
            "" if m > 0.9 else "patching is broken: do NOT interpret anything; check hook name/dtype in run_patched")
    for t in ALGO + ["ioi_control"]:
        best = None
        for L in sorted({r["layer"] for r in rows}):
            e, s = pick(rows, task=t, layer=L, width="16k", intervention="error_only"), pick(rows, task=t, layer=L, width="16k", intervention="sae_only")
            if e and s:
                score = (s[0]["nldr"] - e[0]["nldr"]) if t == "ioi_control" else (e[0]["nldr"] - s[0]["nldr"])
                if best is None or score > best[0]:
                    best = (score, L, e[0], s[0])
        if best is None:
            continue
        _, L, e, s = best
        sep = e.get("nldr_lo", e["nldr"]) > s.get("nldr_hi", s["nldr"])
        txt = f"`{t}` best layer L{L}: error_only {e['nldr']:.2f} vs sae_only {s['nldr']:.2f} (CIs {'separate' if sep else 'overlap'})"
        if t == "ioi_control":
            ok = s["nldr"] > e["nldr"]
            say("OK" if ok else "WARN", "control " + txt, "" if ok else
                "even IOI lives in the error -> the SAE is generally lossy here, H1 is not specific to algorithmic state")
        elif sep and e["nldr"] > 0.5:
            say("OK", "H1 supported: " + txt, "go to H2/H3; write up the layer curve (fig1, fig6)")
        elif e["nldr"] > s["nldr"]:
            say("WARN", "H1 weak: " + txt, "more data: rerun tier3 (n=1000/500) for tighter CIs")
        else:
            say("FAIL", "H1 rejected: " + txt, "the SAE reconstruction carries this state; report as a negative result")
        same, diff = pick(rows, task=t, layer=L, width="16k", intervention="error_resample_same_state"), pick(rows, task=t, layer=L, width="16k", intervention="error_resample_diff_state")
        if same and diff and t != "ioi_control":
            ok = same[0]["nldr"] > diff[0]["nldr"] + 0.2
            say("OK" if ok else "WARN", f"`{t}` L{L}: error from another prompt with the SAME state {same[0]['nldr']:.2f} vs DIFFERENT state {diff[0]['nldr']:.2f}",
                "" if ok else "the error carries prompt identity rather than a clean state code; be careful claiming a 'state manifold'")


def check_h2(root: Path) -> None:
    OUT.append("\n## H2. Is the state a low-rank manifold in the error? (exp2)")
    e2 = load(root / "tier1_core" / "metrics" / "exp2_results.json")
    if not e2 or not e2["probes"]:
        say("FAIL", "no exp2 results", "check tier1 failures")
        return
    pr = e2["probes"]
    for t in ALGO:
        g = {r["source"]: r for r in pr if r["task"] == t and r["layer"] == 12 and r["width"] == "16k"}
        if "eps" not in g:
            continue
        e, x = g["eps"], g.get("x_hat", {})
        key = "r2_circular" if t == "mod_sum" else "acc"
        ok = e.get(key, 0) > x.get(key, 0) + 0.1 and e.get("selectivity", 1) > 0.2
        say("OK" if ok else "WARN", f"`{t}` L12: probe {key} eps {e.get(key, float('nan')):.2f} vs x_hat {x.get(key, float('nan')):.2f}, "
            f"eps selectivity {e.get('selectivity', float('nan')):.2f}",
            "" if ok else "state is not more decodable from the error than from the reconstruction; H2 not supported here")


def check_h3(root: Path) -> None:
    OUT.append("\n## H3. Does a small error subspace repair the state? (exp3)")
    e3 = load(root / "tier1_core" / "metrics" / "exp3_results.json")
    if not e3 or not e3["rows"]:
        say("FAIL", "no exp3 results", "check tier1 failures")
        return
    rows = e3["rows"]
    for t in ALGO:
        ref = pick(rows, task=t, layer=12, width="16k", method="reference")
        sup = sorted(pick(rows, task=t, layer=12, width="16k", method="supervised_centroid_svd", kind="hybrid_patch"), key=lambda r: r["rank"])
        rnd = {r["rank"]: r["nldr"] for r in pick(rows, task=t, layer=12, width="16k", method="random_subspace", kind="hybrid_patch")}
        if not ref or not sup:
            continue
        hit = next((r for r in sup if r["nldr"] >= 0.9 * ref[0]["nldr"]), None)
        if hit and rnd.get(hit["rank"], 0) < 0.5 * hit["nldr"]:
            say("OK", f"`{t}` L12: rank {hit['rank']} recovers {hit['nldr']:.2f} (90% of full error {ref[0]['nldr']:.2f}); random rank-{hit['rank']}: {rnd.get(hit['rank'], float('nan')):.2f}",
                "H3 supported; check generic_text_svd in tier3 to see whether the subspace is task-specific")
        else:
            say("WARN", f"`{t}` L12: no rank <= {sup[-1]['rank']} reaches 90% of full error, or random does as well",
                "the state is spread out (high-rank) at this layer; look at other layers in tier1 SUMMARY.md")
    g = load(root / "tier3_expansions" / "metrics" / "exp3_results.json")
    if g and g["rows"]:
        for t in ALGO:
            gen = {r["rank"]: r["nldr"] for r in pick(g["rows"], task=t, layer=12, width="16k", method="generic_text_svd", kind="hybrid_patch")}
            sup = {r["rank"]: r["nldr"] for r in pick(g["rows"], task=t, layer=12, width="16k", method="supervised_centroid_svd", kind="hybrid_patch")}
            if gen and sup:
                r = 16 if 16 in gen else max(gen)
                say("INFO", f"`{t}` rank {r}: generic-text subspace {gen[r]:.2f} vs task subspace {sup.get(r, float('nan')):.2f}",
                    "generic ≈ task -> universal 'dark' directions (strong claim)" if gen[r] > 0.7 * sup.get(r, 1) else "task-specific dark matter")


def check_width(root: Path) -> None:
    OUT.append("\n## Width scaling (tier2_widths)")
    rows = load(root / "tier2_widths" / "metrics" / "exp1_results.json")
    if not rows:
        say("INFO", "no width-sweep results (tier not run or deadline)")
        return
    from src.models.sae_loader import parse_width
    for t in ALGO:
        pts = sorted([(parse_width(r["width"]), r["width"], r["nldr"]) for r in pick(rows, task=t, intervention="error_only")])
        if len(pts) >= 3:
            drop = pts[0][2] - pts[-1][2]
            say("OK" if drop < 0.2 else "WARN", f"`{t}`: error_only NLDR {pts[0][1]} {pts[0][2]:.2f} -> {pts[-1][1]} {pts[-1][2]:.2f}",
                "state stays dark as width grows (supports 'SAEs systematically miss it')" if drop < 0.2 else
                "wider SAEs absorb the state: the effect is a capacity limit, not a blind spot")


def main() -> int:
    if len(sys.argv) < 2:
        cands = sorted(Path("results").glob("overnight_*"))
        if not cands:
            print("usage: decide.py results/overnight_XXXX")
            return 2
        root = cands[-1]
    else:
        root = Path(sys.argv[1])
    OUT.append(f"# Decisions for `{root.name}`")
    for fn in (check_setup, check_h1, check_h2, check_h3, check_width):
        try:
            fn(root)
        except Exception as e:  # noqa: BLE001
            say("WARN", f"{fn.__name__} could not run: {type(e).__name__}: {e}")
    text = "\n".join(OUT) + "\n"
    (root / "DECISIONS.md").write_text(text)
    print(text)
    print(f"(written to {root / 'DECISIONS.md'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
