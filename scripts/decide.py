#!/usr/bin/env python
"""Task action decider: read an overnight results dir, judge each hypothesis, and say what to do next.

    python scripts/decide.py results/overnight_XXXX        # prints and writes <dir>/DECISIONS.md
    python scripts/decide.py results/pilot_XXXX            # a pilot run: same rules on the `pilot` tier

The report opens with a viability verdict (GO / MAYBE / NO-GO / BLOCKED) built from the checks below.

Every rule here is listed (with its rationale) in docs/RUNBOOK.md, section "Decision rules".
"""
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.data.synthetic_tasks import LEGACY_TASKS, STATE_TASKS

ALGO = STATE_TASKS + LEGACY_TASKS     # tasks absent from a run are skipped
OUT: list = []
CORE = "tier1_core"                   # the tier H1-H3 are judged on ("pilot" for a pilot run)
FACTS = {"blockers": [], "h1": {}, "ioi_ok": None, "h2": {}, "h3": {}}


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
        gate = pf.get("ioi_gate")
        if gate and not gate["passed"]:
            FACTS["blockers"].append("IOI gate failed")
            say("FAIL", f"IOI gate failed: the control keeps only {gate['ioi_best_keep']:.2f} of pairs",
                "the model or pipeline is broken (gemma-2-2b solves IOI): run scripts/check_forward.py before anything else")
        if "fingerprint" not in pf:
            say("WARN", "preflight.json has no fingerprint: it predates the current task code",
                "its task choices may be stale (the 2026-10-19 preflight was): rerun preflight")
        for t, o in pf["task_overrides"].items():
            r = o.get("expected_retention", 1)
            lvl = "OK" if r >= 0.3 else "WARN"
            say(lvl, f"`{t}`: n_shots={o['n_shots']} format={o['answer_format']} filter={o['filter_mode']} retention={r:.2f}",
                "" if lvl == "OK" else "few pairs pass the filter: results for this task are noisier; see "
                "scripts/analyze_tasks.py for an easier difficulty (gen_kwargs) or more shots")
        for t in pf.get("drop_tasks", []):
            say("FAIL", f"`{t}` dropped: gemma-2-2b fails it under every shot count",
                "run scripts/analyze_tasks.py --tasks " + t + " to find a difficulty it can do and set "
                "task_overrides." + t + ".gen_kwargs; mod_sum / dyck_depth / graph_reach are beyond it at every difficulty")
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
        if bad:
            FACTS["blockers"].append(f"{len(bad)} SAE(s) with CE recovered < 0.7")
        for r in bad:
            say("FAIL", f"`{f.parent.parent.name}` SAE L{r['layer']} {r['width']}: CE recovered {r['ce_recovered']:.2f} (delta {r['ce_delta']:.2f} nats)",
                "SAE loading/encoding is wrong for this SAE (expect >0.8): do not interpret its exp1-3 rows; check resolve_sae_id/RealSAE.encode")
    for f in sorted(root.glob("*/metrics/failures.jsonl")):
        n = len([x for x in f.read_text().splitlines() if x.strip()])
        say("WARN", f"{n} failed work units in `{f.parent.parent.name}`",
            f"see {f.relative_to(root)}; missing SAE widths are expected, OOM/other errors -> rerun with --resume")


def check_h1(root: Path) -> None:
    OUT.append("\n## H1. Does the SAE error carry the task state? (exp1)")
    rows = load(root / CORE / "metrics" / "exp1_results.json")
    if not rows:
        info = load(root / CORE / "run_info.json") or {}
        if "tasks_run" in info and not info["tasks_run"]:
            FACTS["blockers"].append("no task kept enough valid pairs")
            say("FAIL", f"{CORE} ran but no task kept enough valid pairs (see the task table in {CORE}/SUMMARY.md)",
                "the base model fails the tasks as configured: check the preflight lines above and scripts/analyze_tasks.py")
        else:
            FACTS["blockers"].append(f"no {CORE} exp1 results")
            say("FAIL", f"no {CORE} exp1 results", f"{CORE} did not finish: check STATUS / failures, then --resume")
        return
    cf = [r["nldr"] for r in pick(rows, intervention="clean_full", positions="from_diff")]
    info = load(root / CORE / "run_info.json") or {}
    if cf and info.get("mock"):
        say("INFO", f"mock run: clean-patch sanity {np.mean(cf):.2f} is low by design (mock writes state after the patch layer)")
    elif cf:
        m = float(np.mean(cf))
        if m <= 0.9:
            FACTS["blockers"].append(f"clean-patch sanity {m:.2f} < 0.9 (patching broken)")
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
            FACTS["ioi_ok"] = ok
            say("OK" if ok else "WARN", "control " + txt, "" if ok else
                "even IOI lives in the error -> the SAE is generally lossy here, H1 is not specific to algorithmic state")
        elif sep and e["nldr"] > 0.5:
            FACTS["h1"][t] = ("supported", L, e["nldr"], s["nldr"])
            say("OK", "H1 supported: " + txt, "go to H2/H3; write up the layer curve (fig1, fig6)")
        elif e["nldr"] > s["nldr"]:
            FACTS["h1"][t] = ("weak", L, e["nldr"], s["nldr"])
            say("WARN", "H1 weak: " + txt, "more data: rerun tier3 (n=1000/500) for tighter CIs")
        else:
            FACTS["h1"][t] = ("rejected", L, e["nldr"], s["nldr"])
            say("FAIL", "H1 rejected: " + txt, "the SAE reconstruction carries this state; report as a negative result")
        same, diff = pick(rows, task=t, layer=L, width="16k", intervention="error_resample_same_state"), pick(rows, task=t, layer=L, width="16k", intervention="error_resample_diff_state")
        if same and diff and t != "ioi_control":
            ok = same[0]["nldr"] > diff[0]["nldr"] + 0.2
            say("OK" if ok else "WARN", f"`{t}` L{L}: error from another prompt with the SAME state {same[0]['nldr']:.2f} vs DIFFERENT state {diff[0]['nldr']:.2f}",
                "" if ok else "the error carries prompt identity rather than a clean state code; be careful claiming a 'state manifold'")


def check_h2(root: Path) -> None:
    OUT.append("\n## H2. Is the state a low-rank manifold in the error? (exp2)")
    e2 = load(root / CORE / "metrics" / "exp2_results.json")
    if not e2 or not e2["probes"]:
        say("FAIL", "no exp2 results", f"check {CORE} failures")
        return
    pr = e2["probes"]
    for t in ALGO:
        g = {r["source"]: r for r in pr if r["task"] == t and r["layer"] == 12 and r["width"] == "16k"}
        if "eps" not in g:
            continue
        e, x, sf = g["eps"], g.get("x_hat", {}), g.get("surface", {})
        key = "r2_circular" if t == "mod_sum" else "acc"
        beats_xhat = e.get(key, 0) > x.get(key, 0) + 0.1 and e.get("selectivity", 1) > 0.2
        # the token baseline must lose on accuracy AND on R^2 (the add_chain total is linear in tokens: R^2 = 1)
        r2 = "r2_circular" if t == "mod_sum" else "r2_ordinal"
        beats_tokens = not sf or all(e.get(k, 0) > sf.get(k, 0) + 0.1 for k in {key, r2})
        txt = (f"`{t}` L12: probe {key} eps {e.get(key, float('nan')):.2f} vs x_hat {x.get(key, float('nan')):.2f}, "
               f"prompt tokens only {sf.get(key, float('nan')):.2f} (R^2 eps {e.get(r2, float('nan')):.2f} vs tokens "
               f"{sf.get(r2, float('nan')):.2f}), eps selectivity {e.get('selectivity', float('nan')):.2f}")
        FACTS["h2"][t] = "clean" if beats_xhat and beats_tokens else "confounded" if beats_xhat else "no"
        if beats_xhat and beats_tokens:
            say("OK", txt)
        elif beats_xhat:
            say("WARN", txt, "the prompt's tokens alone decode the state as well: the probe cannot show the model "
                "computed it (expected for add_chain / var_binding); base the H2 claim on exp1/exp3 or on entity_track")
        else:
            say("WARN", txt, "state is not more decodable from the error than from the reconstruction; H2 not supported here")


def check_h3(root: Path) -> None:
    OUT.append("\n## H3. Does a small error subspace repair the state? (exp3)")
    e3 = load(root / CORE / "metrics" / "exp3_results.json")
    if not e3 or not e3["rows"]:
        say("FAIL", "no exp3 results", f"check {CORE} failures")
        return
    rows = e3["rows"]
    for t in ALGO:
        ref = pick(rows, task=t, layer=12, width="16k", method="reference")
        sup = sorted(pick(rows, task=t, layer=12, width="16k", method="supervised_centroid_svd", kind="hybrid_patch"), key=lambda r: r["rank"])
        rnd = {r["rank"]: r["nldr"] for r in pick(rows, task=t, layer=12, width="16k", method="random_subspace", kind="hybrid_patch")}
        if not ref or not sup:
            continue
        hit = next((r for r in sup if r["nldr"] >= 0.9 * ref[0]["nldr"]), None)
        FACTS["h3"][t] = hit["rank"] if hit and rnd.get(hit["rank"], 0) < 0.5 * hit["nldr"] else None
        if hit and rnd.get(hit["rank"], 0) < 0.5 * hit["nldr"]:
            say("OK", f"`{t}` L12: rank {hit['rank']} recovers {hit['nldr']:.2f} (90% of full error {ref[0]['nldr']:.2f}); random rank-{hit['rank']}: {rnd.get(hit['rank'], float('nan')):.2f}",
                "H3 supported; check generic_text_svd in tier3 to see whether the subspace is task-specific")
        else:
            say("WARN", f"`{t}` L12: no rank <= {sup[-1]['rank']} reaches 90% of full error, or random does as well",
                f"the state is spread out (high-rank) at this layer; look at other layers in {CORE}/SUMMARY.md")
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


def verdict(mock: bool) -> list:
    """GO / MAYBE / NO-GO / BLOCKED: is the project worth a full run? Built from the facts the checks recorded."""
    h1 = FACTS["h1"]
    sup = [t for t, v in h1.items() if v[0] == "supported"]
    weak = [t for t, v in h1.items() if v[0] == "weak"]
    if FACTS["blockers"]:
        v, why = "BLOCKED", "the run cannot be interpreted yet: " + "; ".join(FACTS["blockers"])
        nxt = "fix the blockers listed under '0. Did the run work?', then rerun (./run.sh pilot)"
    elif sup and FACTS["ioi_ok"] is not False:
        v, why = "GO", (f"the SAE error carries the state for {', '.join(sup)} (error_only > sae_only, CIs separate)"
                        + ("" if FACTS["ioi_ok"] is None else "; the IOI control sits in the SAE features as expected"))
        nxt = "run the full schedule: ./run.sh start (H1 at 6 layers x 2 widths, width scaling to 1m, larger n)"
    elif sup or weak:
        v = "MAYBE"
        why = (f"error_only beats sae_only for {', '.join(sup + weak)} but " +
               ("the CIs overlap" if not sup else "the IOI control also lives in the error (the SAE is lossy in general)"))
        nxt = ("tighten the CIs before committing a night: ./run.sh start --tiers tier1_core --hours 2 "
               "(400/200 pairs, 6 layers x {16k, 65k})")
    elif h1:
        v, why = "NO-GO", "sae_only >= error_only for every task: the SAE reconstruction carries the state"
        nxt = "a negative H1 result; before giving up, check other layers in the pilot SUMMARY.md and a wider SAE (65k)"
    else:
        v, why, nxt = "BLOCKED", "no exp1 results to judge", "see '0. Did the run work?'"
    icon = {"GO": "🟢", "MAYBE": "🟡", "NO-GO": "🔴", "BLOCKED": "⛔"}[v]
    L = ["", f"## Verdict: {icon} **{v}**" + ("  (mock run: plumbing check only, not a result)" if mock else ""), "",
         f"- Why: {why}", f"- Next: {nxt}"]
    rows = [f"| `{t}` | {s} | L{L_} | {e:.2f} | {x:.2f} | {FACTS['h2'].get(t, '-')} | "
            f"{'-' if FACTS['h3'].get(t) is None else 'rank ' + str(FACTS['h3'][t])} |" for t, (s, L_, e, x) in h1.items()]
    if rows:
        L += ["", "| task | H1 | best layer | error_only | sae_only | H2 probe (clean = beats x_hat and prompt tokens) | "
              "H3 (smallest rank with 90% repair) |", "|---|---|---|---|---|---|---|"] + rows
    return L


def main() -> int:
    global CORE
    if len(sys.argv) < 2:
        cands = sorted((p for p in Path("results").glob("*_2*") if p.name.split("_")[0] in ("overnight", "pilot")
                        and not p.name.endswith("_mock")), key=lambda p: p.name.split("_", 1)[1])
        if not cands:
            print("usage: decide.py results/overnight_XXXX | results/pilot_XXXX")
            return 2
        root = cands[-1]
    else:
        root = Path(sys.argv[1])
    if not (root / "tier1_core").exists() and (root / "pilot").exists():
        CORE = "pilot"
    OUT.append(f"# Decisions for `{root.name}`")
    for fn in (check_setup, check_h1, check_h2, check_h3, check_width):
        try:
            fn(root)
        except Exception as e:  # noqa: BLE001
            say("WARN", f"{fn.__name__} could not run: {type(e).__name__}: {e}")
    mock = bool((load(root / CORE / "run_info.json") or {}).get("mock"))
    OUT[1:1] = verdict(mock)
    text = "\n".join(OUT) + "\n"
    (root / "DECISIONS.md").write_text(text)
    print(text)
    print(f"(written to {root / 'DECISIONS.md'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
