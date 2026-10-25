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
    # judge on from_diff rows (patch every token from the first differing one): only there does the full clean patch
    # restore the answer at every layer. Layers where clean_full < 0.8 cannot show where the state is.
    pos = "from_diff" if pick(rows, positions="from_diff") else "last"
    OUT.append(f"(judged on `{pos}` patching, at layers where the full clean patch restores >= 0.8)")
    for t in ALGO + ["ioi_control"]:
        best = None
        for L in sorted({r["layer"] for r in rows}):
            q = dict(task=t, layer=L, width="16k", positions=pos)
            e, s, cf, nz = (pick(rows, intervention=iv, **q) for iv in ("error_only", "sae_only", "clean_full", "noise_control"))
            if e and s and (not cf or cf[0]["nldr"] >= 0.8 or info.get("mock")):
                score = (s[0]["nldr"] - e[0]["nldr"]) if t == "ioi_control" else (e[0]["nldr"] - s[0]["nldr"])
                if best is None or score > best[0]:
                    best = (score, L, e[0], s[0], nz[0]["nldr"] if nz else float("nan"))
        if best is None:
            continue
        _, L, e, s, nz = best
        sep = e.get("nldr_lo", e["nldr"]) > s.get("nldr_hi", s["nldr"])
        sae_higher = s.get("nldr_lo", s["nldr"]) > e.get("nldr_hi", e["nldr"])
        ci = "error_only clearly higher" if sep else "sae_only clearly higher" if sae_higher else "CIs overlap"
        txt = (f"`{t}` most error-favourable layer L{L}: error_only {e['nldr']:.2f} vs sae_only {s['nldr']:.2f} "
               f"({ci}); noise_control {nz:.2f} "
               f"(clean error adds {e['nldr'] - nz:+.2f} over erasing the corrupt error)")
        if t == "ioi_control":
            ok = s["nldr"] > e["nldr"]
            FACTS["ioi_ok"] = ok
            say("OK" if ok else "WARN", "control " + txt, "" if ok else
                "even IOI lives in the error -> the SAE is generally lossy here, H1 is not specific to algorithmic state")
        elif sep and e["nldr"] > 0.5:
            FACTS["h1"][t] = ("supported", L, e["nldr"], s["nldr"])
            say("OK", "H1 supported: " + txt, "go to H2/H3; write up the layer curve (fig1, fig6)")
        elif e["nldr"] > s["nldr"] or (s.get("nldr_lo", s["nldr"]) <= e.get("nldr_hi", e["nldr"])):
            FACTS["h1"][t] = ("weak", L, e["nldr"], s["nldr"])
            say("WARN", "H1 weak: " + txt, "more data: rerun tier3 (n=1000/500) for tighter CIs")
        else:
            FACTS["h1"][t] = ("rejected", L, e["nldr"], s["nldr"])
            say("FAIL", "H1 rejected: " + txt, "the SAE reconstruction carries this state; report as a negative result")
        same, diff = (pick(rows, task=t, layer=L, width="16k", intervention=iv, positions=pos)
                      for iv in ("error_resample_same_state", "error_resample_diff_state"))
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
        e1 = load(root / CORE / "metrics" / "exp1_results.json") or []
        cf_last = pick(e1, task=t, layer=12, width="16k", intervention="clean_full", positions="last")
        if beats_xhat and cf_last and cf_last[0]["nldr"] < 0.5:
            FACTS["h2"][t] = "decodable, not used here"
            say("WARN", txt, f"the probes read the final token, but patching the final token at L12 restores only "
                f"{cf_last[0]['nldr']:.2f}: the state is linearly present in eps there but not yet causally used "
                f"(the model reads it from earlier positions). Interesting, but not evidence for H1")
        elif beats_xhat and beats_tokens:
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
    # hybrid patch x_hat_clean + U_r U_r^T eps_clean: rank 0 = sae_only. Score = share of the SAE's gap
    # (1 - sae_only) that the rank-r error subspace closes, against a random rank-r subspace.
    OUT.append("(score: share of the gap 1 - sae_only closed by a rank-r error subspace; random subspace as the null)")
    pos = "from_diff" if pick(rows, positions="from_diff") else "last"
    for t in [t for t in ALGO + ["ioi_control"] if any(r["task"] == t for r in rows)]:
        best = None
        for L in sorted({r["layer"] for r in rows if r["task"] == t}):
            q = dict(task=t, layer=L, width="16k", kind="hybrid_patch", positions=pos)
            rnd = {r["rank"]: r["nldr"] for r in pick(rows, method="random_subspace", **q)}
            for m in ("supervised_centroid_svd", "unsupervised_svd"):
                cur = {r["rank"]: r["nldr"] for r in pick(rows, method=m, **q)}
                if 0 not in cur:
                    continue
                gap = 1.0 - cur[0]
                if gap < 0.05:
                    continue
                for k in sorted(cur):
                    closed, rclosed = (cur[k] - cur[0]) / gap, (rnd.get(k, cur[0]) - cur[0]) / gap
                    if k > 0 and closed >= 0.5 and rclosed < 0.2:
                        if best is None or (k, -gap) < (best[0], -best[2]):
                            best = (k, L, gap, closed, rclosed, m)
                        break
        if best is None:
            FACTS["h3"][t] = None
            say("INFO", f"`{t}`: no layer where a rank <= 64 error subspace closes half of a >= 0.05 SAE gap",
                "the SAE (x_hat) already carries the causal state here; nothing low-rank is missing")
            continue
        k, L, gap, closed, rclosed, m = best
        FACTS["h3"][t] = (k, L, gap, closed)
        say("OK", f"`{t}` L{L}: the SAE misses {gap:.2f} of the effect; a rank-{k} error subspace ({m}) closes "
            f"{closed:.0%} of that gap (random rank-{k}: {rclosed:.0%})",
            "a small, low-rank, task-relevant residue: test width scaling (does it shrink with 65k-1m?) and whether "
            "it is task-specific (generic_text_svd / transfer)")
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
    pos = "from_diff" if pick(rows, positions="from_diff") else "last"
    for t in ALGO + ["ioi_control"]:
        # the SAE's causal gap 1 - sae_only at each width: a blind spot stays, a capacity limit shrinks
        pts = sorted((parse_width(r["width"]), r["width"], 1.0 - r["nldr"])
                     for r in pick(rows, task=t, intervention="sae_only", positions=pos))
        if len(pts) >= 3:
            curve = ", ".join(f"{w} {g:.2f}" for _, w, g in pts)
            stays = pts[-1][2] >= 0.1 and pts[-1][2] >= 0.6 * pts[0][2]
            say("OK" if stays else "INFO", f"`{t}`: SAE gap (1 - sae_only, {pos}) by width: {curve}",
                "the missed state persists at the widest SAE: a systematic blind spot (the refined claim holds)" if stays
                else "wider SAEs absorb it: a capacity limit, not a blind spot (the refined claim fails here)")


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
    elif sup or [t for t in weak if h1[t][2] > h1[t][3]]:
        v = "MAYBE"
        why = (f"error_only beats sae_only for {', '.join(sup + [t for t in weak if h1[t][2] > h1[t][3]])} but " +
               ("the CIs overlap" if not sup else "the IOI control also lives in the error (the SAE is lossy in general)"))
        nxt = ("tighten the CIs before committing a night: ./run.sh start --tiers tier1_core --hours 2 "
               "(400/200 pairs, 6 layers x {16k, 65k})")
    elif h1 and any(FACTS["h3"].get(t) and FACTS["h3"][t][2] >= 0.1 and FACTS["h3"][t][0] <= 8 for t in h1):
        res = [f"{t} (L{FACTS['h3'][t][1]}: gap {FACTS['h3'][t][2]:.2f}, rank {FACTS['h3'][t][0]} closes {FACTS['h3'][t][3]:.0%})"
               for t in h1 if FACTS["h3"].get(t) and FACTS["h3"][t][2] >= 0.1 and FACTS["h3"][t][0] <= 8]
        v = "REFINE"
        why = ("the strong H1 is rejected (the SAE reconstruction carries most of the causal state), but the SAE misses "
               "a small low-rank, task-relevant part: " + "; ".join(res))
        ioi = FACTS["h3"].get("ioi_control")
        if ioi:
            why += (f". Caveat: the IOI control shows a residue too (L{ioi[1]}: gap {ioi[2]:.2f}, rank {ioi[0]}), so this "
                    "is partly a general SAE loss, not specific to state tracking")
        nxt = ("reframe as 'how much causal state do SAEs miss, where, and does it shrink with width': run width scaling "
               "on those layers (./run.sh start --tiers tier2_widths) with from_diff patching")
    elif h1:
        v, why = "NO-GO", "sae_only >= error_only for every task and no low-rank error residue: the SAE carries the state"
        nxt = "a negative H1 result; before giving up, check a wider SAE (65k) and other layers in the SUMMARY.md"
    else:
        v, why, nxt = "BLOCKED", "no exp1 results to judge", "see '0. Did the run work?'"
    icon = {"GO": "🟢", "MAYBE": "🟡", "REFINE": "🔵", "NO-GO": "🔴", "BLOCKED": "⛔"}[v]
    L = ["", f"## Verdict: {icon} **{v}**" + ("  (mock run: plumbing check only, not a result)" if mock else ""), "",
         f"- Why: {why}", f"- Next: {nxt}"]
    rows = [f"| `{t}` | {s} | L{L_} | {e:.2f} | {x:.2f} | {FACTS['h2'].get(t, '-')} | "
            f"{'-' if not FACTS['h3'].get(t) else 'L%d: rank %d closes %.0f%% of a %.2f gap' % (FACTS['h3'][t][1], FACTS['h3'][t][0], 100 * FACTS['h3'][t][3], FACTS['h3'][t][2])} |"
            for t, (s, L_, e, x) in h1.items()]
    if rows:
        L += ["", "| task | H1 | layer | error_only | sae_only | H2 probe (clean = beats x_hat and prompt tokens) | "
              "H3 (error subspace vs the SAE gap) |", "|---|---|---|---|---|---|---|"] + rows
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
    CORE = next((t for t in ("tier1_core", "pilot", "refine_layers") if (root / t).exists()), "tier1_core")
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
