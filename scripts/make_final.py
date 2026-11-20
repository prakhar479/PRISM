#!/usr/bin/env python
"""Pass 2 (DEV, no model): tables for the final compute pass from ingested runs (E9b / E16 / E17 / E18 / E20 / E22 use the E9
rules and E10b / E19 the E10 rules, writing <EXP>_illusion.csv / <EXP>_steering.csv; E2r writes E2r_conditions.csv for comparison
with E2; E17-E22 (2026-11-18) are exploratory: <EXP>_das_stability.csv for rotations of the same rank, E22_transfer.csv).

    python scripts/make_final.py [--only E9,E10,E11,E14] [--n-boot 10000] [--out tables]

Per experiment (complete, non-smoke runs only):
  <EXP>_conditions.csv  every condition key per (cell, evaluation family, filter): flip rate / NLDR / prob recovery, the gap to
                        full_A, paired 95% bootstrap CIs (harness/final.py; same metric code as E8)
  E9_illusion.csv       feat_A_along_U, full_along_U, repair per estimator (incl. DAS) and rank, with the random-U null
                        (mean / p95 / max over seeds) and each DAS rotation's train interchange accuracy. Thresholds for an
                        "illusion" are the humans' (addendum); this table only reports.
  E10_steering.csv      steer_x1/x2/x4 per estimator and rank: flip rate = share of pairs moving to A's answer, with the random
                        null; q2_flip_ge_50, q2_random_p95_le_5, q2_both = the preregistered Q2 rule, applied mechanically
  E11_spec.csv, E14_spec.csv   repair share closed: same task vs every control incl. other_task (E8's specificity code), per
                        evaluation family
  E14_lang_gap.csv      per (task, language, fit source native | english): gaps of feat_A_errB / feat_A_meanPos / feat_A_zero,
                        joined with E13's Gate A / Gate B (tables/E13_behaviour_lang.csv) and the template sign-off;
                        m2_gap_above_en_ci = the gap's lower CI bound is above English's upper bound (same task, native fit);
                        readable = Gate A and Gate B and validated (only such rows may enter M2/M3 statements)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from harness import e8 as E
from harness import final as FN
from harness import langfamilies as LF
from harness.config import das_rank
from harness.conditions import is_steer
from harness.profile import REPO_ROOT, load_profile


ILLUSION_EXPS = ("E9", "E9b", "E16", "E17", "E18", "E20", "E22")   # E9 rules (E16 has no feat_A_along_U: illusion NA)
STEERING_EXPS = ("E10", "E10b", "E19")                             # E10 rules (Q2 columns are confirmatory only for E10)


def runs_of(root: Path, exp: str, include_tiny: bool = False):
    for man in sorted(root.glob(f"{exp}/*/manifest.json")):
        m = json.loads(man.read_text())
        if m.get("status") != "complete" or ((m.get("smoke") or m.get("tiny")) and not include_tiny):
            continue
        yield m, man.parent


def _dir_key(path: Path, run_hash: str) -> dict:
    """directions/<task>_L<layer>_<sae>[__fit-<families>].{json,npz} -> join keys."""
    stem, _, fit = path.stem.partition("__fit-")
    task, rest = stem.split("_L", 1)
    layer = int(rest.split("_")[0])
    return {"run_hash": run_hash, "task": task, "layer": layer, "sae": rest.split("_", 1)[1], "fit_families": fit or "pilot"}


_INIT = {}


def das_init(d: int, k: int, seed: int) -> np.ndarray:
    """The rotation every DAS fit starts from (harness/model/das.py Rotation(d, k, seed)): fits that share (k, seed) share it,
    and 3-epoch fits stay close to it (deep analysis 2026-11-18), so geometry must be read with it projected out."""
    if (d, k, seed) not in _INIT:
        from harness.model.das import Rotation
        _INIT[(d, k, seed)] = Rotation(d, k, seed).R.detach().double().numpy()
    return _INIT[(d, k, seed)]


def das_learned(U: np.ndarray, est: str, cfg_seed: int):
    """(learned component: orthonormal basis of U with its init span projected out, init capture = share of U's energy
    left in the init span; Haar level k/d)."""
    from harness.config import parse_das
    p = parse_das(est)
    I = das_init(U.shape[0], p["k"], cfg_seed if p["seed"] is None else p["seed"])
    U = U.astype(np.float64)
    cap = float(np.sum((I.T @ U) ** 2) / U.shape[1])
    R = U - I @ (I.T @ U)
    q, s_, _ = np.linalg.svd(R, full_matrices=False)
    return q[:, s_ > 1e-6 * max(s_.max(), 1e-30)], cap


def das_info(run_dir: Path, run_hash: str, cfg_seed: int = 0) -> pd.DataFrame:
    rows = []
    for j in run_dir.glob("directions/*.json"):
        d = json.loads(j.read_text())
        npz = j.with_suffix(".npz")
        z = np.load(npz) if npz.exists() else None
        for est, v in d.items():
            if est.startswith("das") and isinstance(v, dict):
                meta = v.get("meta") or {}
                cap = (das_learned(z[est], est, cfg_seed)[1] if z is not None and est in z.files
                       and np.isfinite(z[est]).all() else np.nan)
                rows.append({**_dir_key(j, run_hash), "estimator": est, "das_train_acc": meta.get("train_interchange_acc"),
                             "das_loss_last": meta.get("loss_last_epoch"), "das_init_capture": cap})
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


def with_null(tab, conds):
    t = tab[tab.condition.isin(conds) & (tab.estimator != "random")]
    null = FN.random_null(tab[tab.condition.isin(conds)])
    return t.merge(null, on=["run_hash", *FN.GROUP, "filter", "condition", "r"], how="left") if not null.empty else t


def spec_table(m, pp, filters, n_boot):
    pp = FN.eval_group(pp)
    out = []
    for cell, g in pp.groupby(FN.GROUP, sort=False):
        for flt in filters:
            _, _, sp = E.analyse_cell(g, flt, n_boot, m["config"].get("seed", 0))
            out += [{"exp_id": m["exp_id"], "run_hash": m["run_hash"], **dict(zip(FN.GROUP, cell)),
                     "role": FN.config_role(m, *cell), **r} for r in sp]
    return pd.DataFrame(out)


def lang_of(family: str) -> str:
    return LF.LANG_OF.get(family, ("?",))[0]


def e14_gap(tab: pd.DataFrame, tables: Path) -> pd.DataFrame:
    base = tab[tab.condition.isin(["feat_A_errB", "feat_A_meanPos", "feat_A_zero"])].copy()
    base["lang"] = base.family.map(lang_of)
    base["fit_source"] = np.where(base.fit_families.astype(str).str.endswith("_en") & (base["lang"] != "en"), "english", "native")
    base["validated"] = base.family.map(LF.is_validated)
    beh = tables / "E13_behaviour_lang.csv"
    if beh.exists():
        b = pd.read_csv(beh)
        b = b[b.get("split", pd.Series("test", index=b.index)).fillna("test") == "test"]       # gates of record: test split
        b = b[["task", "family", "gate_A_viable", "gate_B_within_10pts", "acc_clean"]]
        base = base.merge(b, on=["task", "family"], how="left")
    else:
        base["gate_A_viable"] = base["gate_B_within_10pts"] = np.nan
    jk = ["run_hash", "task", "layer", "sae", "condition", "filter"]
    en = base[(base.lang == "en") & (base.fit_source == "native")][jk + ["gap_flip_rate_hi", "gap_nldr_hi"]].rename(
        columns={"gap_flip_rate_hi": "en_gap_flip_hi", "gap_nldr_hi": "en_gap_nldr_hi"})
    base = base.merge(en, on=jk, how="left", validate="m:1")
    gb = base.gate_B_within_10pts.astype(str).isin(["True", "n/a (English)"])
    base["readable"] = base.gate_A_viable.astype(str).eq("True") & gb & base.validated
    # M2 is only stated for readable, non-English rows (addendum D); others are NA, not False
    for met, col in (("flip_rate", "en_gap_flip_hi"), ("nldr", "en_gap_nldr_hi")):
        raw = (base[f"gap_{met}_lo"] > base[col]).astype("boolean")
        base[f"m2_gap_above_en_ci_{met.split('_')[0]}"] = raw.where(base.readable & (base.lang != "en") & base[col].notna(), pd.NA)
    return base


def e9_paired(m, pp, filters, n_boot):
    """full_along_U - feat_A_along_U per (cell, estimator, r), paired over pairs (the illusion contrast)."""
    pp = FN.eval_group(pp)
    rows = []
    for cell, g in pp.groupby(FN.GROUP, sort=False):
        keys = {(k[1], k[2], k[3], k[4]) for k in g.groupby(FN.KEY, dropna=False).groups if k[0] == "full_along_U" and k[1] != "random"}
        for flt in filters:
            pairs = [(("full_along_U", e, s_, r, sd), ("feat_A_along_U", e, s_, r, sd)) for e, s_, r, sd in keys]
            for x in FN.paired_diffs(g, flt, pairs, n_boot, m["config"].get("seed", 0)):
                rows.append({"run_hash": m["run_hash"], **dict(zip(FN.GROUP, cell)), "filter": flt, "estimator": x["key1"][1],
                             "r": x["key1"][3], "illusion_diff": x["diff"], "illusion_diff_lo": x["diff_lo"], "illusion_diff_hi": x["diff_hi"]})
    return rows


def das_cosines(run_dir, run_hash):
    """Top principal cosine between each error estimator at rank k and DAS_k (rank k only: DAS columns are unordered)."""
    from harness.directions import principal_cosines
    rows = []
    for f in run_dir.glob("directions/*.npz"):
        z = np.load(f)
        key = _dir_key(f, run_hash)
        for das in [k for k in z.files if k.startswith("das")]:
            k = das_rank(das)
            if not np.isfinite(z[das]).all():
                continue
            for est in ("pca_pooled", "answer_sep"):
                if est in z.files and z[est].shape[1] >= k:
                    c = principal_cosines(z[est][:, :k], z[das])
                    rows.append({**key, "estimator": est, "r": k, "das": das, "cos_top_das": float(c[0])})
    return rows


def das_stability(run_dir, run_hash, cfg_seed: int = 0):
    """E17: agreement between DAS rotations of the same rank trained with different seeds / epochs / learning rates (top and
    mean principal cosine), with each rotation's train interchange accuracy. Raw cosines are inflated when two fits share an
    initialisation (same k and seed), so the learned components (init span projected out) are compared too. Exploratory."""
    from harness.directions import principal_cosines
    rows = []
    for f in run_dir.glob("directions/*.npz"):
        z = np.load(f)
        meta = json.loads(f.with_suffix(".json").read_text()) if f.with_suffix(".json").exists() else {}
        key = _dir_key(f, run_hash)
        das = sorted(k for k in z.files if k.startswith("das") and np.isfinite(z[k]).all())
        acc = {k: ((meta.get(k) or {}).get("meta") or {}).get("train_interchange_acc") for k in das}
        for i, a in enumerate(das):
            for b in das[i + 1:]:
                if das_rank(a) != das_rank(b):
                    continue
                c = principal_cosines(z[a], z[b])
                (la, ca), (lb, cb) = das_learned(z[a], a, cfg_seed), das_learned(z[b], b, cfg_seed)
                cl = principal_cosines(la, lb) if la.shape[1] and lb.shape[1] else np.array([np.nan])
                rows.append({**key, "r": das_rank(a), "das_a": a, "das_b": b, "acc_a": acc[a], "acc_b": acc[b],
                             "cos_top": float(c[0]), "cos_mean": float(np.mean(c)),
                             "cos_top_learned": float(cl[0]), "cos_mean_learned": float(np.nanmean(cl)),
                             "init_capture_a": ca, "init_capture_b": cb,
                             "haar_p99": FN.HAAR_P99.get(das_rank(a), FN.HAAR_P99_EXPLORATORY.get(das_rank(a)))})
    return rows


def lang_transfer(t):
    """E22: per (task, evaluation language family, filter, condition, estimator, rank): flip rate with the rotation / U fitted on
    the same language (native) and on English (transfer), side by side with each one's random-U p95. Exploratory, descriptive."""
    x = t[(t.estimator != "random") & t.condition.isin(["full_along_U", "feat_A_along_U"]) & (t.r > 0)].copy()
    fam = x.family.astype(str)
    x["fit_source"] = np.where(x.fit_families.astype(str) == fam, "native",
                               np.where(x.fit_families.astype(str).str.endswith("_en"), "english", "other"))
    cols = ["flip_rate", "flip_rate_lo", "flip_rate_hi", "nldr", "random_p95", "das_train_acc", "n_pairs"]
    cols = [c for c in cols if c in x]
    k = ["task", "layer", "family", "filter", "condition", "estimator", "r"]
    nat = x[x.fit_source == "native"][k + cols].rename(columns={c: f"{c}_native" for c in cols})
    eng = x[x.fit_source == "english"][k + cols].rename(columns={c: f"{c}_english" for c in cols})
    out = nat.merge(eng, on=k, how="outer")
    out["transfer_minus_native"] = out.get("flip_rate_english") - out.get("flip_rate_native")
    return out.sort_values(k)


def e9_rules(t, pair_rows, cos_rows):
    """Decision columns fixed on 2026-11-16 (PREREGISTRATION_ADDENDUM.md E; harness/final.py constants). Nullable booleans:
    NA = not evaluable (no random null with >= 20 seeds, no CI, rank 0, or a DAS row whose fit is invalid), never False."""
    keys = ["run_hash", *FN.GROUP, "filter", "estimator", "r"]
    p95 = t.get("random_p95", pd.Series(np.nan, index=t.index))
    rn = t.get("random_n", pd.Series(0, index=t.index)).fillna(0)
    evaluable = p95.notna() & t.flip_rate.notna() & t.flip_rate_lo.notna() & (rn >= E.MIN_RANDOM_SEEDS)
    eff = ((t.flip_rate >= p95 + FN.EFFECT_PTS) & (t.flip_rate_lo > p95)).astype("boolean")
    t["effective"] = eff.where(evaluable, pd.NA)
    if "das_train_acc" in t:
        acc = pd.to_numeric(t.das_train_acc, errors="coerce")
        valid = (acc >= FN.DAS_VALID_ACC).astype("boolean").where(acc.notna(), pd.NA)
        t["das_fit_valid"] = valid.where(t.estimator.str.startswith("das"), pd.NA)
        bad = t.das_fit_valid.fillna(True).eq(False)
        t.loc[bad, "effective"] = pd.NA                        # an invalid DAS fit is not evidence (addendum E)
    if pair_rows:
        pr = pd.DataFrame(pair_rows).drop_duplicates(keys)
        t = t.merge(pr, on=keys, how="left", validate="m:1")
    piv = t.pivot_table(index=keys, columns="condition", values="effective", aggfunc="first", dropna=False)
    piv = piv.rename(columns={"full_along_U": "full_along_U_effective", "feat_A_along_U": "feature_mediated"}).reset_index()
    keep = [c for c in ("full_along_U_effective", "feature_mediated") if c in piv]
    t = t.merge(piv[keys + keep], on=keys, how="left", validate="m:1")
    if "illusion_diff" in t and "full_along_U_effective" in t:
        fe = t.full_along_U_effective.astype("boolean")
        ill = (fe.fillna(False) & (t.illusion_diff >= FN.EFFECT_PTS) & (t.illusion_diff_lo > 0)).astype("boolean")
        t["illusion"] = ill.where(fe.notna() & t.illusion_diff.notna(), pd.NA)
    if cos_rows:
        ck = ["run_hash", "task", "layer", "sae", "fit_families", "estimator", "r"]
        c = pd.DataFrame(cos_rows).drop_duplicates(ck)
        t = t.merge(c[ck + ["cos_top_das"]], on=ck, how="left", validate="m:1")
        t["das_alignment"] = [FN.alignment(x, r) for x, r in zip(t.cos_top_das, t.r)]
    return t


def m3_table(m, pp, filters, n_boot):
    """E14 M3: share closed by repair with the English-fit U minus with the native U, same evaluation pairs (paired)."""
    pp = FN.eval_group(pp)
    rows = []
    for (task, layer, sae, fam), g in pp.groupby(["task", "layer", "sae", "family"]):
        lang = lang_of(fam)
        if lang in ("en", "?"):
            continue
        en_fam = fam.split("_")[0] + "_en"
        gn, gt = g[g.fit_families == fam], g[g.fit_families == en_fam]
        if gn.empty or gt.empty:
            continue
        for flt in filters:
            for est in ("pca_pooled", "answer_sep"):
                for r in (1, 2):
                    x = FN.share_diff(gn, gt, flt, ("repair", est, "same_task", r, -1), n_boot, m["config"].get("seed", 0))
                    if x:
                        rows.append({"run_hash": m["run_hash"], "task": task, "layer": layer, "sae": sae, "family": fam, "lang": lang,
                                     "filter": flt, "estimator": est, "r": r, **x})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="E9,E10,E11,E14,E9b,E10b,E2r,E16,E17,E18,E19,E20,E22")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "tables"))
    ap.add_argument("--runs", default=None, help="default: the profile's runs_root")
    ap.add_argument("--include-tiny", action="store_true", help="plumbing tests only: also read tiny/smoke runs (never results)")
    a = ap.parse_args()
    root = Path(a.runs) if a.runs else load_profile(a.profile or "dev").root("runs_root")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for exp in a.only.split(","):
        tabs, specs, dirs, e9_pairs, cos_rows, m3_rows, stab_rows = [], [], [], [], [], [], []
        for m, d in runs_of(root, exp, a.include_tiny):
            pp = pd.read_parquet(d / "per_pair.parquet")
            flts = m["config"].get("filters", ["f_pilot"])
            tabs.append(FN.run_table(m, pp, flts, a.n_boot))
            if exp in ("E11", "E14"):
                specs.append(spec_table(m, pp, flts, a.n_boot))
            dirs.append(das_info(d, m["run_hash"], m["config"].get("seed", 0)))
            if exp in ILLUSION_EXPS:
                e9_pairs += e9_paired(m, pp, flts, a.n_boot)
                cos_rows += das_cosines(d, m["run_hash"])
            if exp == "E14":
                m3_rows += m3_table(m, pp, flts, a.n_boot)
            if exp in ILLUSION_EXPS:
                stab_rows += das_stability(d, m["run_hash"], m["config"].get("seed", 0))
        if not tabs:
            print(f"{exp}: no complete runs")
            continue
        tab = pd.concat(tabs, ignore_index=True)
        tab.to_csv(out / f"{exp}_conditions.csv", index=False)
        msg = f"{exp}: {len(tab)} condition rows"
        if exp in ILLUSION_EXPS:
            t = with_null(tab, ["feat_A_along_U", "full_along_U", "repair"])
            di = pd.concat(dirs, ignore_index=True) if any(len(x) for x in dirs) else pd.DataFrame()
            if not di.empty:
                t = t.merge(di, on=["run_hash", "task", "layer", "sae", "fit_families", "estimator"], how="left", validate="m:1")
            t = e9_rules(t, e9_pairs, cos_rows)
            t.to_csv(out / f"{exp}_illusion.csv", index=False)
            ill = t[t.condition == "full_along_U"].get("illusion", pd.Series(dtype="boolean"))
            msg += f"; {exp}_illusion {len(t)} (illusion (estimator, rank) cells: {int(ill.fillna(False).sum())})"
            if stab_rows:
                pd.DataFrame(stab_rows).to_csv(out / f"{exp}_das_stability.csv", index=False)
                msg += f"; {exp}_das_stability {len(stab_rows)}"
            if exp == "E22":
                tr = lang_transfer(t)
                tr.to_csv(out / "E22_transfer.csv", index=False)
                msg += f"; E22_transfer {len(tr)}"
        if exp in STEERING_EXPS:
            t = with_null(tab, sorted({c for c in tab.condition.unique() if is_steer(c)}))
            p95 = t.get("random_p95", pd.Series(np.nan, index=t.index))
            rn = t.get("random_n", pd.Series(0, index=t.index)).fillna(0)
            t["q2_flip_ge_50"] = (t.flip_rate >= 0.50).astype("boolean").where(t.flip_rate.notna(), pd.NA)
            ok_null = p95.notna() & (rn >= E.MIN_RANDOM_SEEDS)
            t["q2_random_p95_le_5"] = (p95 <= 0.05).astype("boolean").where(ok_null, pd.NA)   # >= 20 random directions (addendum E)
            t["q2_both"] = (t.q2_flip_ge_50 & t.q2_random_p95_le_5)
            t.to_csv(out / f"{exp}_steering.csv", index=False)
            msg += f"; {exp}_steering {len(t)} (q2_both true: {int(t.q2_both.fillna(False).sum())}, not evaluable: {int(t.q2_both.isna().sum())})"
        if specs:
            sp = pd.concat(specs, ignore_index=True)
            if not sp.empty:
                sp = E.correct(sp)
            sp.to_csv(out / f"{exp}_spec.csv", index=False)
            msg += f"; {exp}_spec {len(sp)}"
        if exp == "E14":
            g = e14_gap(tab, out)
            g.to_csv(out / "E14_lang_gap.csv", index=False)
            m3 = pd.DataFrame(m3_rows)
            if not m3.empty:
                rd = g[g.condition == "feat_A_errB"][["task", "family", "filter", "readable"]].drop_duplicates()
                m3 = m3.merge(rd, on=["task", "family", "filter"], how="left")
                m3["m3_usable"] = m3.readable.fillna(False).astype(bool) & (m3.share_native_lo > 0)
                m3["m3_similar"] = (m3.diff_lo > -FN.M3_MARGIN).astype("boolean").where(m3.m3_usable, pd.NA)
            m3.to_csv(out / "E14_m3_transfer.csv", index=False)
            msg += f"; E14_m3_transfer {len(m3)}"
            nat = g[(g.condition == "feat_A_errB") & (g.fit_source == "native")][["task", "family", "filter", "readable"]].drop_duplicates()
            msg += f"; E14_lang_gap {len(g)} (readable (task, language, filter): {int(nat.readable.sum())} of {len(nat)})"
        print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
