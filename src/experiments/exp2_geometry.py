"""Experiment 2: geometry of the SAE residual error. Spectra, state probes, PCA/LDA manifold coordinates.

Every fitted object (centering, SVD, scalers, probes, PCA, LDA) is fit on the train split only and
evaluated / applied on the held-out test split.
"""
from __future__ import annotations

import contextlib
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.common import RunContext, decompose_batched, latent_features
from src.models.runtime import blas_threads_per_worker, cpu_workers
from src.utils import save_json, save_npz, save_table

log = logging.getLogger("prism")
ALPHAS = [1.0, 10.0, 100.0, 1e3, 1e4]
N_SV_SAVED = 128


def centered_svd(M: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Singular values and right singular vectors (rows of Vt) of the column-centered matrix."""
    Mc = M.double() - M.double().mean(0, keepdim=True)
    _, S, Vt = torch.linalg.svd(Mc, full_matrices=False)
    return S.numpy(), Vt.numpy()


def spectrum_summary(S: np.ndarray) -> Dict[str, float]:
    var = S ** 2
    cum = np.cumsum(var) / max(var.sum(), 1e-30)
    out = {f"evr_top{k}": float(cum[min(k, len(cum)) - 1]) for k in (1, 2, 4, 8, 16, 32)}
    out["rank90"] = int(np.searchsorted(cum, 0.9) + 1)
    out["participation_ratio"] = float(var.sum() ** 2 / max((var ** 2).sum(), 1e-30))
    return out


def fit_probes(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, circular_mod: Optional[int],
               shuffle_seed: Optional[int] = None) -> dict:
    """Ridge probes (alpha by internal CV on train): class accuracy, ordinal R^2, optional circular R^2.

    With shuffle_seed, a control probe is also fit on permuted train labels (acc_shuffled); selectivity = acc - acc_shuffled.
    """
    clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=ALPHAS)).fit(Xtr, ytr)
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(Xtr, ytr.astype(float))
    res = {"acc": float(clf.score(Xte, yte)),
           "r2_ordinal": float(r2_score(yte.astype(float), reg.predict(Xte)))}
    if circular_mod:
        ang = lambda y: np.stack([np.cos(2 * np.pi * y / circular_mod), np.sin(2 * np.pi * y / circular_mod)], 1)
        circ = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(Xtr, ang(ytr))
        res["r2_circular"] = float(r2_score(ang(yte), circ.predict(Xte)))
    if shuffle_seed is not None:
        yperm = np.random.default_rng(shuffle_seed).permutation(ytr)
        ctl = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=ALPHAS)).fit(Xtr, yperm)
        res["acc_shuffled"] = float(ctl.score(Xte, yte))
        res["selectivity"] = res["acc"] - res["acc_shuffled"]
    return res


def manifold_coords(Ftr: np.ndarray, ytr: np.ndarray, Fte: np.ndarray) -> Dict[str, np.ndarray]:
    pca = PCA(n_components=3, random_state=0).fit(Ftr)
    n_cls = len(np.unique(ytr))
    n_lda = max(1, min(2, n_cls - 1))
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=n_lda).fit(Ftr, ytr)
    lda_te = lda.transform(Fte)
    if lda_te.shape[1] < 2:
        lda_te = np.concatenate([lda_te, np.zeros((len(lda_te), 2 - lda_te.shape[1]))], 1)
    return {"pca": pca.transform(Fte), "pca_evr": pca.explained_variance_ratio_, "lda": lda_te}


def _stack(base, layer: int) -> Tuple[torch.Tensor, np.ndarray]:
    """Clean and corrupt prompts both contribute points, labelled with their own state."""
    X = torch.cat([base.resid_clean[layer], base.resid_corrupt[layer]])
    y = np.concatenate([base.states_clean, base.states_corrupt])
    return X, y


def surface_features(train_items, test_items) -> Tuple[np.ndarray, np.ndarray]:
    """Prompt-token baseline in _stack order (clean rows, then corrupt rows): token counts plus (position, token)
    indicators, vocabulary from the train split only. A residual probe only shows that the model computed the
    state if it beats this: e.g. the add_chain total is a linear function of the prompt's tokens."""
    def seqs(items):
        return [it.clean_ids for it in items] + [it.corrupt_ids for it in items]

    tr, te = seqs(train_items), seqs(test_items)
    vocab = {t: j for j, t in enumerate(sorted({t for q in tr for t in q}))}
    slots = {pt: j for j, pt in enumerate(sorted({(k, t) for q in tr for k, t in enumerate(q)}))}

    def encode(qs):
        X = np.zeros((len(qs), len(vocab) + len(slots)), np.float32)
        for r, q in enumerate(qs):
            for k, t in enumerate(q):
                if t in vocab:
                    X[r, vocab[t]] += 1
                j = slots.get((k, t))
                if j is not None:
                    X[r, len(vocab) + j] = 1
        return X

    return encode(tr), encode(te)


def _features(ctx: RunContext, sae, task: str, layer: int) -> dict:
    """GPU part of a unit: decompose train/test residuals and gather SAE latents -> CPU arrays."""
    splits = ctx.data[task]
    Xtr, ytr = _stack(splits["train"], layer)
    Xte, yte = _stack(splits["test"], layer)
    xh_tr, e_tr, _ = decompose_batched(sae, Xtr)
    xh_te, e_te, _ = decompose_batched(sae, Xte)
    feats = {"x": (Xtr.numpy(), Xte.numpy()), "x_hat": (xh_tr.numpy(), xh_te.numpy()), "eps": (e_tr.numpy(), e_te.numpy())}
    if ctx.cfg.get("latent_probe", True):
        feats["latents"] = latent_features(sae, Xtr, Xte, ctx.cfg.get("latent_probe_cols", 4096))
    feats["surface"] = surface_features(splits["train"].items, splits["test"].items)
    return {"feats": feats, "ytr": ytr, "yte": yte}


def _analyze(ctx: RunContext, task: str, layer: int, width: str, f: dict) -> List[dict]:
    """CPU part of a unit (probes, spectra, PCA/LDA); runs on a worker thread while the GPU moves on."""
    feats, ytr, yte = f["feats"], f["ytr"], f["yte"]
    chance = float(np.mean(yte == np.bincount(ytr).argmax()))
    arrays: Dict[str, np.ndarray] = {"labels": yte}
    rows = []
    for src, (tr, te) in feats.items():
        pr = fit_probes(tr, ytr, te, yte, 7 if task == "mod_sum" else None, shuffle_seed=ctx.cfg["seed"] + layer)
        rows.append(dict(kind="probe", task=task, layer=layer, width=width, source=src, chance_acc=chance,
                         n_features=tr.shape[1], **pr))
    for src in ("eps", "x_hat"):
        tr, te = feats[src]
        S, _ = centered_svd(torch.from_numpy(tr))
        rows.append(dict(kind="spectrum", task=task, layer=layer, width=width, source=src,
                         total_var=float((S ** 2).sum() / max(len(tr) - 1, 1)), **spectrum_summary(S)))
        arrays[f"sv_{src}"] = S[:N_SV_SAVED]
        arrays[f"cumevr_{src}"] = (np.cumsum(S ** 2) / (S ** 2).sum())[:N_SV_SAVED]
        mc = manifold_coords(tr, ytr, te)
        arrays[f"pca_{src}"], arrays[f"lda_{src}"], arrays[f"pca_evr_{src}"] = mc["pca"], mc["lda"], mc["pca_evr"]
    save_npz(ctx.out_dir / "tensors" / f"exp2_{task}_L{layer}_{width}.npz", **arrays)
    log.info("exp2 %s L%d %s: probe acc %s", task, layer, width,
             {r["source"]: round(r["acc"], 2) for r in rows if r["kind"] == "probe"})
    return rows


def _unit(ctx: RunContext, sae, task: str, layer: int, width: str) -> List[dict]:
    return _analyze(ctx, task, layer, width, _features(ctx, sae, task, layer))


def _blas_limit(n: int):
    """Limit BLAS threads per worker so parallel probe fits do not oversubscribe the CPU."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=max(1, n))
    except ImportError:
        return contextlib.nullcontext()


def run_exp2(ctx: RunContext, layers: Optional[Sequence[int]] = None, widths: Optional[Sequence[str]] = None) -> List[dict]:
    layers = list(layers if layers is not None else ctx.cfg["geometry_layers"])
    widths = list(widths if widths is not None else ctx.cfg["geometry_widths"])
    ctx.prepare(sorted(set(layers) | set(ctx.cached_layers)))
    units = ctx.unit_log("exp2")
    todo = lambda l, w: [t for t in ctx.data if not units.done(units.key(t, l, w))]
    combos = [(l, w) for l in layers for w in widths if todo(l, w)]
    workers = cpu_workers()
    pending: List[Tuple[str, Future]] = []

    def drain(block: bool) -> None:
        keep = []
        for i, (key, fut) in enumerate(pending):
            if block or fut.done() or len(pending) - i > 4 * workers:   # bound queued feature memory
                # results computed before the deadline are always kept
                ctx.run_unit("exp2", key, fut.result, units, check_deadline=False)
            else:
                keep.append((key, fut))
        pending[:] = keep

    with ThreadPoolExecutor(workers, thread_name_prefix="exp2-cpu") as pool, \
            _blas_limit(blas_threads_per_worker()):
        for layer, width, sae in ctx.iter_saes("exp2", combos):
            for task in todo(layer, width):
                key = units.key(task, layer, width)
                try:
                    f = _features(ctx, sae, task, layer)                       # GPU, main thread
                except Exception as e:  # noqa: BLE001
                    log.error("unit exp2/%s FAILED: %s: %s", key, type(e).__name__, e)
                    ctx.record_failure("exp2", key, e)
                    continue
                pending.append((key, pool.submit(_analyze, ctx, task, layer, width, f)))   # CPU, worker threads
            del sae
            drain(block=False)
        drain(block=True)
    rows = units.rows()
    strip = lambda r: {k: v for k, v in r.items() if k != "kind"}
    probe_rows = [strip(r) for r in rows if r["kind"] == "probe"]
    spec_rows = [strip(r) for r in rows if r["kind"] == "spectrum"]
    out = ctx.out_dir / "metrics"
    save_json({"probes": probe_rows, "spectra": spec_rows}, out / "exp2_results.json")
    if probe_rows:
        save_table(probe_rows, out / "exp2_probes")
        save_table(spec_rows, out / "exp2_spectra")
    return probe_rows
