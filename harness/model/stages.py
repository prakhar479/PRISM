"""Stages 1-3 (handoff §6).

Stage 1  (model)  clean final-position outputs of A and B for every pair of a prompt set, and the layer-L residual at
                  the patched positions first_diff..T-1 of A and B (model dtype) for each requested layer.
Stage 2  (no model) SAE pass over cached span rows: x_hat (fp32) and L0 per row; eps = x - x_hat is re-derived exactly.
Stage 3  (model)  patched runs on B: resid_post[L] of B is computed once per batch of pairs, the condition's rows are
                  written at the patched positions, and blocks L+1.. run from there (the pilot's verified resume scheme).
Only the final position is unembedded. All stage outputs are content-addressed cache entries, filled in shards.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from harness.batching import Budget, group_by_length
from harness.cache import Cache, Entry, lock_many
from harness.conditions import Spans
from harness.device import is_oom, empty_cache
from harness.model import ModelSpec, final_outputs, hook_name
from src.models import runtime

log = logging.getLogger("harness")
OUT_FIELDS = ("logit_A", "logit_B", "lse", "top1_id", "cand_logits")


def _tok(rows, idx, key, device) -> torch.Tensor:
    return torch.tensor([rows[i][key] for i in idx], dtype=torch.long, device=device)


def shard_plan(rows: Sequence[dict], shard_pairs: int) -> List[List[int]]:
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["T"], i))
    return [order[a:a + shard_pairs] for a in range(0, len(order), shard_pairs)]


def site_entry(cache: Cache, spec: ModelSpec, set_hash: str, layer: int, site: str) -> Entry:
    """Span cache of one (layer, site): site = spec.site (resid_post: the cell's own, keys unchanged) or a sublayer site
    of harness.composite (mlp_in, mlp_out, attn_z, resid_post at another layer)."""
    return cache.entry("spans", model=spec.key(), set_hash=set_hash, layer=int(layer), site=site)


def entries_stage1(cache: Cache, spec: ModelSpec, set_hash: str, layers: Sequence):
    """`layers`: ints (resid_post spans, the pilot's) and/or (layer, site) tuples (composite dictionaries, P12)."""
    out = cache.entry("clean_out", model=spec.key(), set_hash=set_hash)
    spans = {}
    for L in layers:
        if isinstance(L, tuple):
            spans[L] = site_entry(cache, spec, set_hash, L[0], L[1])
        else:
            spans[L] = site_entry(cache, spec, set_hash, L, spec.site)
    return out, spans


def site_hook(model, key):
    """(TransformerLens hook name, transform of the hooked activation) for a span key (int = resid_post)."""
    if not isinstance(key, tuple):
        return hook_name(key), None
    l, site = key
    if site == "resid_post":
        return hook_name(l), None
    if site == "mlp_out":                  # = HF post_feedforward_layernorm output (what the MLP adds to the residual)
        return f"blocks.{l}.hook_mlp_out", None
    if site == "mlp_in":                   # = HF pre_feedforward_layernorm output (the transcoders' input)
        w = model.blocks[l].ln2.w
        return f"blocks.{l}.ln2.hook_normalized", (lambda a, w=w: a * w.to(a.device, a.dtype))
    if site == "attn_z":                   # = HF o_proj input (heads concatenated)
        return f"blocks.{l}.attn.hook_z", (lambda a: a.flatten(-2))
    raise ValueError(f"unknown site {site}")


def _forward_capture(model, tokens: torch.Tensor, layers: Sequence, bi: torch.Tensor, pi: torch.Tensor):
    """Last-position logits and, per span key, rows act[bi, pi] (model dtype, CPU)."""
    got: Dict = {}
    hooks = []
    for L in layers:
        name, tf = site_hook(model, L)
        def fn(act, hook, L=L, tf=tf):
            a = act[bi, pi].detach()
            got[L] = (tf(a) if tf is not None else a).cpu()
            return None
        hooks.append((name, fn))
    logits = runtime.last_logits(model, tokens, hooks)
    return logits, got


def write_sublayer_params(model, spec: ModelSpec, cache: Cache, layers: Sequence[int]) -> Dict[int, Entry]:
    """Per layer the attention output map of the residual (W_O [n_heads*d_head, d], b_O, ln1_post weight, eps): what an
    attention-SAE composite needs to map z-space errors into the residual (no model in stage 2)."""
    out = {}
    for l in layers:
        e = cache.entry("sublayer_params", model=spec.key(), layer=int(l))
        if not e.complete:
            with e.writer_lock():
                e.open()
                if not e.complete:
                    b = model.blocks[l]
                    W_O = b.attn.W_O.detach().float().cpu()
                    e.write_shard(0, {"W_O": W_O.reshape(-1, W_O.shape[-1]).contiguous(),
                                      "b_O": b.attn.b_O.detach().float().cpu(), "ln1_post_w": b.ln1_post.w.detach().float().cpu()})
                    e.mark_complete({"eps": float(b.ln1_post.eps), "layer": int(l)})
        out[l] = e
    return out


def load_sublayer_params(cache: Cache, spec: ModelSpec, layers: Sequence[int]) -> Dict[int, dict]:
    out = {}
    for l in layers:
        e = cache.entry("sublayer_params", model=spec.key(), layer=int(l)).require_complete()
        t = e.read_shard(0)
        out[l] = {"W_O": t["W_O"], "b_O": t["b_O"], "ln1_post_w": t["ln1_post_w"], "eps": float(json.loads((e.dir / "_COMPLETE").read_text())["eps"])}
    return out


def stage1(model, spec: ModelSpec, rows: Sequence[dict], set_hash: str, layers: Sequence[int], cache: Cache,
           budget: Budget, device, shard_pairs: int = 96) -> dict:
    """Stage-1 fill under writer locks on every entry it writes (concurrent processes: one fills, the others wait and
    then find the shards done)."""
    out_e, span_e = entries_stage1(cache, spec, set_hash, layers)
    if all(e.complete for e in (out_e, *span_e.values())):
        return {"clean_out": out_e, "spans": span_e}          # already filled (e.g. by an earlier release): no lock, no wait
    with lock_many([out_e, *span_e.values()]):
        return _stage1_fill(model, spec, rows, set_hash, layers, cache, budget, device, shard_pairs)


@torch.inference_mode()
def _stage1_fill(model, spec: ModelSpec, rows: Sequence[dict], set_hash: str, layers: Sequence[int], cache: Cache,
                 budget: Budget, device, shard_pairs: int = 96) -> dict:
    out_e, span_e = entries_stage1(cache, spec, set_hash, layers)
    out_e.open()
    for e in span_e.values():
        e.open()
    plan = shard_plan(rows, shard_pairs)
    for si, P in enumerate(plan):
        need_L = [L for L, e in span_e.items() if not e.shard_done(si)]
        if out_e.shard_done(si) and not need_L:
            continue
        res = {k: [] for k in ("A", "B")}
        caps = {k: {L: [] for L in layers} for k in ("A", "B")}
        order: List[int] = []
        for T, members in group_by_length([rows[i]["T"] for i in P]).items():
            members = [P[m] for m in members]
            a = 0
            while a < len(members):
                b = members[a:a + budget.rows_for(T)]
                try:
                    lens = [T - rows[i]["first_diff"] for i in b]
                    bi = torch.repeat_interleave(torch.arange(len(b)), torch.tensor(lens)).to(device)
                    pi = torch.cat([torch.arange(rows[i]["first_diff"], T) for i in b]).to(device)
                    cand = _tok(rows, b, "cand_ids", device)
                    aA, aB = _tok(rows, b, "ans_A", device), _tok(rows, b, "ans_B", device)
                    done = {}                              # committed only when both prompts succeeded
                    for k, key in (("A", "clean_ids"), ("B", "corrupt_ids")):
                        lg, got = _forward_capture(model, _tok(rows, b, key, device), layers, bi, pi)
                        done[k] = (final_outputs(lg, aA, aB, cand), got)
                    for k, (fo, got) in done.items():
                        res[k].append(fo)
                        for L in layers:
                            caps[k][L].append(got[L])
                except Exception as e:  # noqa: BLE001
                    if is_oom(e):
                        empty_cache()
                        if budget.shrink():
                            continue
                    raise
                order += b
                a += len(b)
        pidx = torch.tensor(order, dtype=torch.long)
        lens = torch.tensor([rows[i]["T"] - rows[i]["first_diff"] for i in order], dtype=torch.long)
        if not out_e.shard_done(si):
            t = {"pair_index": pidx}
            for k in ("A", "B"):
                for f in OUT_FIELDS:
                    t[f"{k}_{f}"] = torch.from_numpy(np.concatenate([r[f] for r in res[k]]))
            out_e.write_shard(si, t)
        for L in need_L:
            span_e[L].write_shard(si, {"pair_index": pidx, "span_len": lens, "x_A": torch.cat(caps["A"][L]),
                                       "x_B": torch.cat(caps["B"][L])})
        log.info("stage1 shard %d/%d (%d pairs)", si + 1, len(plan), len(P))
    info = {"n_pairs": len(rows), "n_shards": len(plan), "budget_backoffs": budget.backoffs}
    for e in [out_e, *span_e.values()]:
        if not e.complete:
            e.mark_complete(info)
    return {"clean_out": out_e, "spans": span_e}


def load_clean(entry: Entry, n_pairs: int) -> Dict[str, Dict[str, np.ndarray]]:
    """{'A': {field: [n_pairs, ...]}, 'B': {...}} in prompt-set order."""
    entry.require_complete()
    names = ["pair_index"] + [f"{k}_{f}" for k in ("A", "B") for f in OUT_FIELDS]
    t = entry.read_all(names)
    order = torch.argsort(t["pair_index"])
    if not torch.equal(t["pair_index"][order], torch.arange(n_pairs)):
        raise ValueError("clean_out does not cover the prompt set")
    return {k: {f: t[f"{k}_{f}"][order].numpy() for f in OUT_FIELDS} for k in ("A", "B")}


def load_spans(entry: Entry, rows: Sequence[dict], pair_idx: Sequence[int], sae_entry: Optional[Entry] = None) -> Spans:
    """Spans (float32) for pairs `pair_idx` in that order; with sae_entry also x_hat / eps / l0."""
    entry.require_complete()
    t = entry.read_all(["pair_index", "span_len", "x_A", "x_B"])
    ptr_all = np.concatenate([[0], np.cumsum(t["span_len"].numpy())])
    where = {int(p): k for k, p in enumerate(t["pair_index"].tolist())}
    sel = [where[int(i)] for i in pair_idx]
    take = torch.cat([torch.arange(ptr_all[k], ptr_all[k + 1]) for k in sel]) if sel else torch.zeros(0, dtype=torch.long)
    lens = np.array([ptr_all[k + 1] - ptr_all[k] for k in sel])
    ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    rs = [rows[i] for i in pair_idx]
    pos = torch.cat([torch.arange(r["first_diff"], r["T"]) for r in rs]) if rs else torch.zeros(0, dtype=torch.long)
    off = torch.cat([torch.arange(r["T"] - 1 - r["first_diff"], -1, -1) for r in rs]) if rs else torch.zeros(0, dtype=torch.long)
    slot = [s for r in rs for s in r["slots"][r["first_diff"]:]] if rs and "slots" in rs[0] else None
    meta = {"T": np.array([r["T"] for r in rs]), "first_diff": np.array([r["first_diff"] for r in rs]),
            "ans_A": np.array([r["ans_A"] for r in rs]), "ans_B": np.array([r["ans_B"] for r in rs]),
            "family": np.array([r.get("family", "pilot") for r in rs], dtype=object),
            "state_A": np.array([r.get("state_clean", -1) for r in rs]), "state_B": np.array([r.get("state_corrupt", -1) for r in rs])}
    sp = Spans([r["pair_id"] for r in rs], ptr, pos, off, t["x_A"][take].float(), t["x_B"][take].float(), slot=slot, meta=meta)
    if sae_entry is not None:
        s = sae_entry.require_complete().read_all(["pair_index", "xh_A", "xh_B", "l0_A", "l0_B"])
        if not torch.equal(s["pair_index"], t["pair_index"]):
            raise ValueError("SAE cache rows are not aligned with the span cache")
        sp.xh_A, sp.xh_B = s["xh_A"][take], s["xh_B"][take]
        sp.e_A, sp.e_B = sp.x_A - sp.xh_A, sp.x_B - sp.xh_B
        sp.l0_A, sp.l0_B = s["l0_A"][take], s["l0_B"][take]
    return sp


def last_rows(sp: Spans) -> torch.Tensor:
    """Index of each pair's final-token row."""
    return torch.from_numpy(sp.row_ptr[1:] - 1).long()


def stage2(cache: Cache, spans_entry: Entry, sae, sae_key: dict, device, block_latents: Optional[int] = None) -> Entry:
    with cache.entry("sae_out", spans=spans_entry.key, sae=sae_key).writer_lock():
        return _stage2_fill(cache, spans_entry, sae, sae_key, device, block_latents)


@torch.inference_mode()
def _stage2_fill(cache: Cache, spans_entry: Entry, sae, sae_key: dict, device, block_latents: Optional[int] = None) -> Entry:
    from harness.sae import decompose

    spans_entry.require_complete()
    e = cache.entry("sae_out", spans=spans_entry.key, sae=sae_key).open()
    for si in spans_entry.shards():
        if e.shard_done(si):
            continue
        s = spans_entry.read_shard(si)
        xh_A, _, l0_A = decompose(sae, s["x_A"], block_latents=block_latents, device=device)
        xh_B, _, l0_B = decompose(sae, s["x_B"], block_latents=block_latents, device=device)
        e.write_shard(si, {"pair_index": s["pair_index"], "xh_A": xh_A.cpu(), "xh_B": xh_B.cpu(), "l0_A": l0_A.cpu(),
                           "l0_B": l0_B.cpu()})
    if not e.complete:
        e.mark_complete({"sae": sae_key})
    return e


def stage2_composite(cache: Cache, spans_entry: Entry, site_entries: Dict, comp, sae_key: dict, sub_params=None) -> Entry:
    """Stage 2 of a composite dictionary (harness.composite): x_hat = x - eps per span row, same entry as stage2()."""
    with cache.entry("sae_out", spans=spans_entry.key, sae=sae_key).writer_lock():
        e = cache.entry("sae_out", spans=spans_entry.key, sae=sae_key).open()
        if e.complete:
            return e
        spans_entry.require_complete()
        for ent in site_entries.values():
            ent.require_complete()
        acc = comp.eps_for_entries(spans_entry, site_entries, sub_params, log=log.info)
        for si, a in acc.items():
            if e.shard_done(si):
                continue
            s = spans_entry.read_shard(si)
            e.write_shard(si, {"pair_index": s["pair_index"], "xh_A": s["x_A"].float() - a["eps_A"],
                               "xh_B": s["x_B"].float() - a["eps_B"], "l0_A": a["l0_A"], "l0_B": a["l0_B"]})
        e.mark_complete({"sae": sae_key, "composite": True})
        return e


def stage2_codes(cache: Cache, spans_entry: Entry, sae, sae_key: dict, device, alpha_min: float,
                 block_latents: Optional[int] = None) -> Tuple[Entry, Entry]:
    with cache.entry("sae_codes", spans=spans_entry.key, sae=sae_key, alpha_min=float(alpha_min)).writer_lock():
        return _stage2_codes_fill(cache, spans_entry, sae, sae_key, device, alpha_min, block_latents)


@torch.inference_mode()
def _stage2_codes_fill(cache: Cache, spans_entry: Entry, sae, sae_key: dict, device, alpha_min: float,
                       block_latents: Optional[int] = None) -> Tuple[Entry, Entry]:
    """E5: sparse sub-threshold codes of the clean prompt A (entries alpha_min*theta <= z <= theta, z > 0; rows local
    to each spans shard) and a DecoderTable with the W_dec rows of every latent that occurs. No model involved."""
    from harness.sae import DecoderTable, encode_sparse, subthreshold_only

    spans_entry.require_complete()
    e = cache.entry("sae_codes", spans=spans_entry.key, sae=sae_key, alpha_min=float(alpha_min)).open()
    for si in spans_entry.shards():
        if e.shard_done(si):
            continue
        s = spans_entry.read_shard(si)
        c = subthreshold_only(encode_sparse(sae, s["x_A"], alpha_min=alpha_min, block_latents=block_latents, device=device))
        e.write_shard(si, {"pair_index": s["pair_index"], "n_rows": torch.tensor([c.n_rows]), "row": c.row,
                           "col": c.col, "z": c.z, "threshold": c.threshold})
    if not e.complete:
        e.mark_complete({"sae": sae_key, "alpha_min": float(alpha_min), "prompt": "A", "entries": "subthreshold"})
    d = cache.entry("sae_codes_dec", codes=e.key).open()
    if not d.complete:
        cols = torch.unique(torch.cat([e.read_shard(si)["col"] for si in e.shards()] or [torch.zeros(0, dtype=torch.long)]))
        tab = DecoderTable.from_sae(sae, cols)
        d.write_shard(0, {"cols": tab.cols, "W": tab.W})
        d.mark_complete({"n_latents": int(len(tab.cols))})
    return e, d


def active_entry(cache: Cache, spans_entry: Entry, sae_key: dict) -> Entry:
    return cache.entry("sae_active", spans=spans_entry.key, sae=sae_key)


def stage2_active(cache: Cache, spans_entry: Entry, sae, sae_key: dict, device, block_latents: Optional[int] = None):
    with active_entry(cache, spans_entry, sae_key).writer_lock():
        return _stage2_active_fill(cache, spans_entry, sae, sae_key, device, block_latents)


def _active(sae, X, block_latents, device):
    """ACTIVE codes of any dictionary as SparseCodes with z = activation (threshold field 0)."""
    from harness.sae import SparseCodes, encode_sparse
    c = encode_sparse(sae, X, alpha_min=1.0, block_latents=block_latents, device=device)
    if getattr(sae, "encode_active", None) is None:              # JumpReLU: keep z > threshold (f = z there)
        m = c.z > c.threshold
        c = SparseCodes(c.row[m], c.col[m], c.z[m], torch.zeros(int(m.sum())), c.n_rows, 1.0)
    return c


@torch.inference_mode()
def _stage2_active_fill(cache: Cache, spans_entry: Entry, sae, sae_key: dict, device, block_latents: Optional[int] = None):
    """P0: active codes of A and B at every span row (rows local to each spans shard) and a DecoderTable of every latent
    that occurs, for the code-level write-back condition."""
    from harness.sae import DecoderTable

    spans_entry.require_complete()
    e = active_entry(cache, spans_entry, sae_key).open()
    for si in spans_entry.shards():
        if e.shard_done(si):
            continue
        s = spans_entry.read_shard(si)
        out = {"pair_index": s["pair_index"]}
        for who in ("A", "B"):
            c = _active(sae, s[f"x_{who}"], block_latents, device)
            out.update({f"n_rows_{who}": torch.tensor([c.n_rows]), f"row_{who}": c.row, f"col_{who}": c.col, f"z_{who}": c.z})
        e.write_shard(si, out)
    # the decoder table is written BEFORE the codes entry is marked complete: a kill in between leaves the entry
    # incomplete, so the next stage 2 redoes both (stage2_done checks both entries)
    d = cache.entry("sae_active_dec", codes=e.key).open()
    if not d.complete:
        cols = torch.unique(torch.cat([e.read_shard(si)[f"col_{w}"] for si in e.shards() for w in ("A", "B")]
                                      or [torch.zeros(0, dtype=torch.long)]))
        base = getattr(sae, "base", sae)                         # AddonDict: the base dictionary's decoder
        tab = DecoderTable.from_sae(base, cols)
        d.write_shard(0, {"cols": tab.cols, "W": tab.W})
        d.mark_complete({"n_latents": int(len(tab.cols))})
    if not e.complete:
        e.mark_complete({"sae": sae_key, "entries": "active", "prompts": "A,B"})
    return e, d


def load_active(active: Entry, spans_entry: Entry, pair_idx: Sequence[int]):
    """(SparseCodes A, SparseCodes B) of ACTIVE entries for pairs `pair_idx`, rows numbered like load_spans."""
    from harness.sae import SparseCodes

    active.require_complete()
    t = spans_entry.require_complete().read_all(["pair_index", "span_len"])
    ptr_all = np.concatenate([[0], np.cumsum(t["span_len"].numpy())])
    where = {int(p): k for k, p in enumerate(t["pair_index"].tolist())}
    sel = [where[int(i)] for i in pair_idx]
    take = torch.cat([torch.arange(ptr_all[k], ptr_all[k + 1]) for k in sel]) if sel else torch.zeros(0, dtype=torch.long)
    out = []
    for who in ("A", "B"):
        rows, cols, zs, off = [], [], [], 0
        for si in active.shards():
            s = active.read_shard(si)
            rows.append(s[f"row_{who}"] + off)
            cols.append(s[f"col_{who}"])
            zs.append(s[f"z_{who}"])
            off += int(s[f"n_rows_{who}"][0])
        if off != int(ptr_all[-1]):
            raise ValueError("active-code rows are not aligned with the span cache")
        z = torch.cat(zs)
        full = SparseCodes(torch.cat(rows), torch.cat(cols), z, torch.zeros_like(z), off, 1.0)
        out.append(full.subset_rows(take))
    return out[0], out[1]


def load_codes(codes_entry: Entry, spans_entry: Entry, pair_idx: Sequence[int]):
    """SparseCodes of A for pairs `pair_idx`, rows numbered like load_spans (same pair order)."""
    from harness.sae import SparseCodes

    codes_entry.require_complete()
    t = spans_entry.require_complete().read_all(["pair_index", "span_len"])
    ptr_all = np.concatenate([[0], np.cumsum(t["span_len"].numpy())])
    where = {int(p): k for k, p in enumerate(t["pair_index"].tolist())}
    rows, cols, zs, ths, off = [], [], [], [], 0
    for si in codes_entry.shards():
        s = codes_entry.read_shard(si)
        rows.append(s["row"] + off)
        cols.append(s["col"])
        zs.append(s["z"])
        ths.append(s["threshold"])
        off += int(s["n_rows"][0])
    if off != int(ptr_all[-1]):
        raise ValueError("code rows are not aligned with the span cache")
    full = SparseCodes(torch.cat(rows), torch.cat(cols), torch.cat(zs), torch.cat(ths), off,
                       float(codes_entry.parts["alpha_min"]))
    sel = [where[int(i)] for i in pair_idx]
    take = torch.cat([torch.arange(ptr_all[k], ptr_all[k + 1]) for k in sel]) if sel else torch.zeros(0, dtype=torch.long)
    return full.subset_rows(take)


def load_decoder(dec_entry: Entry):
    from harness.sae import DecoderTable

    t = dec_entry.require_complete().read_shard(0)
    return DecoderTable(t["cols"], t["W"])


@torch.inference_mode()
def stage3(model, rows: Sequence[dict], pair_idx: Sequence[int], layer: int, ptr: np.ndarray,
           conds: Dict[str, Optional[torch.Tensor]], budget: Budget, device, hooked: bool = False) -> Dict[str, dict]:
    """Outputs per condition for pairs pair_idx (rows of a Spans built for the same pair order; ptr = its row_ptr).
    conds[name] = [M, d] patch rows, or None for the unpatched B run. hooked=True uses full forwards with a hook
    instead of resuming at L+1 (for verification)."""
    names = list(conds)
    res = {n: {f: [None] * len(pair_idx) for f in OUT_FIELDS} for n in names}
    groups = group_by_length([rows[i]["T"] for i in pair_idx])
    for T, ks in groups.items():
        a = 0
        while a < len(ks):
            cap = budget.rows_for(T)
            kb = ks[a:a + cap]
            per = max(1, cap // len(kb))            # conditions per forward
            try:
                pidx = [pair_idx[k] for k in kb]
                tB = _tok(rows, pidx, "corrupt_ids", device)
                R = None if hooked else runtime.resid_at(model, tB, layer)
                aA, aB = _tok(rows, pidx, "ans_A", device), _tok(rows, pidx, "ans_B", device)
                cand = _tok(rows, pidx, "cand_ids", device)
                for c0 in range(0, len(names), per):
                    chunk = names[c0:c0 + per]
                    outs = _run_chunk(model, rows, pidx, kb, chunk, conds, ptr, layer, T, tB, R, device, hooked)
                    for n, lg in zip(chunk, outs):
                        fo = final_outputs(lg, aA, aB, cand)
                        for f in OUT_FIELDS:
                            for j, k in enumerate(kb):
                                res[n][f][k] = fo[f][j]
            except Exception as e:  # noqa: BLE001
                if is_oom(e):
                    empty_cache()
                    if budget.shrink():
                        continue
                raise
            a += len(kb)
    return {n: {f: np.stack(v) for f, v in d.items()} for n, d in res.items()}


def _run_chunk(model, rows, pidx, kb, chunk, conds, ptr, layer, T, tB, R, device, hooked):
    b = len(kb)
    X = None if hooked else R.repeat(len(chunk), 1, 1)
    writes = []                                      # (batch row, start pos, rows tensor)
    for j, n in enumerate(chunk):
        V = conds[n]
        if V is None:
            continue
        for q, k in enumerate(kb):
            fd = rows[pidx[q]]["first_diff"]
            writes.append((j * b + q, fd, V[int(ptr[k]):int(ptr[k + 1])]))
    tokens = tB.repeat(len(chunk), 1)
    if hooked:
        def hook(act, hook):
            act = act.clone()
            for r, fd, v in writes:
                act[r, fd:T] = v.to(act.device, act.dtype)
            return act
        lg = runtime.last_logits(model, tokens, [(hook_name(layer), hook)])
    else:
        for r, fd, v in writes:
            X[r, fd:T] = v.to(X.device, X.dtype)
        lg = runtime.last_logits_from(model, X, layer, tokens)
    return [lg[j * b:(j + 1) * b] for j in range(len(chunk))]


def stage1_generic(model, spec: ModelSpec, ids: Sequence[Sequence[int]], corpus_hash: str, layers: Sequence[int],
                   skip: int, cache: Cache, budget: Budget, device, shard_docs: int = 16) -> Dict[int, Entry]:
    ents = [cache.entry("generic_resid", model=spec.key(), corpus=corpus_hash, layer=int(L), skip=skip, site=spec.site)
            for L in layers]
    with lock_many(ents):
        return _stage1_generic_fill(model, spec, ids, corpus_hash, layers, skip, cache, budget, device, shard_docs)


@torch.inference_mode()
def _stage1_generic_fill(model, spec: ModelSpec, ids: Sequence[Sequence[int]], corpus_hash: str, layers: Sequence[int],
                         skip: int, cache: Cache, budget: Budget, device, shard_docs: int = 16) -> Dict[int, Entry]:
    """Residuals of generic text at positions >= skip (pilot: BOS / attention-sink positions excluded) per layer."""
    ents = {L: cache.entry("generic_resid", model=spec.key(), corpus=corpus_hash, layer=int(L), skip=skip, site=spec.site).open()
            for L in layers}
    T = len(ids[0])
    for si, a in enumerate(range(0, len(ids), shard_docs)):
        need = [L for L, e in ents.items() if not e.shard_done(si)]
        if not need:
            continue
        docs = ids[a:a + shard_docs]
        got = {L: [] for L in need}
        b0 = 0
        while b0 < len(docs):
            n = budget.rows_for(T)
            t = torch.tensor(docs[b0:b0 + n], dtype=torch.long, device=device)
            part: Dict[int, torch.Tensor] = {}                 # merged only after the forward succeeded
            try:
                hooks = [(hook_name(L), (lambda act, hook, L=L: part.__setitem__(L, act[:, skip:].reshape(-1, act.shape[-1]).detach().cpu())))
                         for L in need]
                runtime.final_resid(model, t, hooks)
            except Exception as e:  # noqa: BLE001
                if is_oom(e):
                    empty_cache()
                    if budget.shrink():
                        continue
                raise
            for L in need:
                got[L].append(part[L])
            b0 += n
        for L in need:
            ents[L].write_shard(si, {"x": torch.cat(got[L])})
    for e in ents.values():
        if not e.complete:
            e.mark_complete({"n_docs": len(ids), "seq_len": T, "skip": skip})
    return ents


def stage2_generic(cache: Cache, ent: Entry, sae, sae_key: dict, device, block_latents: Optional[int] = None) -> Entry:
    with cache.entry("generic_sae", resid=ent.key, sae=sae_key).writer_lock():
        return _stage2_generic_fill(cache, ent, sae, sae_key, device, block_latents)


@torch.inference_mode()
def _stage2_generic_fill(cache: Cache, ent: Entry, sae, sae_key: dict, device, block_latents: Optional[int] = None) -> Entry:
    from harness.sae import decompose

    ent.require_complete()
    e = cache.entry("generic_sae", resid=ent.key, sae=sae_key).open()
    for si in ent.shards():
        if e.shard_done(si):
            continue
        x = ent.read_shard(si)["x"]
        xh, eps, l0 = decompose(sae, x, block_latents=block_latents, device=device)
        e.write_shard(si, {"eps": eps.cpu(), "l0": l0.cpu(), "x_norm": x.float().norm(dim=-1)})
    if not e.complete:
        e.mark_complete({"sae": sae_key})
    return e
