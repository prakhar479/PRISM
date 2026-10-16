"""Shared machinery: residual caching, patched forward passes, NLDR (+bootstrap CIs), interventions,
resumable work units and the shared RunContext.

Patch positions
    "last"      : overwrite resid_post[layer][:, -1] only.
    "from_diff" : overwrite every position from the first token where clean and corrupt differ to the end.
                  Positions before that are identical in both prompts, so patching the full clean residual
                  here recovers NLDR = 1 at *every* layer (the state is not hidden at earlier positions).
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.data.synthetic_tasks import TokenizedPair, batch_by_length, build_task_splits, make_tensors
from src.models.sae_loader import hook_name
from src.utils import free_memory, model_device

EPS = 1e-8
POSITIONS = ("last", "from_diff")
log = logging.getLogger("prism")


@dataclass
class Baseline:
    """Unpatched clean/corrupt runs for one (task, split). Activations live on CPU, aligned by row.

    resid_*[layer]  : [N, d] float32 residual at the last position.
    span_*[layer]   : [M, d] residual (model dtype) at positions diff_pos..L-1 of every item, flattened;
                      rows of item i are row_ptr[i]:row_ptr[i+1] and its last row is the last position.
    """
    items: List[TokenizedPair]
    spans: List[Tuple[int, int]]              # contiguous equal-length batches over `items`
    dl_clean: torch.Tensor                    # [N] logit diff on the clean prompt
    dl_corrupt: torch.Tensor                  # [N] logit diff on the corrupt prompt
    top1_clean_run: torch.Tensor              # [N] bool: argmax(clean run) == clean answer
    top1_corrupt_run: torch.Tensor            # [N] bool: argmax(corrupt run) == clean answer
    resid_clean: Dict[int, torch.Tensor]
    resid_corrupt: Dict[int, torch.Tensor]
    clean_ans: torch.Tensor
    corrupt_ans: torch.Tensor
    states_clean: np.ndarray
    states_corrupt: np.ndarray
    row_ptr: np.ndarray = None                # [N+1]
    row_item: torch.Tensor = None             # [M] item index of each span row
    row_pos: torch.Tensor = None              # [M] token position of each span row
    span_clean: Dict[int, torch.Tensor] = field(default_factory=dict)
    span_corrupt: Dict[int, torch.Tensor] = field(default_factory=dict)
    _boot: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.items)

    def acts(self, layer: int, kind: str, positions: str = "last") -> torch.Tensor:
        """float32 activations for a patch mode: [N, d] ('last') or [M, d] ('from_diff')."""
        if positions == "last":
            return (self.resid_clean if kind == "clean" else self.resid_corrupt)[layer]
        store = self.span_clean if kind == "clean" else self.span_corrupt
        if layer not in store:
            raise KeyError(f"span activations for layer {layer} were not cached (cache_spans=False?)")
        return store[layer].float()

    def boot_idx(self, n_boot: int, seed: int = 0) -> np.ndarray:
        """Fixed resampling indices shared by every intervention (paired bootstrap)."""
        if self._boot is None or self._boot.shape[0] != n_boot:
            self._boot = np.random.default_rng(seed).integers(0, len(self), size=(n_boot, len(self)))
        return self._boot


def _logit_diff(logits_last: torch.Tensor, ca: torch.Tensor, ka: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(len(ca), device=logits_last.device)
    return logits_last[idx, ca] - logits_last[idx, ka]


@torch.inference_mode()
def cache_baseline(model, items: Sequence[TokenizedPair], layers: Sequence[int], batch_size: int = 32,
                   device=None, cache_spans: bool = True) -> Baseline:
    device = device if device is not None else model_device(model)
    batches = batch_by_length(list(items), batch_size)
    ordered = [it for b in batches for it in b]
    spans, s = [], 0
    for b in batches:
        spans.append((s, s + len(b)))
        s += len(b)
    lens = [len(it.clean_ids) - it.diff_pos for it in ordered]
    row_ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    row_item = torch.repeat_interleave(torch.arange(len(ordered)), torch.tensor(lens, dtype=torch.long))
    row_pos = torch.cat([torch.arange(it.diff_pos, len(it.clean_ids)) for it in ordered]) if ordered else torch.zeros(0, dtype=torch.long)
    last = {k: {l: [] for l in layers} for k in ("clean", "corrupt")}
    span = {k: {l: [] for l in layers} for k in ("clean", "corrupt")}
    dl = {"clean": [], "corrupt": []}
    top1 = {"clean": [], "corrupt": []}
    for (s0, s1), b in zip(spans, batches):
        t = make_tensors(b, device)
        ca, ka = t["clean_ans"], t["corrupt_ans"]
        bi = (row_item[row_ptr[s0]:row_ptr[s1]] - s0).to(device)
        pi = row_pos[row_ptr[s0]:row_ptr[s1]].to(device)
        for kind in ("clean", "corrupt"):
            hooks = []
            for l in layers:
                def make(l=l, kind=kind):
                    def fn(act, hook):
                        last[kind][l].append(act[:, -1].detach().float().cpu())
                        if cache_spans:
                            span[kind][l].append(act[bi, pi].detach().cpu())
                        return None
                    return fn
                hooks.append((hook_name(l), make()))
            logits = model.run_with_hooks(t[kind], fwd_hooks=hooks)[:, -1].float()
            dl[kind].append(_logit_diff(logits, ca, ka).cpu())
            top1[kind].append((logits.argmax(-1) == ca).cpu())
    cat = lambda xs: torch.cat(xs)
    return Baseline(
        items=ordered, spans=spans, dl_clean=cat(dl["clean"]), dl_corrupt=cat(dl["corrupt"]),
        top1_clean_run=cat(top1["clean"]), top1_corrupt_run=cat(top1["corrupt"]),
        resid_clean={l: cat(v) for l, v in last["clean"].items()},
        resid_corrupt={l: cat(v) for l, v in last["corrupt"].items()},
        clean_ans=torch.tensor([i.clean_ans for i in ordered]), corrupt_ans=torch.tensor([i.corrupt_ans for i in ordered]),
        states_clean=np.array([i.pair.clean_state_id for i in ordered]),
        states_corrupt=np.array([i.pair.corrupt_state_id for i in ordered]),
        row_ptr=row_ptr, row_item=row_item, row_pos=row_pos,
        span_clean={l: cat(v) for l, v in span["clean"].items()} if cache_spans else {},
        span_corrupt={l: cat(v) for l, v in span["corrupt"].items()} if cache_spans else {},
    )


@torch.inference_mode()
def run_patched(model, base: Baseline, layer: int, vecs: torch.Tensor, device=None,
                positions: str = "last") -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the corrupt prompts with resid_post[layer] overwritten by `vecs` at the chosen positions.

    vecs: [N, d] for positions='last', [M, d] (span rows) for positions='from_diff'.
    Returns (logit diff [N], top-1 == clean answer [N]).
    """
    assert positions in POSITIONS, positions
    device = device if device is not None else model_device(model)
    dls, tops = [], []
    for (s, e) in base.spans:
        t = make_tensors(base.items[s:e], device)
        if positions == "last":
            v = vecs[s:e]

            def fn(act, hook, v=v):
                act = act.clone()
                act[:, -1, :] = v.to(device=act.device, dtype=act.dtype)
                return act
        else:
            r0, r1 = int(base.row_ptr[s]), int(base.row_ptr[e])
            v = vecs[r0:r1]
            bi = (base.row_item[r0:r1] - s).to(device)
            pi = base.row_pos[r0:r1].to(device)

            def fn(act, hook, v=v, bi=bi, pi=pi):
                act = act.clone()
                act[bi, pi] = v.to(device=act.device, dtype=act.dtype)
                return act

        logits = model.run_with_hooks(t["corrupt"], fwd_hooks=[(hook_name(layer), fn)])[:, -1].float()
        dls.append(_logit_diff(logits, t["clean_ans"], t["corrupt_ans"]).cpu())
        tops.append((logits.argmax(-1) == t["clean_ans"]).cpu())
    return torch.cat(dls), torch.cat(tops)


def summarize(dl_patched: torch.Tensor, top1: torch.Tensor, base: Baseline, n_boot: int = 1000) -> Dict[str, float]:
    """NLDR (ratio of means = primary; per-pair mean also reported), paired-bootstrap 95% CI, top-1 recovery."""
    dc, dk = base.dl_clean.double(), base.dl_corrupt.double()
    p = dl_patched.double()
    nldr = (p.mean() - dk.mean()) / (dc.mean() - dk.mean() + EPS)
    pair = ((p - dk) / (dc - dk + EPS)).mean()
    a_c, a_k, a_p = base.top1_clean_run.float().mean(), base.top1_corrupt_run.float().mean(), top1.float().mean()
    out = {"nldr": float(nldr), "nldr_pair_mean": float(pair), "nldr_pair_median": float(((p - dk) / (dc - dk + EPS)).median()),
           "dl_patched": float(p.mean()), "top1_rate": float(a_p),
           "top1_recovery": float((a_p - a_k) / (a_c - a_k)) if abs(float(a_c - a_k)) > 1e-6 else float("nan")}
    if n_boot and len(base) > 1:
        idx = base.boot_idx(n_boot)
        pn, cn, kn = p.numpy()[idx].mean(1), dc.numpy()[idx].mean(1), dk.numpy()[idx].mean(1)
        bs = (pn - kn) / (cn - kn + EPS)
        out["nldr_lo"], out["nldr_hi"] = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
    return out


@torch.inference_mode()
def decompose_batched(sae, X: torch.Tensor, chunk: int = 256) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Decompose [N, d] residuals in chunks -> (x_hat [N,d], eps [N,d]) on CPU float32 and mean L0."""
    xs, es, l0, n = [], [], 0.0, 0
    for i in range(0, len(X), chunk):
        xh, e, acts = sae.decompose(X[i:i + chunk].float())
        xs.append(xh.cpu())
        es.append(e.cpu())
        l0 += float((acts > 0).sum(-1).float().sum())
        n += len(xh)
    if not xs:
        d = X.shape[-1]
        return torch.zeros(0, d), torch.zeros(0, d), 0.0
    return torch.cat(xs), torch.cat(es), l0 / max(n, 1)


@torch.inference_mode()
def latent_features(sae, Xtr: torch.Tensor, Xte: torch.Tensor, max_cols: int = 4096,
                    chunk: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """SAE latent activations restricted to the `max_cols` latents most often active on train (memory-safe for 1m)."""
    counts = None
    for i in range(0, len(Xtr), chunk):
        a = sae.encode(Xtr[i:i + chunk].float().to(sae.device, sae.dtype))
        c = (a > 0).sum(0)
        counts = c if counts is None else counts + c
    cols = torch.argsort(counts, descending=True)[:max_cols]
    cols = cols[counts[cols] > 0]
    if len(cols) == 0:
        cols = torch.arange(1, device=counts.device)

    def gather(X: torch.Tensor) -> np.ndarray:
        out = [sae.encode(X[i:i + chunk].float().to(sae.device, sae.dtype))[:, cols].float().cpu()
               for i in range(0, len(X), chunk)]
        return torch.cat(out).numpy()

    return gather(Xtr), gather(Xte)


def rescaled_noise(eps_ref: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Gaussian noise rescaled per token so ||eta|| == ||eps_ref||."""
    eta = torch.randn(eps_ref.shape, generator=gen, dtype=eps_ref.dtype)
    return eta * (eps_ref.norm(dim=-1, keepdim=True) / (eta.norm(dim=-1, keepdim=True) + EPS))


def task_cfg(cfg: dict, task: str) -> dict:
    """Global config merged with cfg['task_overrides'][task] (e.g. written by scripts/preflight.py)."""
    return {**cfg, **(cfg.get("task_overrides") or {}).get(task, {})}


def prepare_task(task: str, model, tok, cfg: dict, mock: bool, layers: Sequence[int], device=None,
                 n_train: int = None, n_test: int = None, cache_spans: bool = True) -> Dict[str, Baseline]:
    """Build prompt-disjoint train/test splits and cache baselines at `layers` ({} if too few valid pairs)."""
    c = task_cfg(cfg, task)
    n_train = n_train if n_train is not None else c["n_train"]
    n_test = n_test if n_test is not None else c["n_test"]
    splits = build_task_splits(task, tok, n_train, n_test, c["seed"], model=model, mock=mock,
                               candidate_multiplier=c.get("candidate_multiplier", 3),
                               min_logit_diff=c.get("min_logit_diff", 2.0), n_shots=c.get("n_shots", 0),
                               batch_size=c["batch_size"], answer_format=c.get("answer_format", "auto"),
                               filter_mode=c.get("filter_mode", "strict"))
    out = {}
    for split, items in splits.items():
        if len(items) < c.get("min_pairs", 20):
            log.warning("%s/%s: only %d valid pairs (< min_pairs) -> skipped", task, split, len(items))
            return {}
        out[split] = cache_baseline(model, items, layers, c["batch_size"], device, cache_spans=cache_spans)
    return out


# --------------------------------------------------------------------------- resumable work units
class UnitLog:
    """Append-only JSONL of finished work units: one line {"key": ..., "rows": [...]} per unit.

    With resume=True, finished keys are skipped on restart; otherwise the log is truncated.
    """

    def __init__(self, path: Path, resume: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.units: Dict[str, List[dict]] = {}
        if resume and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    try:
                        u = json.loads(line)
                    except json.JSONDecodeError:   # torn last line after a crash
                        continue
                    self.units[u["key"]] = u["rows"]
        else:
            self.path.write_text("")

    @staticmethod
    def key(*parts) -> str:
        return "|".join(str(p) for p in parts)

    def done(self, key: str) -> bool:
        return key in self.units

    def add(self, key: str, rows: List[dict]) -> None:
        from src.utils import _jsonable

        self.units[key] = rows
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": key, "rows": rows}, default=_jsonable) + "\n")

    def rows(self) -> List[dict]:
        return [r for rs in self.units.values() for r in rs]


@dataclass
class RunContext:
    """Everything the experiments share: model, tokenizer, config, cached per-task baselines, run control."""
    model: object
    tok: object
    cfg: dict
    mock: bool
    device: object
    out_dir: Path
    data: Dict[str, Dict[str, Baseline]] = field(default_factory=dict)
    cached_layers: Tuple[int, ...] = ()
    resume: bool = False
    deadline: Optional[float] = None          # unix time after which no new unit starts
    task_info: Dict[str, dict] = field(default_factory=dict)
    generic: Dict[int, torch.Tensor] = field(default_factory=dict)
    generic_batches: Optional[List[torch.Tensor]] = None

    def get_generic_batches(self) -> List[torch.Tensor]:
        """Generic-text token batches, loaded once per run (download/streaming is slow)."""
        if self.generic_batches is None:
            self.generic_batches = generic_token_batches(self.tok, self.cfg, self.mock)
        return self.generic_batches

    @property
    def cache_spans(self) -> bool:
        return any(p == "from_diff" for k in ("exp1_positions", "exp3_positions") for p in self.cfg.get(k, ["last"]))

    def prepare(self, layers: Sequence[int]) -> None:
        layers = tuple(sorted(set(layers)))
        if self.data and set(layers) <= set(self.cached_layers):
            return
        self.data = {}
        for task in self.cfg["tasks"]:
            t0 = time.time()
            d = prepare_task(task, self.model, self.tok, self.cfg, self.mock, layers, self.device,
                             cache_spans=self.cache_spans)
            info = {"n_train": len(d["train"]) if d else 0, "n_test": len(d["test"]) if d else 0,
                    "skipped": not d, "seconds": round(time.time() - t0, 1),
                    **{k: task_cfg(self.cfg, task).get(k) for k in ("n_shots", "answer_format", "filter_mode")}}
            if d:
                b = d["test"]
                info.update(dl_clean=float(b.dl_clean.mean()), dl_corrupt=float(b.dl_corrupt.mean()),
                            clean_top1=float(b.top1_clean_run.float().mean()), mean_len=float(np.mean([len(i.clean_ids) for i in b.items])),
                            mean_span=float(np.diff(b.row_ptr).mean()))
                self.data[task] = d
            self.task_info[task] = info
        self.cached_layers = layers
        log.info("prepared tasks %s at layers %s", list(self.data), list(layers))

    @property
    def d_model(self) -> int:
        return self.model.cfg.d_model

    def unit_log(self, exp: str) -> UnitLog:
        return UnitLog(self.out_dir / "metrics" / f"{exp}_units.jsonl", resume=self.resume)

    def out_of_time(self) -> bool:
        if self.deadline is not None and time.time() > self.deadline:
            log.warning("deadline reached: skipping remaining units")
            return True
        return False

    def run_unit(self, exp: str, key: str, fn: Callable[[], List[dict]], units: UnitLog) -> bool:
        """Run one work unit, record its rows; on failure log + record and continue. Returns success."""
        if units.done(key):
            return True
        if self.out_of_time():
            return False
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            rows = fn()
        except Exception as e:  # noqa: BLE001 - one failed unit must not kill an overnight run
            log.error("unit %s/%s FAILED: %s: %s", exp, key, type(e).__name__, e)
            self.record_failure(exp, key, e)
            free_memory()
            return False
        units.add(key, rows)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
        log.info("unit %s/%s done in %.1fs (%d rows, peak VRAM %.1f GiB)", exp, key, time.time() - t0, len(rows), peak)
        return True

    def record_failure(self, exp: str, key: str, e: BaseException) -> None:
        p = self.out_dir / "metrics" / "failures.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps({"exp": exp, "key": key, "error": f"{type(e).__name__}: {e}",
                                "traceback": traceback.format_exc(), "time": time.time()}) + "\n")


# --------------------------------------------------------------------------- generic text corpus
def generic_token_batches(tok, cfg: dict, mock: bool) -> List[torch.Tensor]:
    """Token batches of generic text (for fitting a task-agnostic error subspace). [] if unavailable."""
    g = cfg.get("generic_text") or {}
    n_docs, seq = int(g.get("n_docs", 200)), int(g.get("seq_len", 128))
    if mock:
        rng = np.random.default_rng(cfg["seed"] + 5)
        ids = rng.integers(1, tok.vocab_size, size=(n_docs, seq))
        ids[:, 0] = tok.BOS
        return list(torch.tensor(ids).split(16))
    texts = _load_generic_texts(g.get("dataset", "NeelNanda/pile-10k"), n_docs * 3)
    rows = []
    for t in texts:
        ids = tok.encode(t)
        if len(ids) >= seq:
            rows.append(ids[:seq])
        if len(rows) >= n_docs:
            break
    if not rows:
        return []
    return list(torch.tensor(rows).split(16))


def _load_generic_texts(dataset: str, n: int) -> List[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train", streaming=True)
        return [r["text"] for _, r in zip(range(n), ds)]
    except Exception as e:  # noqa: BLE001
        log.warning("datasets.load_dataset(%s) failed (%s); trying parquet via huggingface_hub", dataset, e)
    try:
        import pandas as pd
        from huggingface_hub import HfApi, hf_hub_download

        files = [f for f in HfApi().list_repo_files(dataset, repo_type="dataset") if f.endswith(".parquet")]
        df = pd.read_parquet(hf_hub_download(dataset, files[0], repo_type="dataset"))
        return df["text"].tolist()[:n]
    except Exception as e:  # noqa: BLE001
        log.warning("generic text corpus unavailable (%s): generic_text_svd will be skipped", e)
        return []


@torch.inference_mode()
def generic_resid(ctx: RunContext, layer: int) -> Optional[torch.Tensor]:
    """Residuals of generic text at `layer` (positions >= skip, BOS/attention-sink tokens excluded), cached."""
    if layer in ctx.generic:
        return ctx.generic[layer]
    g = ctx.cfg.get("generic_text") or {}
    skip = int(g.get("skip_positions", 4))
    batches = ctx.get_generic_batches()
    if not batches:
        ctx.generic[layer] = None
        return None
    store: List[torch.Tensor] = []

    def fn(act, hook):
        store.append(act[:, skip:].reshape(-1, act.shape[-1]).detach().cpu())
        return None

    for b in batches:
        ctx.model.run_with_hooks(b.to(ctx.device), fwd_hooks=[(hook_name(layer), fn)])
    ctx.generic[layer] = torch.cat(store)
    log.info("generic text: %d residual rows at layer %d", len(ctx.generic[layer]), layer)
    return ctx.generic[layer]


@torch.inference_mode()
def sae_fidelity(ctx: RunContext, sae, layer: int) -> Optional[dict]:
    """Standard SAE quality on generic text at `layer`: next-token CE clean / with x_hat spliced in / with the
    residual mean-ablated, and CE-loss-recovered = (CE_ablate - CE_splice) / (CE_ablate - CE_clean).
    Positions < skip_positions (BOS / attention sinks) are left untouched and excluded from the loss."""
    g = ctx.cfg.get("generic_text") or {}
    skip = int(g.get("skip_positions", 4))
    batches = ctx.get_generic_batches()[: int(g.get("fidelity_batches", 4))]
    if not batches:
        return None
    G = generic_resid(ctx, layer)
    mean = G.float().mean(0) if G is not None else None
    tot = {"clean": 0.0, "splice": 0.0, "ablate": 0.0}
    n = 0
    for b in batches:
        b = b.to(ctx.device)

        def ce(logits):
            lp = torch.log_softmax(logits[:, skip:-1].float(), -1)
            return -lp.gather(-1, b[:, skip + 1:, None]).squeeze(-1).sum().item()

        def splice(act, hook):
            act = act.clone()
            x = act[:, skip:]
            xh, _, _ = decompose_batched(sae, x.reshape(-1, x.shape[-1]))      # chunked: 1m SAEs are huge
            act[:, skip:] = xh.reshape(x.shape).to(act.device, act.dtype)
            return act

        def ablate(act, hook):
            act = act.clone()
            act[:, skip:] = mean.to(act.device, act.dtype)
            return act

        tot["clean"] += ce(ctx.model.run_with_hooks(b, fwd_hooks=[]))
        tot["splice"] += ce(ctx.model.run_with_hooks(b, fwd_hooks=[(hook_name(layer), splice)]))
        tot["ablate"] += ce(ctx.model.run_with_hooks(b, fwd_hooks=[(hook_name(layer), ablate)]))
        n += b.shape[0] * (b.shape[1] - skip - 1)
    c, s_, a = (tot[k] / n for k in ("clean", "splice", "ablate"))
    return {"ce_clean": c, "ce_splice": s_, "ce_ablate": a, "ce_delta": s_ - c,
            "ce_recovered": (a - s_) / (a - c) if abs(a - c) > 1e-9 else float("nan"), "n_tokens": n}
