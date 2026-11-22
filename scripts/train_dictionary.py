#!/usr/bin/env python
"""Train the breadth pilot's own dictionaries (docs/PILOT_BREADTH_PLAN.md) on the generic fit set; one GPU, one dictionary.

    python scripts/train_dictionary.py --mode klft --layer 18 --width 16k --name klft_L18_16k          # P4 (b)
    python scripts/train_dictionary.py --mode frozen_decoder --layer 12 --width 16k --k 82 --name frozendec_L12_16k   # control
    python scripts/train_dictionary.py --tiny --mode klft --layer 1 --name t_klft --steps 3                          # tests

klft            starts from the Gemma Scope SAE of (layer, width, canonical L0 unless --l0) and fine-tunes W_enc, b_enc, W_dec,
                b_dec (JumpReLU thresholds fixed) with the KL+MSE recipe of arXiv 2503.17272 (github.com/adamkarvonen/
                sae_kl_finetune): loss = 0.5 (MSE + alpha KL), alpha = (MSE / KL).detach() per batch, MSE = mean squared error,
                KL(p_clean || p_splice) taken at `kl_positions` random positions >= skip per window (memory: logits for a few
                positions only); AdamW, lr 5e-5 decayed linearly to 0, gradient clipping 1.0; decoder rows kept unit norm by an
                exact rescaling (the recipe projects out the parallel gradient instead). Deviation, stated in the runbook: ~0.7-2M
                tokens (1-3 epochs of the fit set) instead of the recipe's 15M.
frozen_decoder  the 'Sanity checks for SAEs' control (Korznikov et al.): a TopK dictionary whose decoder rows are random unit
                vectors (fixed) and whose b_dec is the mean residual (fixed); only the encoder is trained, on MSE.
Data: data/generic/pile10k_6000x128_skip100.json (disjoint from the 100-document evaluation set). Output:
<cache_root>/trained_dicts/<name>/{params.npz, info.json, complete.json}; configs use it as sae: {dict: "trained:<name>"}.
A finished dictionary is not retrained (delete the directory to redo it). The held-out CE on pile10k_100x128 before and
after training is recorded in info.json.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
from harness import provenance
from harness.profile import REPO_ROOT, load_profile

log = logging.getLogger("harness")
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"


def load_ids(rel: str, n: int = None):
    d = json.loads((REPO_ROOT / "data" / rel).read_text())
    ids = d["ids"][:n] if n else d["ids"]
    return ids, d


def eval_ce(model, runtime, sae, layer, ids, skip, device, batch=8):
    """Mean next-token CE (positions >= skip) clean and with x -> x_hat spliced at `layer`."""
    import torch
    from harness.model import hook_name
    from harness.sae import decompose
    tot_c = tot_s = n = 0.0
    for a in range(0, len(ids), batch):
        t = torch.tensor(ids[a:a + batch], device=device)

        def splice(act, hook):
            act = act.clone()
            xh, _, _ = decompose(sae, act[:, skip:].reshape(-1, act.shape[-1]), device=device)
            act[:, skip:] = xh.reshape(act[:, skip:].shape).to(act.dtype)
            return act
        with torch.inference_mode():
            tot_c += runtime.token_nll_sum(model, t, [], skip)
            tot_s += runtime.token_nll_sum(model, t, [(hook_name(layer), splice)], skip)
        n += t.shape[0] * (t.shape[1] - 1 - skip)
    return {"ce_clean": tot_c / n, "ce_splice": tot_s / n, "ce_delta": (tot_s - tot_c) / n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["klft", "frozen_decoder"])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--width", default="16k")
    ap.add_argument("--l0", type=int, default=None, help="klft: Gemma Scope variant (default canonical)")
    ap.add_argument("--k", type=int, default=None, help="frozen_decoder: active latents per token (default: the canonical L0)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--profile", default=os.environ.get("HL_PROFILE", "server"))
    ap.add_argument("--fit-set", default="generic/pile10k_6000x128_skip100.json")
    ap.add_argument("--eval-set", default="generic/pile10k_100x128.json")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=None, help="cap on optimizer steps (tests)")
    ap.add_argument("--batch", type=int, default=4, help="windows per step")
    ap.add_argument("--lr", type=float, default=None, help="default: 5e-5 (klft), 1e-3 (frozen_decoder)")
    ap.add_argument("--mse-coef", type=float, default=1.0, help="klft: multiplies the MSE term (recipe: 1)")
    ap.add_argument("--kl-positions", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    prof = load_profile("dev" if a.tiny else a.profile)
    prof.apply_env()
    import torch
    from harness import dicts as Dct
    from harness.device import resolve_device, set_threads
    from harness.model import ModelSpec, build_model
    from src.models import runtime

    root = (REPO_ROOT / "tests" / "_tiny_cache") if a.tiny else prof.root("cache_root")
    out = Dct.trained_dir(root, a.name)
    if (out / "complete.json").exists():
        log.info("%s already trained (%s); nothing to do", a.name, out)
        return 0
    git = provenance.git_state()
    if git["dirty"] and not (a.allow_dirty or a.tiny):
        raise SystemExit("working tree is dirty: commit, or pass --allow-dirty")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu") if a.tiny else resolve_device(prof.device)
    set_threads(None)
    torch.manual_seed(a.seed)
    spec = ModelSpec(name="tiny", dtype="float32", tiny_vocab=512, tiny_layers=a.layer + 2) if a.tiny else \
        ModelSpec(revision=MODEL_REVISION, dtype=prof.model_dtype)
    model = build_model(spec, device, offload_embed=False if a.tiny else prof.offload_embed)
    fit_ids, fit_meta = load_ids(a.fit_set)
    ev_ids, ev_meta = load_ids(a.eval_set)
    if a.tiny:
        fit_ids = [[min(t, 511) for t in r] for r in fit_ids[:16]]
        ev_ids = [[min(t, 511) for t in r] for r in ev_ids[:4]]
    skip = int(fit_meta["skip_positions"])
    d_model = model.cfg.d_model

    # ---- initial dictionary
    if a.mode == "klft":
        if a.tiny:
            sae = Dct.random_dict(d_model, 128, "jumprelu", seed=a.layer, layer=a.layer, width=a.width)
        else:
            from harness.sae import SAEId, canonical_l0
            inv = json.loads((REPO_ROOT / "configs" / "sae_inventory.json").read_text())
            l0 = a.l0 or canonical_l0(inv["inventory"][str(a.layer)][a.width])
            sid = SAEId(a.layer, a.width, l0, revision=inv["revision"])
            sae = _gemma_scope_as_dict(sid)
        lr = a.lr or 5e-5
    else:
        n = 128 if a.tiny else __import__("harness.sae", fromlist=["parse_width"]).parse_width(a.width)
        k = a.k or (8 if a.tiny else _canonical_l0(a.layer, a.width))
        g = torch.Generator().manual_seed(a.seed + 99)
        W_dec = torch.randn(n, d_model, generator=g)
        W_dec /= W_dec.norm(dim=1, keepdim=True)
        mean = _mean_resid(model, runtime, fit_ids[:256], a.layer, skip, device)
        W_enc_T = (W_dec.clone() * 0.1)
        sae = Dct.Dictionary(W_enc_T, torch.zeros(n), torch.zeros(n), W_dec, mean, a.width, a.layer, kind="topk", k=k)
        lr = a.lr or 1e-3
    sae.to_device(device)                    # FIRST: .to() returns new tensors, so the parameter list is taken after the move
    params = [sae.W_enc_T, sae.b_enc, sae.W_dec, sae.b_dec] if a.mode == "klft" else [sae.W_enc_T, sae.b_enc]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr) if a.mode == "klft" else torch.optim.Adam(params, lr=lr)
    before = eval_ce(model, runtime, sae, a.layer, ev_ids, skip, device)
    log.info("%s before: %s", a.name, before)

    # ---- training
    rng = np.random.default_rng(a.seed)
    steps_per_epoch = max(1, len(fit_ids) // a.batch)
    total = int(round(a.epochs * steps_per_epoch))
    if a.steps:
        total = min(total, a.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: max(0.0, 1.0 - st / max(1, total))) if a.mode == "klft" else None
    losses, t0 = [], time.time()
    from harness.model import hook_name
    for step in range(total):
        if step % steps_per_epoch == 0:
            order = rng.permutation(len(fit_ids))
        sel = order[(step % steps_per_epoch) * a.batch:(step % steps_per_epoch + 1) * a.batch]
        t = torch.tensor([fit_ids[i] for i in sel], device=device)
        with torch.no_grad():
            x = runtime.resid_at(model, t, a.layer)
        xs = x[:, skip:].float().reshape(-1, d_model)
        xh = _decode_train(sae, xs)
        fvu = ((xs - xh) ** 2).sum(-1).mean() / ((xs - xs.mean(0)) ** 2).sum(-1).mean().clamp_min(1e-9)
        mse = torch.nn.functional.mse_loss(xh, xs) if a.mode == "klft" else fvu
        if a.mode == "klft":
            P = torch.as_tensor(np.sort(rng.choice(np.arange(skip, t.shape[1] - 1), size=min(a.kl_positions, t.shape[1] - 1 - skip),
                                                    replace=False)), device=device)
            with torch.no_grad():
                Rc = runtime.final_resid(model, t)
                lp_c = torch.log_softmax(runtime.head(model, Rc[:, P]), -1)
            xx = x.clone()
            xx[:, skip:] = xh.reshape(x[:, skip:].shape).to(x.dtype)
            Rs = model.run_with_hooks(xx, fwd_hooks=[], start_at_layer=a.layer + 1, stop_at_layer=model.cfg.n_layers, tokens=t)
            lp_s = torch.log_softmax(runtime.head(model, Rs[:, P]), -1)
            kl = (lp_c.exp() * (lp_c - lp_s)).sum(-1).mean()
            alpha = (mse / (kl + 1e-8)).detach()
            loss = 0.5 * (a.mse_coef * mse + alpha * kl)
        else:
            kl = torch.zeros(())
            loss = mse
        opt.zero_grad()
        loss.backward()
        if a.mode == "klft":
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        if a.mode == "klft":
            with torch.no_grad():         # unit-norm decoder rows, rescaling the latent exactly (z, b_enc, threshold x n): x_hat unchanged
                nrm = sae.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
                sae.W_dec.div_(nrm)
                sae.W_enc_T.mul_(nrm)
                sae.b_enc.mul_(nrm[:, 0])
                sae.threshold.mul_(nrm[:, 0].to(sae.threshold.dtype))
        losses.append({"step": step, "loss": float(loss), "kl": float(kl), "mse": float(mse), "fvu": float(fvu)})
        if step % 50 == 0 or step == total - 1:
            log.info("%s step %d/%d loss %.4f kl %.4f mse %.4f (%.0fs)", a.name, step + 1, total, float(loss), float(kl),
                     float(mse), time.time() - t0)
    for p in params:
        p.requires_grad_(False)
    after = eval_ce(model, runtime, sae, a.layer, ev_ids, skip, device)
    log.info("%s after: %s", a.name, after)
    Dct.save_npz_dict(out / "params.npz", sae)
    info = {"name": a.name, "mode": a.mode, "layer": a.layer, "width": a.width, "kind": sae.kind, "k": sae.k, "lr": lr,
            "epochs": a.epochs, "steps": total, "batch_windows": a.batch, "mse_coef": a.mse_coef, "kl_positions": a.kl_positions,
            "seed": a.seed, "fit_set": {"file": a.fit_set, "hash": fit_meta["hash"]}, "eval_set": {"file": a.eval_set, "hash": ev_meta["hash"]},
            "eval_before": before, "eval_after": after, "losses_tail": losses[-20:], "loss_every_50": losses[::50],
            "wall_s": round(time.time() - t0, 1), "git": git, "model": spec.key(), "tiny": a.tiny,
            "created": provenance.now()}
    (out / "info.json").write_text(json.dumps(info, indent=1, default=str))
    (out / "complete.json").write_text(json.dumps({"name": a.name, "created": provenance.now()}))
    # a small run record (no weights) under runs_root/PD/<name>, so bundle / publish ship the training outcome
    rec = ((REPO_ROOT / "tests" / "_tiny_runs") if a.tiny else prof.root("runs_root")) / "PD" / a.name
    rec.mkdir(parents=True, exist_ok=True)
    (rec / "info.json").write_text(json.dumps(info, indent=1, default=str))
    (rec / "manifest.json").write_text(json.dumps({"exp_id": "PD", "run_hash": a.name, "status": "complete", "smoke": False,
                                                   "tiny": a.tiny, "config": {"name": "trained_dictionary", "args": vars(a)},
                                                   "git": git, "created": provenance.now(), "packages": provenance.packages()},
                                                  indent=1, default=str))
    log.info("%s -> %s", a.name, out)
    return 0


def _decode_train(sae, xs):
    """Differentiable x_hat (JumpReLU with fixed thresholds, or TopK) for training."""
    import torch
    z = xs.to(sae.W_enc_T.dtype) @ sae.W_enc_T.T + sae.b_enc
    if sae.kind == "topk":
        post = torch.relu(z)
        v, i = torch.topk(post, sae.k, dim=-1)
        f = torch.zeros_like(post).scatter(-1, i, v)
    else:
        f = z * (z > sae.threshold).to(z.dtype)
    return f.float() @ sae.W_dec.float() + sae.b_dec.float()


def _canonical_l0(layer, width):
    from harness.sae import canonical_l0
    inv = json.loads((REPO_ROOT / "configs" / "sae_inventory.json").read_text())["inventory"]
    return canonical_l0(inv[str(layer)][width])


def _gemma_scope_as_dict(sid):
    """Gemma Scope params.npz -> a float32 Dictionary (JumpReLU) whose tensors can be trained."""
    import torch
    from harness import dicts as Dct
    from harness.sae import load_params, params_path
    s = load_params(params_path(sid), sid.layer, sid.width, "float32", sid.key)
    return Dct.Dictionary(s.W_enc_T.clone(), s.b_enc.clone(), s.threshold.clone(), s.W_dec.clone(), s.b_dec.clone().float(),
                          sid.width, sid.layer, sid.key, kind="jumprelu")


def _mean_resid(model, runtime, ids, layer, skip, device):
    import torch
    s, n = None, 0
    for a in range(0, len(ids), 8):
        t = torch.tensor(ids[a:a + 8], device=device)
        with torch.inference_mode():
            x = runtime.resid_at(model, t, layer)[:, skip:].float().reshape(-1, model.cfg.d_model)
        s = x.sum(0) if s is None else s + x.sum(0)
        n += len(x)
    return (s / max(1, n)).cpu()


if __name__ == "__main__":
    sys.exit(main())
