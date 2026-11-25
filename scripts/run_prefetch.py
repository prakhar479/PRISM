#!/usr/bin/env python
"""Download exactly what a release needs (model at its pinned revision, every SAE the queue's configs use), report
sizes and verify sha256 against the hub's LFS metadata. Run before the queue; afterwards set offline: true.

    python scripts/run_prefetch.py --queue queue/release1.yaml [--out preflight/prefetch.json] [--verify-only]
(The pilot's scripts/prefetch.py is unchanged and still used by run_server_gpu.sh.)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from harness.config import ExpConfig
from harness.profile import REPO_ROOT
from harness.queue import Queue
from harness.sae import SAEId

MODEL_REPO = "google/gemma-2-2b"
MODEL_PATTERNS = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "tokenizer.model", "model.safetensors.index.json")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def targets(queue: Queue) -> dict:
    """Model revisions per repo and every dictionary file the queue's configs (and its prefetch block) use. Registry dictionaries
    (configs/dict_registry.yaml) bring their own repo / files; trained dictionaries are produced by the queue (nothing to fetch);
    an add-on needs its base dictionary only."""
    from harness import dicts as Dct
    from harness.model import MODEL_REPOS
    inv = json.loads((REPO_ROOT / "configs" / "sae_inventory.json").read_text())
    reg = Dct.load_registry()
    model_revs, saes, models = set(), {}, set()
    for c in queue.configs():
        cfg = ExpConfig.load(REPO_ROOT / c)
        model_revs.add(cfg.model_revision)
        models.add((MODEL_REPOS[cfg.model], cfg.model_revision))
        for cell in cfg.cells:
            sid = Dct.resolve(cell.layer, cell.sae, inv["inventory"], inv["revision"], reg).base()
            if sid.repo_files():
                saes[sid.key] = sid
    for name in queue.prefetch.get("dicts", []):                  # registry dictionaries used by script jobs only
        e = reg[name]
        sid = Dct.DictId(int(e["layer"]), str(e["width"]), int(e.get("l0") or 0), repo=e["repo"], revision=e["revision"],
                         source="registry", name=name, arch=e["arch"], files=tuple(e["files"]))
        saes[sid.key] = sid
    for repo, rev in queue.prefetch.get("models", []):
        models.add((repo, rev))
    for layer, width, l0 in queue.prefetch.get("saes", []):      # SAEs that script jobs (E12) use without an experiment config
        sid = SAEId(int(layer), str(width), int(l0), revision=inv["revision"])
        saes[sid.key] = sid
    if queue.prefetch.get("model_revision"):
        model_revs.add(queue.prefetch["model_revision"])
        models.add((MODEL_REPO, queue.prefetch["model_revision"]))
    return {"model_revisions": sorted(r for r in model_revs if r), "models": sorted(m for m in models if m[1]),
            "saes": list(saes.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify-only", action="store_true", help="no downloads (local_files_only); check hashes")
    a = ap.parse_args()
    from harness.profile import load_profile
    load_profile().apply_env()          # HF_HOME of the active profile (HL_PROFILE) before huggingface_hub is imported
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"   # no hub metadata: sizes only
    q = Queue.load(a.queue)
    t = targets(q)
    report = {"queue": a.queue, "tag": q.tag, "files": [], "errors": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    files = []
    for repo_m, rev in t["models"]:
        names = [f for f in api.list_repo_files(repo_m, revision=rev)
                 if f in MODEL_PATTERNS or f.endswith(".safetensors")] if not a.verify_only else None
        if names is None:
            from huggingface_hub import scan_cache_dir
            names = sorted({f.file_name for r in scan_cache_dir().repos if r.repo_id == repo_m
                            for v in r.revisions if v.commit_hash == rev for f in v.files})
        files += [(repo_m, f, rev) for f in names]
    for s_ in t["saes"]:
        files += list(getattr(s_, "repo_files", lambda: ((s_.repo, s_.path_in_repo, s_.revision),))())
    def retry(fn, tries=4):
        """A 280 GB download meets dropped connections (RemoteProtocolError / timeouts): retry with a growing pause."""
        for k in range(tries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                gated = type(e).__name__ in ("GatedRepoError", "RepositoryNotFoundError", "EntryNotFoundError", "RevisionNotFoundError")
                if k == tries - 1 or gated or isinstance(e, (FileNotFoundError, PermissionError)):
                    raise
                time.sleep(5 * 3 ** k)

    for repo, f, rev in files:
        rec = {"repo": repo, "file": f, "revision": rev}
        t0 = time.time()
        try:
            p = Path(retry(lambda: hf_hub_download(repo, f, revision=rev, local_files_only=a.verify_only)))
            rec["bytes"] = p.stat().st_size
            info = None if offline else next(iter(retry(lambda: api.get_paths_info(repo, [f], revision=rev))), None)
            lfs = getattr(info, "lfs", None) if info else None
            if lfs is not None:
                want = lfs.sha256 if hasattr(lfs, "sha256") else lfs.get("sha256")
                got = sha256(p)
                rec.update(sha256=got, sha256_ok=got == want)
                if got != want:
                    report["errors"].append(f"hash mismatch {repo}/{f}")
            rec["seconds"] = round(time.time() - t0, 1)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            report["errors"].append(f"{repo}/{f}: {rec['error'][:200]}")
        report["files"].append(rec)
        print(f"{'ERR ' if 'error' in rec else 'ok  '} {repo}/{f} {rec.get('bytes', 0) / 2**20:,.0f} MB"
              f"{'' if rec.get('sha256_ok', True) else '  HASH MISMATCH'}")
    report["total_gb"] = round(sum(r.get("bytes", 0) for r in report["files"]) / 2 ** 30, 2)
    out = Path(a.out or REPO_ROOT / "preflight" / f"prefetch_{q.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"{len(files)} files, {report['total_gb']} GB, {len(report['errors'])} errors -> {out}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
