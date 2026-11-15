#!/usr/bin/env python
"""Build the per-language token-window sets for E12 (fidelity) and E13 (behaviour) from public corpora.

    python scripts/build_lang_sets.py                          # all sets -> data/lang/ (needs internet; ~1 GB read)
    python scripts/build_lang_sets.py --only hi_deva_web,fr_web --n-windows 2000
    python scripts/build_lang_sets.py --verify                 # re-hash data/lang/*.json against MANIFEST.json (no network)

Sources and licences are in SOURCES below and in docs/PLAN_COMPLETION.md section 4. Each output is in the pilot's generic-set
format (harness/langsets.py) with its licence, source revision and fertility; MANIFEST.json records the sha256 of each
file. Third-party text ships only as token ids, each file under its own licence (data/lang/LICENSES.md).
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from harness import langsets as L
from harness.profile import REPO_ROOT

OUT = REPO_ROOT / "data" / "lang"
FLOOR = 200_000          # natural tokens per language/script (handoff 7.10)
SEED = 0

# name -> (kind, params, lang, script, genre, licence, how windows are made)
SOURCES = {
    "en_pile":            ("pile", {}, "en", "Latn", "web mix (Pile)", "Pile-10k: see dataset card; research use", "doc"),
    "hi_deva_web":        ("fineweb2", {"cfg": "hin_Deva"}, "hi", "Deva", "web", "ODC-By 1.0", "doc"),
    "fr_web":             ("fineweb2", {"cfg": "fra_Latn"}, "fr", "Latn", "web", "ODC-By 1.0", "doc"),
    "ta_web":             ("fineweb2", {"cfg": "tam_Taml"}, "ta", "Taml", "web", "ODC-By 1.0", "doc"),
    "bn_web":             ("fineweb2", {"cfg": "ben_Beng"}, "bn", "Beng", "web", "ODC-By 1.0", "doc"),
    "hi_deva_dakshina":   ("dakshina", {"lang": "hi", "col": "indic"}, "hi", "Deva", "Wikipedia sentences", "CC BY-SA 4.0 (Dakshina)", "packed"),
    "hi_latn_dakshina":   ("dakshina", {"lang": "hi", "col": "roman"}, "hi", "Latn (human romanization)", "Wikipedia sentences", "CC BY-SA 4.0 (Dakshina)", "packed"),
    "ta_taml_dakshina":   ("dakshina", {"lang": "ta", "col": "indic"}, "ta", "Taml", "Wikipedia sentences", "CC BY-SA 4.0 (Dakshina)", "packed"),
    "ta_latn_dakshina":   ("dakshina", {"lang": "ta", "col": "roman"}, "ta", "Latn (human romanization)", "Wikipedia sentences", "CC BY-SA 4.0 (Dakshina)", "packed"),
    "bn_beng_dakshina":   ("dakshina", {"lang": "bn", "col": "indic"}, "bn", "Beng", "Wikipedia sentences", "CC BY-SA 4.0 (Dakshina)", "packed"),
    "bn_latn_dakshina":   ("dakshina", {"lang": "bn", "col": "roman"}, "bn", "Latn (human romanization)", "Wikipedia sentences", "CC BY-SA 4.0 (Dakshina)", "packed"),
    "hinglish_phinc":     ("phinc", {}, "hi-en", "Latn", "tweets, code-mixed", "CC BY 4.0 (PHINC)", "packed"),
}
PILE_REV = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"      # same revision as data/generic/pile10k_100x128.json


def tokenizer():
    import glob

    from huggingface_hub import HfApi
    from tokenizers import Tokenizer
    rev = "c5ebcd40d208330abc697524c919956e692655cf"          # pinned in every config (model_revision)
    paths = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--google--gemma-2-2b/snapshots" / rev / "tokenizer.json"))
    if paths:
        tk = Tokenizer.from_file(paths[0])
    else:
        from huggingface_hub import hf_hub_download
        tk = Tokenizer.from_file(hf_hub_download("google/gemma-2-2b", "tokenizer.json", revision=rev))
    bos = tk.token_to_id("<bos>")
    return tk, bos, rev


def fetch(kind, p, seed):
    """-> (iterator of texts, source repo, source revision, selection dict). Deterministic given the revision and seed."""
    from huggingface_hub import HfApi
    api = HfApi()
    if kind == "pile":
        from datasets import load_dataset
        ds = load_dataset("NeelNanda/pile-10k", split="train", revision=PILE_REV)
        return (r["text"] for r in ds), "NeelNanda/pile-10k", PILE_REV, {"order": "dataset order"}
    if kind == "fineweb2":
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
        repo, cfg = "HuggingFaceFW/fineweb-2", p["cfg"]
        rev = api.dataset_info(repo).sha
        files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(f"data/{cfg}/train/"))
        fs = HfFileSystem()
        pf = pq.ParquetFile(fs.open(f"datasets/{repo}@{rev}/{files[0]}", "rb"))
        order = list(range(pf.num_row_groups))
        random.Random(seed).shuffle(order)

        def gen():
            for g in order:
                t = pf.read_row_group(g, columns=["text", "language_score"]).to_pydict()
                for txt, sc in zip(t["text"], t["language_score"]):
                    if sc is not None and sc >= 0.9:
                        yield L.normalize(txt)
        return gen(), repo, rev, {"file": files[0], "row_groups_shuffled_seed": seed, "min_language_score": 0.9}
    if kind == "dakshina":
        from datasets import load_dataset
        repo = "suhani-sarvam/google-dakshina"
        rev = api.dataset_info(repo).sha
        ds = load_dataset(repo, split="train", revision=rev)
        idx = [i for i, c in enumerate(ds["lang_code"]) if c == p["lang"]]
        random.Random(seed).shuffle(idx)                   # same order for the Devanagari and the romanized column
        col = ds[p["col"]]
        return (L.normalize(col[i]) for i in idx), repo, rev, {"lang_code": p["lang"], "column": p["col"], "n_sentences": len(idx), "shuffle_seed": seed}
    if kind == "phinc":
        from datasets import load_dataset
        repo = "veezbo/phinc"
        rev = api.dataset_info(repo).sha
        ds = load_dataset(repo, split="train", revision=rev)
        rows = [L.scrub_social(L.normalize(t)) for t in ds["Sentence"]]
        keep = [t for t in rows if t and L.is_hinglish(t)]
        random.Random(seed).shuffle(keep)
        return iter(keep), repo, rev, {"rows": len(rows), "kept_by_romanized_hindi_marker_filter": len(keep),
                                       "min_markers": L.MIN_MARKERS, "shuffle_seed": seed, "scrub": "@handles -> @user, URLs -> http"}
    raise ValueError(kind)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(names, n_windows, out):
    out.mkdir(parents=True, exist_ok=True)
    tk, bos, trev = tokenizer()
    enc = lambda s: tk.encode(s, add_special_tokens=False).ids
    dec = lambda ids: tk.decode(ids, skip_special_tokens=True)
    man_path = out / "MANIFEST.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    for name in names:
        kind, p, lang, script, genre, licence, how = SOURCES[name]
        t0 = time.time()
        texts, repo, rev, sel = fetch(kind, p, SEED)
        make = L.doc_windows if how == "doc" else L.packed_windows
        ids = make(texts, enc, bos, n_windows)
        s = L.make_set(ids, dec, lang=lang, script=script, genre=genre, licence=licence, source=repo, source_revision=rev,
                       tokenizer_revision=trev, how=how, selection=sel, created=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
        s["meets_floor_200k"] = bool(s["tokens_scored"] >= FLOOR)
        f = out / f"{name}.json"
        f.write_text(json.dumps(s, separators=(",", ":")))
        man[name] = {"file": f.name, "sha256": sha256(f), "hash": s["hash"], "lang": lang, "script": script, "licence": licence,
                     "source": repo, "source_revision": rev, "n_docs": s["n_docs"], "tokens_scored": s["tokens_scored"],
                     "words": s["words"], "tokens_per_word": s["tokens_per_word"], "meets_floor_200k": s["meets_floor_200k"],
                     "selection": sel}
        print(f"{name:20s} {s['n_docs']:5d} windows  {s['tokens_scored']:7d} scored tokens  {s['tokens_per_word']:.2f} tok/word  "
              f"floor {'ok' if s['meets_floor_200k'] else 'NOT MET'}  ({time.time() - t0:.0f}s)", flush=True)
        man_path.write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


def verify(out):
    man = json.loads((out / "MANIFEST.json").read_text())
    bad = []
    for name, m in man.items():
        f = out / m["file"]
        if not f.exists() or sha256(f) != m["sha256"]:
            bad.append(name)
            continue
        d = json.loads(f.read_text())
        if L.content_hash(d["ids"]) != d["hash"] or d["hash"] != m["hash"]:
            bad.append(name)
    print("verified" if not bad else f"MISMATCH: {bad}", f"({len(man)} sets)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="")
    ap.add_argument("--n-windows", type=int, default=2000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.verify:
        return verify(out)
    names = [n for n in a.only.split(",") if n] or list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        ap.error(f"unknown sets {unknown}; known: {list(SOURCES)}")
    build(names, a.n_windows, out)
    return verify(out)


if __name__ == "__main__":
    sys.exit(main())
