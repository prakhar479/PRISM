#!/usr/bin/env python
"""Stage 0 for generic text: the pilot's generic-text token batches (src.experiments.common.generic_token_batches,
NeelNanda/pile-10k, first documents with >= seq_len tokens, truncated) saved as token ids with a content hash.

    python scripts/build_generic_set.py --n-docs 100 --seq-len 128      # paper_residue settings -> data/generic/
    python scripts/build_generic_set.py --n-docs 6000 --skip-docs 100   # breadth pilot fit set, disjoint from the 100 above

--skip-docs N drops the first N qualifying documents (those the smaller set uses), so fit and evaluation sets never share a
document; the output name then carries _skip<N>.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NeelNanda/pile-10k")
    ap.add_argument("--n-docs", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--skip-positions", type=int, default=4)
    ap.add_argument("--skip-docs", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    from harness.profile import load_profile
    from src.experiments.common import generic_token_batches

    rev = HfApi().model_info("google/gemma-2-2b").sha
    ds_rev = HfApi().dataset_info(a.dataset).sha
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", revision=rev)
    cfg = {"seed": 0, "generic_text": {"dataset": a.dataset, "n_docs": a.n_docs + a.skip_docs, "seq_len": a.seq_len}}
    batches = generic_token_batches(tok, cfg, mock=False)
    ids = [row.tolist() for b in batches for row in b][a.skip_docs:]
    h = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    out = Path(a.out) if a.out else load_profile().root("data_root") / "generic" / (f"pile10k_{a.n_docs}x{a.seq_len}"
                                                                             + (f"_skip{a.skip_docs}" if a.skip_docs else "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": a.dataset, "dataset_revision": ds_rev, "tokenizer_revision": rev,
                               "n_docs": len(ids), "skip_docs": a.skip_docs, "seq_len": a.seq_len, "skip_positions": a.skip_positions, "hash": h,
                               "source": "src.experiments.common.generic_token_batches (pilot)",
                               "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "ids": ids}))
    print(f"{len(ids)} x {a.seq_len} tokens -> {out} (hash {h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
