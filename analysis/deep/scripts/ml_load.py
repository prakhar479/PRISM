"""Exploratory (post hoc, 2026-11-18). Per-window E12 table: one row per (set, layer, window) with window-level token features.
Writes deep/ml_windows.parquet. Window token features come from the stored token ids + the local gemma-2-2b tokenizer."""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SP = __import__("layout_shim").root()
LANG = SP / "hw" / "data" / "lang"


def tokenizer():
    from tokenizers import Tokenizer
    f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/tokenizer.json"))[0]
    return Tokenizer.from_file(f)


def main():
    man = json.loads((LANG / "MANIFEST.json").read_text())
    tok = tokenizer()
    vocab_n = tok.get_vocab_size()
    feats = {}
    for name in man:
        d = json.loads((LANG / man[name]["file"]).read_text())
        skip = int(d["skip_positions"])
        rows = []
        for ids in d["ids"]:
            sc = ids[skip:]
            pieces = [tok.id_to_token(i) or "" for i in sc]
            byte = np.mean([p.startswith("<0x") for p in pieces])
            plen = np.mean([len(p.replace("▁", "")) for p in pieces])
            word_start = np.mean([p.startswith("▁") for p in pieces])     # SentencePiece word-initial pieces
            rows.append({"byte_frac": byte, "piece_chars": plen, "word_start_frac": word_start,
                         "tok_per_word_w": 1 / max(word_start, 1e-3), "mean_id": float(np.mean(sc)),
                         "uniq_frac": len(set(sc)) / len(sc)})
        feats[name] = pd.DataFrame(rows)
    out = []
    for f in sorted(SP.glob("ana_runs/E12/*/per_window_*_L*.npz")):
        stem = f.stem[len("per_window_"):]
        name, L = stem.rsplit("_L", 1)
        man_run = json.loads((f.parent / "manifest.json").read_text())
        width = man_run["config"].get("width", "16k")
        z = np.load(f)
        df = pd.DataFrame({k: z[k] for k in z.files})
        df["set"], df["layer"], df["width"], df["run"], df["window"] = name, int(L), width, f.parent.name, np.arange(len(df))
        df = pd.concat([df, feats[name]], axis=1)
        df["lang"], df["script"], df["genre"] = man[name]["lang"], man[name]["script"], man[name].get("genre", "")
        df["tpw_set"] = float(man[name]["tokens_per_word"])
        out.append(df)
    w = pd.concat(out, ignore_index=True)
    p = w.pred
    w["ce_clean"], w["ce_splice"], w["ce_mean"], w["ce_zero"] = (w[k] / p for k in ("nll_clean", "nll_splice", "nll_mean", "nll_zero"))
    w["ce_delta"] = w.ce_splice - w.ce_clean
    w["rec_mean"] = (w.ce_mean - w.ce_splice) / (w.ce_mean - w.ce_clean)
    w["fvu"] = w.fvu_num / w.fvu_den
    w["eps_x"] = w.eps_ratio_sum / w.pos
    w["l0"] = w.l0_sum / w.pos
    w.to_parquet(SP / "deep" / "ml_windows.parquet")
    print(w.groupby(["set", "layer"]).size().unstack())


if __name__ == "__main__":
    main()
