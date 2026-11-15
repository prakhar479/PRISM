"""Fixed-length token windows of natural text per language/script, in the pilot's generic-set format
(`data/generic/pile10k_100x128.json`: `ids` = n_docs x seq_len token ids, BOS first, `skip_positions`, `hash`), plus the
facts E12/E13 report per set: language, script, genre, licence, source revision, tokens scored, words, fertility.

Pure functions (no network, no model): the fetchers live in scripts/build_lang_sets.py. Two ways to make windows:
  - `doc_windows`: one window per document that has >= seq_len tokens (the pilot's rule: first seq_len tokens);
  - `packed_windows`: short texts (sentences, tweets) are joined with "\\n" and cut into consecutive windows.
Windows are never padded and never contain machine-transliterated or synthetic text: that is a property of the source
(see docs/PLAN_COMPLETION.md section 4), not something this module can check.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Callable, Dict, Iterable, List, Sequence

SKIP_POSITIONS = 4          # BOS / attention-sink positions are left untouched and excluded from the loss (pilot)

# unambiguous romanized-Hindi function words (no common English homograph); a Hinglish row needs >= MIN_MARKERS distinct ones
HINDI_MARKERS = frozenset("""hai hain nahi nahin nhi kya kyun kyu kaise kaun kab kahan hoga hogi honge tha thi the_
mera meri mere tera teri tere apna apni apne aap tum tumhare hum hamara yeh woh wo yahan wahan bhi aur lekin magar
agar toh ko ke ki ka se mein mai abhi bahut bohot sab kuch koi isliye kyunki liye raha rahi rahe karna karo kar
dekh dekho bol bolo chal chalo jao aao accha acha theek bas yaar bhai""".split()) - {"the_"}
MIN_MARKERS = 2


def normalize(text: str) -> str:
    """NFC, collapse whitespace runs (kept as one space, newlines kept as one newline). No other cleaning."""
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return re.sub(r"\s*\n\s*", "\n", t).strip()


def scrub_social(text: str) -> str:
    """Tweets: replace @handles with `@user` and URLs with `http` (no personal handles in a public repo); nothing else changes."""
    t = re.sub(r"@\w+", "@user", text)
    return re.sub(r"https?://\S+|www\.\S+", "http", t)


def is_hinglish(text: str, min_markers: int = MIN_MARKERS) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & HINDI_MARKERS) >= min_markers


def doc_windows(texts: Iterable[str], encode: Callable[[str], List[int]], bos: int, n_windows: int, seq_len: int = 128) -> List[List[int]]:
    """First seq_len tokens (BOS + seq_len-1) of each text that has at least that many; stops at n_windows."""
    out: List[List[int]] = []
    for t in texts:
        ids = encode(t)
        if len(ids) >= seq_len - 1:
            out.append([bos] + ids[:seq_len - 1])
            if len(out) >= n_windows:
                break
    return out


def packed_windows(texts: Iterable[str], encode: Callable[[str], List[int]], bos: int, n_windows: int, seq_len: int = 128,
                   sep: str = "\n") -> List[List[int]]:
    """Join short texts with `sep` (tokenised one by one, so no text is split mid-token by the join) and cut windows of
    BOS + seq_len-1 tokens. The tail that does not fill a window is dropped."""
    buf: List[int] = []
    out: List[List[int]] = []
    sep_ids = encode(sep) if sep else []
    for t in texts:
        buf.extend(encode(t))
        buf.extend(sep_ids)
        while len(buf) >= seq_len - 1:
            out.append([bos] + buf[:seq_len - 1])
            buf = buf[seq_len - 1:]
            if len(out) >= n_windows:
                return out
    return out


def content_hash(ids: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(json.dumps([list(map(int, r)) for r in ids], separators=(",", ":")).encode()).hexdigest()[:16]


def fertility(ids: Sequence[Sequence[int]], decode: Callable[[List[int]], str], skip: int = SKIP_POSITIONS) -> Dict[str, float]:
    """Tokens per whitespace word and per character over the scored positions (windows decoded without special tokens)."""
    n_tok = words = chars = 0
    for r in ids:
        n_tok += len(r) - skip
        txt = decode(list(r[skip:]))
        words += len(txt.split())
        chars += len(txt)
    return {"tokens_scored": n_tok, "words": words, "chars": chars,
            "tokens_per_word": round(n_tok / max(words, 1), 3), "tokens_per_char": round(n_tok / max(chars, 1), 4)}


def make_set(ids: Sequence[Sequence[int]], decode: Callable[[List[int]], str], *, lang: str, script: str, genre: str,
             licence: str, source: str, source_revision: str, tokenizer_revision: str, how: str, selection: dict,
             seq_len: int = 128, skip: int = SKIP_POSITIONS, created: str = "") -> dict:
    if any(len(r) != seq_len for r in ids):
        raise ValueError("every window must have exactly seq_len tokens")
    stats = fertility(ids, decode, skip)
    return {"dataset": source, "dataset_revision": source_revision, "tokenizer_revision": tokenizer_revision,
            "n_docs": len(ids), "seq_len": seq_len, "skip_positions": skip, "hash": content_hash(ids),
            "source": "harness.langsets via scripts/build_lang_sets.py", "created": created,
            "lang": lang, "script": script, "genre": genre, "licence": licence, "windows": how, "selection": selection,
            **stats, "ids": [list(map(int, r)) for r in ids]}
