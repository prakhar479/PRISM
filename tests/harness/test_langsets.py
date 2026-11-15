"""Language-set windows: format, packing, the Hinglish filter, determinism."""
import json

import pytest

from harness import langsets as L

enc = lambda s: [ord(c) % 250 + 3 for c in s]            # one token per character: deterministic, no model needed
dec = lambda ids: "".join(chr((i - 3) % 250) for i in ids)


def test_doc_windows_take_first_tokens_of_long_docs_only():
    texts = ["a" * 300, "b" * 10, "c" * 200]
    w = L.doc_windows(texts, enc, bos=2, n_windows=5, seq_len=128)
    assert len(w) == 2 and all(len(r) == 128 and r[0] == 2 for r in w)
    assert w[0][1:] == enc("a" * 127)


def test_packed_windows_are_contiguous_and_stop_at_n():
    texts = ["x" * 50] * 20
    w = L.packed_windows(texts, enc, bos=2, n_windows=3, seq_len=128)
    assert len(w) == 3 and all(len(r) == 128 for r in w)
    flat = enc(("x" * 50 + "\n") * 20)
    assert w[0][1:] == flat[:127] and w[1][1:] == flat[127:254]
    assert len(L.packed_windows(texts, enc, 2, 99, 128)) == (20 * 51) // 127   # tail dropped


def test_hinglish_filter_needs_two_markers():
    assert L.is_hinglish("woh dekh kitna bada saap hai")
    assert not L.is_hinglish("this is plain english text")
    assert not L.is_hinglish("Radio ki baat")                    # only one marker ("ki") -> rejected on purpose
    assert L.is_hinglish("Radio ki baat hi kuch aur hai")


def test_make_set_format_hash_and_fertility():
    w = L.packed_windows(["hello world foo bar"] * 40, enc, 2, 4, 128)
    s = L.make_set(w, dec, lang="xx", script="Latn", genre="t", licence="CC0", source="s", source_revision="r",
                   tokenizer_revision="t", how="packed", selection={"seed": 0})
    assert s["n_docs"] == 4 and s["seq_len"] == 128 and s["skip_positions"] == 4 and len(s["ids"]) == 4
    assert s["tokens_scored"] == 4 * 124 and 0 < s["tokens_per_word"] < 10
    assert s["hash"] == L.content_hash(w)
    json.dumps(s)
    with pytest.raises(ValueError):
        L.make_set([[2, 3]], dec, lang="x", script="x", genre="x", licence="x", source="x", source_revision="x",
                   tokenizer_revision="x", how="x", selection={})


def test_normalize_keeps_content():
    assert L.normalize("a  b\t c\n\n  d ") == "a b c\nd"


def test_scrub_social_removes_handles_and_urls():
    t = L.scrub_social("@someone_1 check https://x.co/abc and www.y.org now @b")
    assert "someone" not in t and "x.co" not in t and "y.org" not in t and t.count("@user") == 2 and "http" in t
