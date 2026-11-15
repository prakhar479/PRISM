# data/lang: licences and provenance

Each `*.json` here is a set of token-id windows (Gemma-2 tokenizer, revision `c5ebcd40…`) cut from a public corpus by
`scripts/build_lang_sets.py` (NFC normalisation and whitespace collapse; windows of 127 tokens after BOS; for tweets, @handles
replaced by `@user` and URLs by `http`). The ids can be decoded back to the text, so each file is **an adaptation of its source
and carries that source's licence**. `MANIFEST.json` lists source, revision, sha256 and counts per file; run
`python scripts/build_lang_sets.py --verify` to check the files.

| files | source (revision in MANIFEST.json) | licence | attribution / terms |
|---|---|---|---|
| `en_pile.json` | `NeelNanda/pile-10k` (a 10k-document sample of the Pile) | see the dataset card; mixed-source web text, used here for research only | the Pile (EleutherAI); sample by Neel Nanda |
| `hi_deva_web`, `fr_web`, `ta_web`, `bn_web` | `HuggingFaceFW/fineweb-2` (`hin_Deva`, `fra_Latn`, `tam_Taml`, `ben_Beng`) | ODC-By 1.0 | FineWeb-2, Hugging Face; underlying text is CommonCrawl web data, subject to its own terms |
| `hi_deva_dakshina`, `hi_latn_dakshina`, `ta_*_dakshina`, `bn_*_dakshina` | Dakshina v1.0 (Google Research) via the hub mirror `suhani-sarvam/google-dakshina` | **CC BY-SA 4.0** | Dakshina dataset, Google Research. ShareAlike: these files stay under CC BY-SA 4.0. The mirror's contents were checked on a sample against the originals' structure only, not byte-for-byte |
| `hinglish_phinc` | `veezbo/phinc` (PHINC, human-annotated code-mixed tweets) | CC BY 4.0 | PHINC authors; handles and URLs scrubbed; rows filtered to those with >= 2 romanized-Hindi function words |

Nothing in this directory is machine-transliterated. The Dakshina romanizations are the dataset's human ones.
