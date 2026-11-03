# Multilingual corpora for E12 / E13 (survey 2026-11-03, for the humans' decision)

Read-only check of the Hugging Face hub and the web (handoff 7.10: >= 200k natural tokens per language/script; no
machine transliteration). Token counts marked ~ are estimates, not verified.

| slot | candidate | size | licence | caveat (humans decide) |
|---|---|---|---|---|
| English | Pile sample already in `data/generic/` | in use | as now | none |
| Hindi, Devanagari | FineWeb-2 `hin_Deva` (HuggingFaceFW/fineweb-2) | 22.15M docs | ODC-By, not gated | web genre; Dakshina's native-script side gives a genre-matched (Wikipedia) control but is small |
| romanized Hindi (no English) | Dakshina v1.0 `hi/romanized/hi.romanized.rejoined.tsv` (Google; github.com/google-research-datasets/dakshina) | ~10k human-romanized Wikipedia sentences, ~250-400k tokens | CC BY-SA 4.0 | close to the 200k floor; Wikipedia genre; English only as names/terms; parallel Devanagari available |
| romanized Hindi (alternative) | FineWeb-2 `hin_Latn` | 97k docs, ~10^8 tokens | ODC-By | sampled rows mix natural romanized Hindi, Hinglish, machine-transliterated pages (bharatdiscovery / krishnakosh / angrejinews) and ~half adult content: needs domain filtering |
| Hinglish (code-mixed) | L3Cube-HingCorpus (github.com/l3cube-pune/code-mixed-nlp) | 52.9M tweets, ~1.04B tokens | CC BY-NC-SA 4.0 | non-commercial; tweets; official copy via Google Drive (hub mirror `hinglish-orthobench/HingCorpus` has no licence card) |
| Hinglish (alternative) | `hin_Latn` filtered to code-mixed blogs; or small sets `veezbo/phinc` (13.7k tweets, CC BY 4.0), `festvox/cmu_hinglish_dog` (10k chat turns, CC BY-SA 3.0) | small sets < 200k tokens | as listed | the small sets alone miss the 200k floor |
| French | FineWeb-2 `fra_Latn` | 360M docs | ODC-By | none |

Not natural running text (excluded): `diwank/hinglish-dump` (transliteration mix), IndicXlit / Aksharantar (word pairs),
`sk-community/romanized_hindi` outside its `dakshina_data/` file (includes synthetic romanizations). Not checked: GLUECoS;
LinCE's current hub id. No source requires machine transliteration, so the handoff's stop-and-ask rule is not triggered;
the source choice for romanized Hindi and Hinglish remains the humans'.
