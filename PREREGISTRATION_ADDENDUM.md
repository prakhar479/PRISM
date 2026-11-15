# Pre-registration ADDENDUM: multilingual extension and the gated experiments (DRAFT, not frozen)

Status: proposed by the assistant on 2026-11-15 for the humans to edit and freeze. `PREREGISTRATION.md` is not touched. Nothing
below has been run; E12/E13 results must not be read before this is frozen. Items marked **[HUMAN]** are decisions I could not
make for you; they are left open on purpose and must be filled before freezing.

## A. Languages, data, and what is exploratory

All multilingual work is **exploratory** (the main file names only OT L12 and RS L18 at 16k as confirmatory). Languages: English
(baseline), Hindi (Devanagari), romanized Hindi, Hinglish (code-mixed, Latin script), French (non-Indian control). Tamil and
Bengali: fidelity (E12) only, plus Dakshina script pairs; no task packs because the tokenizer makes single-token answers
impossible for Devanagari/Tamil/Bengali city names (DECISIONS.md 2026-11-15).

Data sets are fixed by content hash (data/lang/MANIFEST.json); changing one after freezing needs a new addendum:

| set | source | licence | windows | scored tokens | content hash |
|---|---|---|---|---|---|
| bn_beng_dakshina | suhani-sarvam/google-dakshina | CC BY-SA 4.0 (Dakshina) | 2000 | 248000 | `67fc022f8f253f11` |
| bn_latn_dakshina | suhani-sarvam/google-dakshina | CC BY-SA 4.0 (Dakshina) | 2000 | 248000 | `72b6adc7d9ed957e` |
| bn_web | HuggingFaceFW/fineweb-2 | ODC-By 1.0 | 2000 | 248000 | `fbc62e241271b66a` |
| en_pile | NeelNanda/pile-10k | Pile-10k: see dataset card; research use | 2000 | 248000 | `7aa0be6671ce9c2d` |
| en_pilot100 | NeelNanda/pile-10k | Pile-10k: see dataset card; research use | 100 | 12400 | `eb64527cbbe066fb` |
| fr_web | HuggingFaceFW/fineweb-2 | ODC-By 1.0 | 2000 | 248000 | `a70323c666c3a590` |
| hi_deva_dakshina | suhani-sarvam/google-dakshina | CC BY-SA 4.0 (Dakshina) | 2000 | 248000 | `3abe90067318e796` |
| hi_deva_web | HuggingFaceFW/fineweb-2 | ODC-By 1.0 | 2000 | 248000 | `4d027cfbf600f844` |
| hi_latn_dakshina | suhani-sarvam/google-dakshina | CC BY-SA 4.0 (Dakshina) | 2000 | 248000 | `61ae6b8df8768d1a` |
| hinglish_phinc | veezbo/phinc | CC BY 4.0 (PHINC) | 1663 | 206212 | `6bea89bb5bb1e8eb` |
| ta_latn_dakshina | suhani-sarvam/google-dakshina | CC BY-SA 4.0 (Dakshina) | 2000 | 248000 | `c6b9835bc340dd27` |
| ta_taml_dakshina | suhani-sarvam/google-dakshina | CC BY-SA 4.0 (Dakshina) | 2000 | 248000 | `5f6de221f439beb2` |
| ta_web | HuggingFaceFW/fineweb-2 | ODC-By 1.0 | 2000 | 248000 | `3729a6cfefdaee54` |

Known confounds, to be stated in the report, not tested away: genre differs across sets (Pile web mix, FineWeb-2 web, Wikipedia
sentences, tweets); Hindi-Devanagari fertility (tokens per word) differs from English's; the two Dakshina scripts cover overlapping
but not identical sentence windows.

## B. E12 (M1): SAE fidelity on ordinary text per language

Measures per (set, layer in {12, 18}), 16k pilot variants (L12 l0=82, L18 l0=74): CE-loss-recovered with mean ablation (the pilot's
measure; primary) and with zero ablation, FVU, |eps|/|x|, L0, fertility; 95% bootstrap CIs over windows.
**M1 is supported for a language and layer** if its CE-recovered (mean ablation) upper CI bound is below English's lower bound
(`en_pile`, same layer): "clearly lower than English" in the handoff, read as non-overlapping CIs. Report the result for both layers
and both ablations; a language that is lower at one layer and not the other is reported as mixed. The parity check is `en_pilot100`
against the pilot's published 98-99% (not a hypothesis test). Script versus language: the Dakshina native and romanized rows are
compared directly (same sentence pool). **[HUMAN]** Do you want a stricter rule (for example a minimum difference in points) or a
difference-of-means CI instead of CI non-overlap? Default if unanswered: the rule above.

## C. E13: behaviour and the two capability gates

Per (task in {object tracking, running sum, people-cities}, language): clean accuracy, keep-rates, length-matching success.
**Gate A** (runnable): clean accuracy >= 0.6 and clean_only keep >= 0.3 (the pilot's VIABLE rule). **Gate B** (comparable): clean
accuracy within 10 points of English on the same task (the main preregistration's rule). A language that fails A is reported as a
capability result. M2/M3 gap comparisons are made **only** for languages and tasks that pass **both** gates **and** whose template
pack is human-validated (`validated: true`). **[HUMAN]** Who validates the Hindi, Hinglish and French templates (data/lang/
templates_review.md)? Until then E14 does not run.

## D. M2 and M3 (E14, only if the conditions in C hold; not built)

M2: the gap (1 - SAE-parts share) in a language is larger than English with non-overlapping CIs, using E2's conditions at OT L12 / RS L18.
M3: the direction fitted on English, applied to the language's pairs, closes a similar share of that language's gap (**[HUMAN]**
define "similar", for example within 10 points of the English share). Holm across the two confirmatory cells; BH (q = 0.05) across
languages within the exploratory family.

## E. Gated experiments (E9, E10, E11): only if RQ1 = GO for the cell

E9 (illusion checks and DAS) runs on a cell only after RQ1 = GO for it. Thresholds for what counts as an "illusion" and for the
cosine between U and the DAS direction are **[HUMAN]** decisions and are deliberately not invented here. E10 uses the main file's Q2
criterion unchanged: >= 50% of pairs move to the other box when steering along the direction; matched-norm random directions
<= 5% (>= 20 random directions). E11 (people-cities gap and repair at L8-16, U transfer with OT both ways) is a stretch goal.
