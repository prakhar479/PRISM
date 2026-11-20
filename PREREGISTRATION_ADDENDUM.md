# Pre-registration ADDENDUM: multilingual extension and the gated experiments (frozen on 2026-11-16)

Status: drafted by the assistant on 2026-11-15; open items decided by the humans on 2026-11-16 (the humans chose the thresholds in
E to be set by a calibration analysis on data that already existed, `analysis/calibrate_thresholds.txt`, and the M3 definition to
be chosen by the assistant); frozen on 2026-11-16 together with `PREREGISTRATION.md`, before any E9-E14 output existed and before
release-3 results were read. Do not edit; a change needs a new, dated addendum.

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
compared directly (same sentence pool). Decided 2026-11-16: the CI non-overlap rule above, unchanged.

## C. E13: behaviour and the two capability gates

Per (task in {object tracking, running sum, people-cities}, language): clean accuracy, keep-rates, length-matching success.
**Gate A** (runnable): clean accuracy >= 0.6 and clean_only keep >= 0.3 (the pilot's VIABLE rule). **Gate B** (comparable): clean
accuracy within 10 points of English on the same task (the main preregistration's rule). A language that fails A is reported as a
capability result. M2/M3 gap comparisons are made **only** for languages and tasks that pass **both** gates **and** whose template
pack is human-validated (`validated: true`). Template validation: all 15 packs (en, hi, hi_latn, hinglish, fr x 3 tasks) were
reviewed and approved by an author (name redacted for review) on 2026-11-16 (data/lang/signoff.json, bound to each pack's content hash).

## D. M2 and M3 (E14; every language and task is run, a row is READ only if it passes C)

Readable row: Gate A viable, Gate B within 10 points of English, template validated (`readable` in tables/E14_lang_gap.csv).
M2: the gap M(full_A) - M(feat_A_errB) in a language is larger than English's (same task, native fit) with non-overlapping 95% CIs;
primary metric flip rate, NLDR reported alongside; tasks OT L12, RS L18, people-cities L12, 16k pilot variants.
M3 (definition chosen 2026-11-16): non-inferiority of transfer. On the SAME evaluation pairs of language L, share closed by repair
with the English-fitted U minus share closed with the L-fitted U, paired bootstrap; **similar** if the 95% lower bound of that
difference is above -10 points (the same 10-point margin as RQ1 and Gate B). Primary: flip rate, pca_pooled, r = 1 and r = 2
(E8's ranks); answer_sep reported alongside. M3 is evaluated only where the native share's own 95% lower bound is above 0 (there is
something to transfer) and the row is readable (tables/E14_m3_transfer.csv: m3_similar, m3_usable). Rationale: "closes a similar
share" asks whether English directions are as good as native ones; a non-inferiority margin answers that directly and does not
reward a wide CI, which a CI-overlap rule would.
Multiplicity: everything in D is exploratory; BH (q = 0.05) across languages within each task where p-values are reported.

## E. Gated experiments (E9, E10, E11): only if RQ1 = GO for the cell

E9 (illusion checks and DAS) is confirmatory only for a cell with RQ1 = GO (running sum L18, humans 2026-11-16); object tracking
L12 is run as an exploratory, redundant row (two-pass plan, docs/PLAN_TWO_PASS.md).

E9 decision rules (fixed 2026-11-16 from release-2 noise and random geometry only, analysis/calibrate_thresholds.txt; constants in
harness/final.py), per (cell, filter, estimator, rank), primary metric flip rate:
- **effective**: flip rate >= random-U p95 + 10 points AND its 95% lower bound > random-U p95 (same condition and rank, 20 seeds).
  Calibration: random-U repair moves flips by <= 1 point (p95 just above the features-only baseline), so beating random alone is too
  easy; a flip-rate gap's CI half-width is ~3 points at 450 pairs and ~5-6 at 150, so 10 points is >= 2 half-widths.
- **illusion**: full_along_U is effective AND full_along_U - feat_A_along_U >= 10 points with a paired 95% CI above 0 (the subspace
  works on the full residual but not through the SAE features' difference). **feature-mediated**: feat_A_along_U is effective.
- **U-DAS alignment** (top principal cosine between the estimator's rank-k U and DAS_k): aligned >= 0.5; unrelated <= the 99th
  percentile for independent random subspaces of R^2304 (k = 1, 2, 4, 8: 0.052, 0.072, 0.097, 0.131); partial in between.
- **DAS fit valid**: train interchange accuracy >= 0.5; a DAS row with an invalid fit is reported as a failed fit, not as evidence.

E10 (decided 2026-11-16): the main file's Q2 criterion with "move to the other box" read, for every task, as "the steered B run's
top-1 answer becomes A's answer": steer flip rate >= 50% AND the matched-norm random directions' p95 <= 5% (>= 20 random
directions); confirmatory only for running sum L18 (tables/E10_steering.csv: q2_both). E11 (people-cities gap and repair at L8-16,
U transfer with OT both ways) is run as exploratory.
