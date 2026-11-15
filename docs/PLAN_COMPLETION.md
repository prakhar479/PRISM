# Plan to the end of the project (written 2026-11-15, for the humans' approval; nothing here is built yet)

Report window per the handoff: Dec 1-7, 2026 (about three weeks from today). `plan_runs` estimated release 2 at ~14 h of wall time on the shared node (an estimate, not a measurement); the node is shared, so every run below is sized against that.

## 1. Where we are

| item | state |
|---|---|
| Release 2 (E3, E4, E5, E8 fill) | stage 1-3 compute ran on gnode077; the four `*_fin` jobs failed (CUDA asked for on a CPU-only job). Fix written and tested, **staged, unsigned** (GPG timed out). |
| Preregistration | `PREREGISTRATION.md` still says "frozen on <date>": the humans have not frozen it. |
| Experiments with configs | E0-E8 only. Not in the harness: E9, E10, E11 (P2, gated on RQ1), E12, E13 (P1, not gated), E14 (P3). |
| Multilingual data | none in the repo. The handoff says IndicCorp v2 and FineWeb-2 are "already downloaded"; on DEV the HF cache holds only their README files. |

## 2. What "finished" means (tiers; each has a cut line)

| tier | contents | output tables | done when |
|---|---|---|---|
| **MUST** | Release 2 analysed: RQ1 gate (E3), repair specificity (E8, 36 cells), width persistence (E4), subthreshold add-back (E5) | `E3_fresh`, `E8_allcells`, `E4_width_l0`, `E5_subthresh` (+ gate decision table) | each table exists, from ingested bundles with matching run hashes, computed by the preregistered rules (Holm/BH, CI rules) |
| **MUST** | Multilingual measurement: E12 fidelity per language/script (M1); E13 behaviour per language/script incl. people-cities | `E12_fidelity_lang`, `E13_behaviour_lang` | every language row present or an explicit "failed capability gate" row; both capability gates reported |
| **MUST** | Report + figures + RESULTS_INDEX complete; every number traceable to a run hash | `tables/`, `figures/` | figure scripts read only the CSVs |
| **IF GO** (RQ1 GO in >= 1 confirmatory cell) | E9 illusion checks + geometry (+ DAS ranks 1/2/4), E10 steering and binding directions (the proposal's Q2 criterion) | `E9_geometry`, `E10_steering` | criteria in the addendum evaluated, or reported as not run because of NO-GO |
| **STRETCH** | E11 full (people-cities gap/repair L8-16, U transfer both ways); E14 multilingual gap for languages passing both gates; E4_1m; Tamil/Bengali beyond fidelity + capability | `E11_binding_task`, `E14_lang_gap` | only if MUST + IF-GO are finished by Nov 30 |

NO-GO in both cells is a valid end state: the report then states the negative result with CIs, E9-E11 are not run (the
preregistration gates them), and the multilingual work becomes the main extension.

## 3. Order of work (gate-driven, not by experiment number)

1. **Today:** commit the finalize fix (needs the humans to unlock GPG), tag a new run commit; the humans run the CPU finalize loop now (`HL_DEVICE=cpu`, one command per experiment) so they do not wait for the tag.
2. **Humans freeze `PREREGISTRATION.md`** (3 gate decisions already given) **before anyone looks at results.** The addendum (section 7) is frozen with it or immediately after, still before ingest.
3. **Ingest + `make_gate.py` + `make_e8.py` on DEV** (minutes). The gate result chooses a branch:
   - GO in 2 cells: build and run E9, E10 on both; E11 stretch.
   - GO in 1 cell: E9/E10 on that cell only (OT L12 spans are the cached ones; see item 7 below).
   - GO in 0 cells: no E9-E11 runs; skip building DAS/steering; spend the time on E12/E13/E14 and the report.
4. **In parallel from now (not gated):** build E12 and E13 and the data pipeline (sections 4-6). **E13 includes people-cities, so the people-cities task generator, filters and keep-rate checks are built now**, even though the full E11 experiment on it stays gated.
5. Release 3 (E12 + E13) goes to the server as soon as release 2's finalize is done; release 4 (E9, E10 [, E11]) only after the gate.

## 4. Dataset decision (informed; measured where I could)

Constraints: >= 200k natural Gemma tokens per language/script, human-written (no machine transliteration), clear licences
(the GitHub repo `prakhar479/PRISM` is **public**, and ShareAlike/NonCommercial terms would carry over to anything
committed), small enough to ship or fetch quickly, and the same measure as the English 98-99% fidelity number.

| slot | choice | why | measured / verified | open check |
|---|---|---|---|---|
| English | existing Pile sample (`data/generic/`) | already in use | - | genre baseline |
| Hindi (Devanagari), M1 | **FineWeb-2 `hin_Deva`** (ODC-By, not gated; verified via hub API) | large, natural, clear licence | licence verified; not downloaded | sample >= 300k tokens by streaming one shard; record revision + sha256 |
| Hindi, native vs Latin script on identical content | **Dakshina, sentence-aligned** (`hi`, Devanagari and human-romanized columns; CC BY-SA 4.0). The hub mirror `suhani-sarvam/google-dakshina` has columns `lang_code, indic, roman` (aligned rows, also `bn`, others) | the cleanest script-vs-language control: same sentences in two scripts, so any M1 difference is script, not content | structure seen in first rows; token count not yet measured | **verify the mirror against the original Dakshina file** (sample lines + counts) before trusting it; count tokens (survey estimate 250-400k, unverified) |
| Hinglish (code-mixed, Latin script) | **PHINC** (`veezbo/phinc`, CC BY 4.0, human-annotated tweets) | permissive licence, human-written, 13,738 rows | **352,915 Gemma tokens measured** (>= 200k), avg 25.7 tok/row. Genre is tweets; some rows are pure English | add a documented filter (require romanized-Hindi markers); re-count: if the filtered set drops under 200k, add `festvox/cmu_hinglish_dog` (CC BY-SA, its parse failed on first try) or ask the humans to fetch L3Cube (NC; manual Google Drive download) |
| French control | **FineWeb-2 `fra_Latn`** (ODC-By) | same source and licence as Hindi, separates "non-English" from "Indian/low-resource" | licence verified | sample >= 300k tokens |
| Tamil, Bengali (fidelity + capability only) | FineWeb-2 `tam_Taml`, `ben_Beng`; Dakshina `ta`, `bn` rows via the same mirror | cheap (text only); handoff expects the capability gate to fail, which is itself a finding | not downloaded | - |
| **Excluded** | IndicCorp v2 | the handoff names it, but the hub copy is a loader script that is gone (only a README is cached on DEV) and FineWeb-2 covers the same languages with a clearer licence; reason recorded in DECISIONS.md. `hin_Latn` (mixed natural, Hinglish, machine-transliterated, ~half adult), the unlicensed L3Cube hub mirror, `diwank/hinglish-dump` | - | - |

How data ships: **third-party text is not committed** to the public repo. The repo gets a manifest per set (source, revision,
licence, selection seed, row ids, sha256 of the extracted file); `scripts/build_lang_sets.py` fetches and rebuilds, and
fails if the hash differs. Only the CC BY / ODC-By derived samples may be committed (with attribution) if the server has
no internet; the CC BY-SA Dakshina text is fetched, not shipped. DECISIONS.md will log that the humans delegated this
choice on 2026-11-15 ("for dataset i want you to make a infomred choice keeping scope and quality of dat in mind").

## 5. Language scope and the two gates (kept separate)

- Core: English, Hindi (Devanagari), romanized Hindi (Dakshina), Hinglish (PHINC), French (control). Tamil and Bengali: fidelity (E12) and capability only.
- **Gate A (handoff, per language):** usable pairs >= 30% on `scripts/analyze_tasks.py`. Decides whether a language is runnable at all. A failure is reported as a result.
- **Gate B (preregistration):** task accuracy within 10 points of English. Governs any **gap comparison** (E14, M2/M3). Languages that pass A but fail B appear in E13 with accuracy and keep-rates only.
- Templates: Hindi, Hinglish and French versions of OT, RS and people-cities are drafted from the handoff examples (answer is the last token; single-token answers; clean/corrupt equal length). **Automated checks** (tokenizer: single-token answer, equal length, names single-token) run on DEV; a review file is written for the humans. Templates are stamped **unvalidated** until a native speaker signs; E13 allows unvalidated-stamped rows, E14 does not.

## 6. Build list (all DEV-testable; DEV has no GPU)

| id | work | files | DEV check |
|---|---|---|---|
| B1 | finalize fix committed, new `run-` tag | already staged | pytest set green |
| B2 | `scripts/build_lang_sets.py` + manifests (fetch, filter, hash, token count with the Gemma tokenizer already cached on DEV) | `data/lang/*.manifest.json`, script | counts reproduce; hash mismatch fails |
| B3 | E12: fidelity per language (CE recovered mean- and zero-ablation, FVU, ||eps||/||x||, L0, fertility) at L12 and L18, 16k, >= 200k tokens | `harness/fidelity.py`, `configs/experiments/E12.yaml`, queue job | tiny-model run; English row matches the pilot 98-99% within the preflight tolerance |
| B4 | language templates + validator + review file; people-cities generator, filters, keep-rates | `harness/families.py` / `src/data/synthetic_tasks.py` additions, `data/prompt_sets/*_<lang>`, `scripts/validate_templates.py` | tokenizer checks pass or are listed; length-matching success rate reported |
| B5 | E13: behaviour per language/script (accuracy, keep-rates, length-matching) with Gate A/B columns | `configs/experiments/E13.yaml`, `harness/lang.py` | tiny run; dry-run passes |
| B6 | release-3 queue + notes; tables/SCHEMA.md rows; figure scripts reading only CSVs | `queue/release3-server.yaml`, `releases/release3-server.md` | queue dry-run on a fake 4-GPU probe |
| B7 | (after the gate) E9: `feat_A_along_U`, `full_along_U`, geometry (weight-only part on DEV), DAS ranks 1/2/4, cos(U, DAS) | `harness/conditions`, `harness/das.py`, `E9.yaml` | tiny run; DAS recovers a planted direction in a synthetic test |
| B8 | (after the gate) E10: `binding_dirs`, alignment with U and DAS, steering vs >= 20 random directions | `harness/steer.py`, `E10.yaml` | planted-direction test; random-direction baseline in the same run |
| B9 | (stretch) E11 full run on the people-cities task; E14 | configs | dry-run |

Rules for all of it: never fabricate a number; `--smoke` path and DEV `--dry-run` for every config; a CSV with a schema
and a figure script that reads only the CSV; tests for the new pure logic; DECISIONS.md entry for each choice; tags only on RUN.

## 7. Preregistration addendum (drafted by me, frozen by the humans; I do not edit `PREREGISTRATION.md`)

File `PREREGISTRATION_ADDENDUM.md`, covering what the main file has only as one line each: M1 (fidelity lower than English with
non-overlapping CIs, per language), M2 (gap larger than English with non-overlapping CIs, only where Gate B passes), M3
(English-fitted direction applied to Hindi closes a similar share), the E9 illusion-check rules, the E10 Q2 criterion
(>= 50% of pairs move to the other box; matched-norm random <= 5%), the language list and the two gates, and the
dataset manifests above. It must be frozen **before release 3 results are read**.

## 8. Fixes to the current scripts (none blocks the finalize)

1. Finalize jobs on CPU (done, staged).
2. When completing release 2 via sbatch, pass `CLEAN_CACHE_ON_SUCCESS=0` while E9 is still possible: a successful run otherwise deletes the OT L12 span caches E9 reuses.
3. The failed run already wrote `results_server/release2-server/PUBLISHED.json`; the sbatch success check compares against a start stamp, but the humans must ingest the **post-finalize** publish, not the earlier one. Add the finalize run hashes to `PUBLISHED.json` and have `ingest_results.py` refuse a publish whose runs lack a complete `manifest.json`.
4. Let preflight reuse a pass for the same tag and machine so a resubmit does not cost ~32 min.
5. Check `storage_report.py` output on gnode077 (disk kinds, SSD/HDD split) once; it has only been run on WSL.

## 9. Timeline (cut lines in bold)

| dates | work |
|---|---|
| Nov 15-17 | finalize + ingest + gate/E8 tables (humans freeze prereg first); B1, B2 start |
| Nov 17-22 | B2-B6: language sets, E12, templates, E13, release-3 queue; humans review templates |
| Nov 22-27 | release 3 runs (E12 first, ~hours; then E13); if gate = GO, build B7/B8 meanwhile; **cut line: if the templates are not reviewed by Nov 24, E13 runs stamped unvalidated and E14 is dropped** |
| Nov 27-Dec 1 | release 4 (E9, E10) if GO; **cut line: E11 and E4_1m only if everything above is finished by Nov 30** |
| Dec 1-7 | tables, figures, report; no new runs after Dec 2 |

## 10. Human decisions and tasks

1. Unlock GPG so the finalize fix and run tag can be signed (I will not bypass signing).
2. Freeze `PREREGISTRATION.md` and the addendum.
3. Approve this dataset choice (or name the Hinglish alternative); decide whether to fetch L3Cube (NC, manual download) if filtered PHINC falls below 200k.
4. Name who validates the Hindi, Hinglish and French templates, or accept "unvalidated" stamps with E14 dropped.
5. Confirm E11 stays stretch (E13 only needs its task generator).
6. HF token on the server stays the humans'; the language-set builder needs internet on the machine that runs it (DEV has it).

## 11. Risks

- Hindi names and numerals may be multi-token in Gemma's tokenizer; the pipeline already drops and counts such pairs, and the plan uses letters/digits as answers where needed.
- gemma-2-2b may fail the tasks in Tamil/Bengali: reported as capability results.
- Genre mismatch (Pile web vs. tweets vs. Wikipedia sentences vs. FineWeb): M1 differences can reflect genre as well as language; the Dakshina aligned pair is the control that removes content, and the genre caveat is stated in the report.
- Shared-node queueing: a run estimated at 14 h can slip a day; release 3 is ordered so the cheap, text-only E12 lands first.
- DAS (B7) is new training code with no precedent in this repo; it is built only after a GO and is the first thing cut if time runs short.
