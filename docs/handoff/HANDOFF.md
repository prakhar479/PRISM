# Handoff: course project proposal (LaTeX) + presentation (Beamer)

You are writing two documents for a course professor about a research project called **PRISM**. Everything you
need is in this folder: the context, the results with exact numbers, seven ready figures (vector PDF), the data
behind every figure, and the plan for a new multilingual extension. **Do not run experiments and do not invent
numbers or citations.** Every number you write must come from `data/*.csv` or this file.

Contents of `docs/handoff/`:

| Path | What it is |
|---|---|
| `HANDOFF.md` | This brief |
| `figures/f1..f7.pdf` (+ `.png` previews) | Final figures, vector PDF for `\includegraphics` |
| `data/f1..f7_*.csv`, `data/key_numbers.csv` | The exact numbers behind each figure and the headline claims |
| `../../scripts/make_handoff_figures.py` | Regenerates all of the above from the results folders |

---

## 1. The assignment (from the professor, verbatim)

> Overall, you should try to cover three questions: What, Why and How. (optional fourth: When).
> **What** is the main problem statement of the project? What will you investigate? What do you hope to achieve?
> **Why** this problem? Explained in terms of your motivation, and potential impact of the problem.
> **How** will you go about tackling it? Task description, have you finalised any dataset, tasks to evaluate etc?
> You may also include a timeline of the project, and estimated completions of various steps in the project
> (including the progress made if any).
> Multilingual/Indian language projects are not mandatory but recommended.

**Deliverables**

1. `proposal/proposal.tex`: a **short, simple** project proposal, 2 to 3 pages (A4, 11pt, single column), with
   sections What / Why / How / When. Plain language a non-specialist can follow; define every technical term
   once. At most 3 figures.
2. `slides/slides.tex`: a Beamer deck, **12 to 15 slides**, 16:9, for a 10 to 12 minute talk.
3. A shared `refs.bib` (see section 9: only references you have verified).

The student wants to **add a multilingual extension** (Indian languages recommended by the course): the same
analysis, repeated in other languages. It is a plan, not a result: section 6 has the design.

---

## 2. The project in one paragraph

Sparse autoencoders (SAEs) are the main tool interpretability researchers use to break a language model's
internal signal into human-readable parts ("features"). Whatever an SAE cannot explain is left over as
reconstruction **error**, sometimes called "dark matter". We asked: when a model keeps track of something while
reading (a running total, where an object is), is that tracked information captured by the SAE's readable parts,
or does it hide in the error? Using Google's gemma-2-2b model and its official Gemma Scope SAEs, we found that
the SAE's parts carry most of it (74% to 100%), so the popular "dark matter hides the computation" story is
largely wrong here. But for one task, **tracking which object is in which box**, the SAE misses about a quarter of
the effect, half of that missing piece sits in a **single direction**, the direction is **specific to that task**,
and a **16 times wider SAE shrinks the gap but does not close it**. We also found three common testing traps that
give false results. Next, we test (a) whether the missing piece is the model's *binding* mechanism and (b)
whether SAEs lose more in Indian languages than in English.

**Suggested title:** *Where Do Sparse Autoencoders Lose Task State? A Causal Audit in English and Indian Languages*
(short: "PRISM").

---

## 3. Background, explained simply (use these definitions)

- **Residual stream (x):** the model's running internal signal at one layer, a vector of 2304 numbers per token.
- **SAE:** a tool that rewrites x as a short list of active, readable parts: `x = x_hat + eps`, where `x_hat` is the
  SAE's reconstruction (sum of a few readable parts) and `eps = x - x_hat` is the **leftover error**.
- **Clean / corrupt prompts:** two prompts that differ in one step, so the right answer differs.
  Example: clean `"Start with 2. Add 3. Add 1. Add 2. Total ="` → 8; corrupt changes `Add 1` to `Add 3` → 10.
- **Patching ("copying"):** run the corrupt prompt, but at one layer replace part of its signal with the same part
  from the clean prompt, and see whether the clean answer comes back.
- **NLDR (share of the right answer restored):**
  `NLDR = (D_patched - D_corrupt) / (D_clean - D_corrupt)`, where `D` = logit(clean answer) - logit(corrupt answer),
  averaged over test pairs. 0 = nothing restored, 1 (100%) = fully restored.
- **The causal split (main test):**
  - "SAE parts only" = copy `x_hat` from clean, keep `eps` from corrupt;
  - "error only" = copy `eps` from clean, keep `x_hat` from corrupt.
  If tracked information hides in the error, "error only" should restore more than "SAE parts only".
- **Controls:** a full copy (should give 100%); replacing the error with random noise of the same size; random
  directions as a null for every "directions" result.
- **"SAE parts + N error directions" (repair test):** SAE parts from clean, plus the clean error along the top N
  directions of the error, the rest of the error from corrupt. N = 0 is "SAE parts only".

---

## 4. What was done (setup)

| Item | Value |
|---|---|
| Model | gemma-2-2b (26 layers, residual width 2304), bf16 |
| SAEs | Gemma Scope residual-stream JumpReLU SAEs (`gemma-scope-2b-pt-res-canonical`), widths 16k to 262k |
| SAE quality check | the 16k SAEs recover 98.0% to 99.1% of the language-modelling loss on ordinary web text (`key_numbers.csv`) |
| Compute | one 8 GB laptop GPU (RTX 5060); the three main runs took about 6.3 hours |
| Test size | 300 training pairs and 150 test pairs per task and setting; 95% bootstrap confidence intervals |
| Code | a tested pipeline (78 unit tests) that checks itself against Hugging Face's own model |

**Tasks (all with clean/corrupt pairs that differ in one step):**

| Task | Example (clean) | What the model tracks | Setting used |
|---|---|---|---|
| Running sum | `Start with 2. Add 3. Add 1. Add 2. Total =` → 8 | a running total | 3 steps, 8 examples |
| Object tracking | `The key is in box B. The cup is in box E. The key moves to box C. ... The key is in box` → C | which object is in which box (binding) | 2 moves, 8 examples |
| Variable copying | `t = 7. q = 4. x = q. q = 2. x =` → 4 | the value bound to a variable | 1 hop, 2 examples |
| Names (control) | `When Mary and John went to the store, John gave a drink to` → Mary | which name is the recipient (a well-studied circuit) | 2 examples |

The original task set (sums with a remainder, bracket depth, graph size) was **dropped**: gemma-2-2b answers at
chance on them at every difficulty (figure f1). That is why the first overnight run produced nothing.

---

## 5. Results (all numbers from `data/`)

### 5.1 Headline findings

1. **The original idea is ruled out.** Copying the SAE's parts restores 74% to 100% of the right answer for the
   three tracking tasks at every layer tested (4 to 20); the error alone restores 0% to 13% except object tracking
   in the middle layers, layers 8 to 16 (f2). A full copy restores 100% everywhere, so the test works.
2. **One focused gap: object tracking.** At layer 12 the SAE misses 26% of the effect (SAE parts restore 74%,
   95% CI 73% to 76%). One direction taken from the error closes about half of that gap (74% → 88%); random
   directions close none (f7).
3. **The missing piece is task-specific.** Directions fitted on ordinary web text close 1% of the object-tracking
   gap; directions from other tasks close 0% to 2%; the task's own directions close 39% to 53% (f6). Same pattern
   for the running sum at layer 18 (58% vs 0% to 1%).
4. **Wider SAEs shrink the gap but do not close it.** For object tracking at layer 12 the gap goes 26% → 20% → 24%
   → 23% → 15% from 16k to 262k parts, and at every width one error direction still recovers about half of what
   is missing (f5). The other tasks' gaps are small (at most 7%) at every width.
5. **Three testing traps** (each would have given a false result):
   - **Trap 1, copying only the last token:** in early and middle layers the state still sits on earlier tokens, so
     copying only the last token restores 0% to 47% even with the full signal; copying from the first changed
     token restores 100% (f3).
   - **Trap 2, the prompt's words give the answer away:** a probe that sees only the prompt's words (no model) scores
     100% on variable copying and the names task, so a high probe score on those tasks proves nothing (f4).
   - **Trap 3, readable is not the same as used:** at layer 12 a probe reads the state far better from the error
     than from the SAE's parts (object tracking 84% vs 48%; names 97% vs 20%), yet copying that token restores
     only 0% to 47%: the model is not using it there yet (f4).

### 5.2 Figures (use the PDFs; titles are already in the image, so captions can be short)

| File | Shows | Use in |
|---|---|---|
| `f1_task_viability` | Which tasks the model can do (usable pairs per task; 30% needed) | proposal (How), slide "Tasks" |
| `f2_causal_split` | SAE parts vs error, 4 tasks x layers 4 to 20, with CIs | **proposal main figure**, slide "Main result" |
| `f3_trap_positions` | Trap 1: last-token vs from-first-change copying across layers | slide "Traps" |
| `f4_trap_probes` | Traps 2 and 3 at layer 12: probes vs prompt-words baseline vs use | slide "Traps" |
| `f5_widths` | Gap vs SAE width (16k to 262k), with and without one error direction | proposal (results), slide "Does a bigger SAE fix it?" |
| `f6_specificity` | Share of the gap closed by own-task, other-task, web-text and random directions | slide "Task-specific" |
| `f7_repair_curve` | Object tracking: adding N error directions vs web-text and random | slide "One direction" |

### 5.3 Honest caveats (state these; do not overclaim)

- Results are for **one model** (gemma-2-2b) and **one SAE family** (Gemma Scope), on synthetic tasks.
- The names control also loses up to 24% to 36% at layers 18 to 20, but needs about 32 directions to recover half,
  so part of the late-layer loss is a general SAE weakness, not specific to tracking.
- For object tracking the error-only number (62% at layer 12) is inflated: replacing the error with noise alone
  already gives 51% (removing the wrong prompt's error matters). The clean error adds about 11 points.
- In the width sweep, the Gemma Scope 131k and 262k SAEs also use more active parts per token (L0 about 124 to
  139 vs 75 to 89 at 16k), so part of the shrinking gap may be due to sparsity, not width. A sparsity sweep is planned.
- The **binding** explanation is a **hypothesis** (motivated by the object-tracking result), not a finding yet.
- The multilingual part is a **plan**; there are no multilingual results.

---

## 6. The multilingual extension (new; this is "planned work")

**Question.** Gemma Scope SAEs were trained on gemma-2-2b's activations over its pretraining text (assumed mostly
English; verify before stating it). Do they lose more of the model's tracked information in Indian languages than
in English, and is the missing piece the same across languages?

**Why it matters.** SAEs are being used to audit and steer models for users who do not speak English. If SAEs
systematically miss more in Hindi or Tamil, SAE-based safety and interpretability claims would hold less for
those users. India-focused and low-resource languages are exactly where this risk is least measured.

**Hypotheses (each testable with the existing pipeline):**

| ID | Hypothesis | Measure | Supported if |
|---|---|---|---|
| M1 | SAEs reconstruct Indian-language text worse | SAE loss recovered on ordinary text per language (same measure as the 98-99% English number) | clearly lower than English for Hindi / Tamil |
| M2 | SAEs miss more task state in Indian languages | the causal split and the gap (1 - SAE parts) per language | gap larger than English with non-overlapping CIs |
| M3 | The missing piece is language-independent | fit the missing direction on English, apply it to Hindi (cross-lingual transfer) | it closes a similar share of the Hindi gap |

**Languages (in priority order):**

1. **English**: baseline (results above).
2. **Hindi** (Devanagari script): the most-resourced Indian language; most likely to work.
3. **Hinglish** (Hindi in Latin script, code-mixed): very common online; separates "language" from "script".
4. **Tamil or Bengali**: a Dravidian / another-script language, likely harder for a 2B model.
5. **French or Spanish**: a high-resource non-Indian control, to separate "non-English" from "Indian/low-resource".

**How (design):**

- **Same tasks, translated templates**, keeping the answer as the **last token** (Hindi is verb-final, so use a
  question + "उत्तर:" (answer) format where needed). Illustrative examples, **to be checked by a native speaker
  and against the tokenizer**:
  - Running sum (Hindi): `शुरुआत 2 से करें। 3 जोड़ें। 1 जोड़ें। 2 जोड़ें। कुल =` → 8
  - Object tracking (Hindi): `सिक्का डिब्बे B में है। चाबी डिब्बे E में है। चाबी डिब्बे C में जाती है। चाबी किस डिब्बे में है? डिब्बा` → C
  - Names (Hinglish): `Jab Rahul aur Priya market gaye, Priya ne ek book di. Book kisko mili? Answer:` → Rahul
  - Variable copying: code-like, so it stays the same in every language (a natural language-neutral control).
- **Pipeline constraints:** clean and corrupt prompts must have the same number of tokens, and the answer must be
  one token. Hindi names are often several tokens in Gemma's tokenizer, so names will be chosen from those that
  are single tokens; the pipeline already drops and counts pairs that violate this.
- **Step 1, capability gate (per language):** the existing task-analysis script checks whether gemma-2-2b can do
  each task (usable pairs ≥ 30%). Languages that fail are reported as such (itself a finding).
- **Step 2, SAE fidelity (M1):** ordinary text per language from **IndicCorp v2** (AI4Bharat) and **FineWeb-2**
  (both already downloaded on the project machine).
- **Step 3, causal split + gap + repair (M2):** the same three paper runs, per language.
- **Step 4, cross-lingual transfer (M3):** reuse the "transfer" method, with an English-fitted direction applied to
  Hindi pairs.

**Risks and fallbacks:** gemma-2-2b may fail the tasks in Tamil/Bengali (then report capability, keep Hindi and
Hinglish); single-token constraints may limit names (use letters/digits as answers, which are single tokens); an
8 GB GPU limits us to gemma-2-2b (a larger multilingual model would need a rented GPU).

---

## 7. Timeline (for "When")

**Done (Oct 19 to 26, 2026):**

| Date | Step |
|---|---|
| Oct 19 | First overnight run: no usable results (the model could not do the first tasks) |
| Oct 20 | Code verified against Hugging Face (same top answer on 95% of prompts); task capability study; new task suite |
| Oct 25 | Pilot run (24 min): original idea ruled out; small focused gap found |
| Oct 26 | Three main runs (6.3 h): gap is task-specific and survives a 16x wider SAE; three traps measured |

**Planned (dates are estimates; adjust to the course calendar):**

| Weeks | Step | Output |
|---|---|---|
| 1 (Oct 27 to Nov 2) | Binding test (does the missing direction match the binding mechanism? does editing along it swap objects?); sparsity sweep | English results complete |
| 2 (Nov 3 to 9) | Hindi and Hinglish templates; capability gate; SAE fidelity per language (M1) | Which languages are usable |
| 3 to 4 (Nov 10 to 23) | Causal split + gap per language (M2); cross-lingual transfer (M3); Tamil/Bengali if usable | Multilingual results |
| 5 (Nov 24 to 30) | More pairs for final numbers; extra checks | Final figures |
| 6 (Dec 1 to 7) | Report and final presentation | Submission |

---

## 8. What each document should contain

### 8.1 Proposal (`proposal.tex`, 2 to 3 pages)

- **Title, author line** (leave `\author{}` for the student to fill), date.
- **What (about 0.5 page):** problem statement in 3 to 4 sentences; the three research questions:
  (RQ1) does the tracked information live in the SAE's parts or its error; (RQ2) what exactly do SAEs miss, and
  does it survive wider SAEs; (RQ3, multilingual) do SAEs miss more in Indian languages, and is the missing piece
  shared across languages. Expected outcomes: a careful testing protocol, a measured "blind spot" (or a clean
  negative result), and the first multilingual causal audit of SAEs we know of (phrase carefully, see 9).
- **Why (about 0.5 page):** SAEs are the main interpretability tool; claims that "the error hides the computation"
  are common but untested causally; three traps show how easily such tests mislead; multilingual users deserve
  interpretability tools that work for them.
- **How (about 1 page):** model and SAEs; the four tasks (small table); the causal split, NLDR, controls (plain
  words); results so far as **preliminary findings** with **f2** (and optionally f5 or f6); the multilingual design
  (languages, translated templates, M1 to M3, datasets).
- **When (about 0.3 page):** the two timeline tables (done + planned).
- Optional: a short "Risks" paragraph (from 6).

### 8.2 Slides (`slides.tex`, 12 to 15 slides, 16:9)

| # | Title | Content | Figure |
|---|---|---|---|
| 1 | Title | Title, student, course, date | none |
| 2 | Why look inside models? | Interpretability in one line; SAEs as the main tool; the "dark matter" question | simple `x = x_hat + eps` diagram (TikZ, see 8.3) |
| 3 | Our question | RQ1 to RQ3 in plain words | none |
| 4 | How we test it | clean vs corrupt prompt; copying SAE parts vs error; NLDR in one line | TikZ diagram |
| 5 | Tasks the model can do | four tasks with examples; the first three dropped | f1 |
| 6 | Main result | SAE parts carry 74-100%; error little | f2 |
| 7 | One focused gap | object tracking: one direction closes half | f7 |
| 8 | It is task-specific | own vs other-task vs web-text vs random | f6 |
| 9 | Does a bigger SAE fix it? | 26% → 15% at 16x width; one direction still helps | f5 |
| 10 | Three traps | last-token copying; prompt words; readable ≠ used | f3 and/or f4 |
| 11 | Next: is it binding? | hypothesis + planned test (swap edit) | none |
| 12 | Multilingual extension | languages, M1 to M3, example Hindi/Hinglish prompts | none (a table) |
| 13 | Timeline | done + planned | table |
| 14 | Summary | three takeaways | none |
| (15) | Backup: caveats | section 5.3 bullets | none |

Keep one message per slide, the figure large, at most 3 bullets. Put the finding in the slide title.

### 8.3 LaTeX notes

- Figures: `\includegraphics[width=\linewidth]{../docs/handoff/figures/f2_causal_split.pdf}` (adjust the path, or
  copy `figures/` next to the .tex files). They are wide (about 2.5:1 to 3.5:1); in Beamer use `width=\textwidth`.
- Packages: `graphicx`, `booktabs`, `hyperref`, `xcolor`, `tikz`; for Devanagari/Tamil examples compile with
  **XeLaTeX** using `fontspec` + `polyglossia` (or `babel`) and a font that has the script (for example
  "Noto Sans Devanagari", "Noto Sans Tamil"); otherwise show the Hindi examples as images or in transliteration.
- Beamer: `\documentclass[aspectratio=169]{beamer}`; the `metropolis` theme looks clean (needs Fira fonts; fall
  back to `\usetheme{default}` with `\usecolortheme{dove}` if they are missing).
- Colours, to match the figures: accent blue `#2a78d6`, grey `#8a8985`.
- The `x = x_hat + eps` diagram: a box "residual signal x" splitting into "SAE parts x_hat (readable)" and "leftover
  error eps"; below it two prompts "clean → 8" and "corrupt → 10" with an arrow "copy one part at layer L".
- Build: `latexmk -xelatex proposal.tex` and `latexmk -xelatex slides.tex`.

---

## 9. Rules and references

**Rules**

- Numbers only from `data/*.csv` or section 5; round as in this file (whole percent).
- Present the multilingual work and the binding hypothesis as **planned**, never as results.
- Use plain language; every technical term defined once (section 3).
- Do **not** invent references. The works below are relevant, but the titles and authors were recalled from memory:
  look each one up and put it in `refs.bib` **only if you can confirm it**, otherwise leave it out.
  - Bricken et al. 2023 / Cunningham et al. 2023: sparse autoencoders find interpretable features in LMs.
  - Lieberum et al. 2024: Gemma Scope (the SAEs we use).
  - Marks et al. 2024: Sparse Feature Circuits (SAE error terms matter in circuits).
  - Engels et al. 2024: "Decomposing the dark matter of sparse autoencoders".
  - Feng and Steinhardt 2023/2024: how language models bind entities in context (binding IDs).
  - Wang et al. 2022: interpretability in the wild (the IOI / names task).
  - Recent work on multilingual or language-specific SAE features (search before claiming novelty).
- "First multilingual causal audit of SAEs" must be phrased as "to our knowledge" and only after a literature search.

**Acceptance checklist**

- [ ] Proposal is 2 to 3 pages, covers What / Why / How / When, mentions the multilingual extension.
- [ ] Slides: 12 to 15, 16:9, one message per slide, findings in titles.
- [ ] Every number matches `data/` (spot-check f2, f5, f6 values).
- [ ] Caveats from 5.3 appear (proposal: one short paragraph; slides: backup slide).
- [ ] Hindi/Hinglish examples marked "illustrative, to be validated".
- [ ] Both documents compile with XeLaTeX without errors.
