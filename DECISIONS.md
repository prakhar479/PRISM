# DECISIONS (harness)

Dated log of non-trivial choices. Pilot run verdicts live in `results/*/DECISIONS.md`; this file is about the harness.

## 2026-10-30 (Phase 0)

- **Pilot code = `src/` at HEAD**, identical to `430aa27`, which produced `results/paper_20261026_001341`. It is frozen
  as the reference, and the new code goes in `harness/` (proposed; see docs/IMPLEMENTATION_PLAN.md).
- **Hook site recorded:** `blocks.{L}.hook_resid_post` (TransformerLens, `from_pretrained_no_processing`). Model
  runs in bf16 on CUDA. SAE weights are fp32 below 524k. D uses post-softcap float32 last-position logits.
- **Golden export** (`scripts/export_golden.py`):
  - The 16 golden test pairs are drawn with `np.random.default_rng(20261030)` from the pilot test order.
  - Sparse z keeps entries with `z > 0` and `z >= 0.5·threshold` (E5's smallest α is 0.5).
  - B's residual is stored only from `first_diff` on, because earlier positions equal A's; the stored
    `prefix_A_minus_B_maxabs` checks this.
  - Condition logits are stored as candidate-set logits + logsumexp + top-50. Full-vocab logits are kept only for
    clean A/B, to stay under 200 MB.
  - Tolerances used there are provisional and become central in `harness/tolerances.py` in Phase 1:
    - D vs pilot: 0.05;
    - batch-1 vs batched D: 0.25 (bf16);
    - U columns: |cos| > 0.999;
    - NLDR: ±1 point.
- **HF agreement** (`scripts/hf_agreement.py`): "near-tie" = disagreement whose top-1 margin is < 0.25 logits in
  either model. HF runs eager attention, same dtype, batch 1, `logits_to_keep=1`.
