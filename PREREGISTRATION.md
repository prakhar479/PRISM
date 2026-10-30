# Pre-registration (frozen on <date the humans confirm>)

Confirmatory cells: object tracking, layer 12, Gemma Scope 16k (pilot L0 variant);
running sum, layer 18, Gemma Scope 16k (pilot L0 variant).
Everything else is exploratory.

Primary metric: flip rate (top-1 = A's answer on runs of B, among pairs where clean B
has top-1 = B's answer). Secondary: normalized probability recovery; NLDR (pilot metric).

RQ1 (existence, gate due Nov 3, 2026): GO if, on fresh template families, the flip-rate
shortfall of feat_A_meanPos relative to full_A is >= 10 points with a 95% CI excluding 0,
for the confirmatory cell in question. Otherwise NO-GO for that cell.

Repair specificity (carried over from the proposal): same-task directions close >= 30
points more of the gap than any other source (other task, generic text, random 95th
percentile), at r <= 2, under the primary metric.

Width persistence: evaluated at matched realized L0 (within 25%) between 16k and the
widest available SAE; reported as a ratio with CI; the pilot's 60% threshold is applied
to the CI, and a CI straddling 60% is reported as inconclusive.

Q2 steering (proposal criterion): >= 50% of pairs move to the other box when steering
along the direction; matched-norm random directions <= 5%. Run only after RQ1 = GO and
after illusion checks (E9) are reported.

Multiple comparisons: Holm across confirmatory cells; BH (q = 0.05) across exploratory
cells within each experiment family.

Multilingual: task-gap comparisons only on validated templates and only where task
accuracy is within 10 points of English; compare normalized gaps.
