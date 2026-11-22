"""Typed experiment configs (configs/E*.yaml). One experiment type, `patch_grid`, covers E0-E2 and E6-E7: cells
(task x layer x SAE) x conditions x direction (estimator, source, r, seed), evaluated on test pairs selected by one or
more filters, with every fitted object fitted on train pairs selected by `fit_filter`."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from harness.conditions import CONDITIONS, is_steer
from harness.tasks import FILTERS

ESTIMATORS = ("pca_pooled", "answer_sep", "pca_pos_centered", "pca_slot_centered_pairdiff", "das1", "das2", "das4", "das8")
U_CONDITIONS = ("repair", "repair_mean", "feat_A_along_U", "full_along_U", "steer_x1", "steer_x2", "steer_x4",
                "resampU_sameAns", "resampU_diffAns", "feat_codes_along_U")
CODE_CONDITIONS = ("feat_codes_along_U",)        # need stage-2 active codes of A and B (P0 write-back operator)
DAS_DEFAULTS = {"epochs": 3, "lr": 1e-2}          # E9/E10/E16 rotations (cache keys unchanged for plain das<k>)
_DAS_RE = re.compile(r"^das(\d+)(?:_e(\d+))?(?:_lr([0-9.]+e-?\d+|[0-9.]+))?(?:_s(\d+))?$")


def parse_das(name: str) -> Optional[dict]:
    """das<k>[_e<epochs>][_lr<lr>][_s<seed>] -> {k, epochs, lr, seed (None = the config seed)}; None if not a DAS estimator.
    Plain das<k> is the E9 setting (3 epochs, lr 1e-2, config seed). The variants (E17) only change training."""
    m = _DAS_RE.match(name)
    if not m or int(m.group(1)) < 1:
        return None
    return {"k": int(m.group(1)), "epochs": int(m.group(2)) if m.group(2) else DAS_DEFAULTS["epochs"],
            "lr": float(m.group(3)) if m.group(3) else DAS_DEFAULTS["lr"], "seed": int(m.group(4)) if m.group(4) else None}


DAS2_OBJECTIVES = ("ce", "ld", "kl")
DAS2_DEFAULTS = {"epochs": 10, "lr": 1e-2, "holdout": 0.2, "step_tokens": 1024}
_DAS2_RE = re.compile(r"^das(\d+)v2_(ce|ld|kl)(?:_e(\d+))?(?:_lr([0-9.]+e-?\d+|[0-9.]+))?(?:_s(\d+))?$")


def parse_das2(name: str) -> Optional[dict]:
    """Pilot (P0) DAS v2: das<k>v2_<objective>[_e<epochs>][_lr<lr>][_s<seed>].

    objective: ce = cross-entropy toward A's answer (v1's loss); ld = squared error of the per-pair normalised logit difference
    against the FULL patch (NLDR_i -> 1); kl = KL(full-patch || patched) over the pair's candidate answers plus one 'rest' bucket.
    ld and kl are bounded by the full-state effect, so a fitted subspace cannot score above the full patch by construction
    ("Whose Effect Is It", 2026). v2 also holds out a validation share of the train pairs (validity is judged there), draws
    its initial rotation from (seed, task, layer, train set) so different cells never share a start, and trains on
    length-grouped mini-batches."""
    m = _DAS2_RE.match(name)
    if not m or int(m.group(1)) < 1:
        return None
    return {"k": int(m.group(1)), "objective": m.group(2),
            "epochs": int(m.group(3)) if m.group(3) else DAS2_DEFAULTS["epochs"],
            "lr": float(m.group(4)) if m.group(4) else DAS2_DEFAULTS["lr"],
            "seed": int(m.group(5)) if m.group(5) else None, "version": 2,
            "holdout": DAS2_DEFAULTS["holdout"], "step_tokens": DAS2_DEFAULTS["step_tokens"]}


def das_spec(name: str) -> Optional[dict]:
    """v1 (parse_das, with version 1) or v2 (parse_das2) spec of a DAS estimator; None otherwise."""
    d = parse_das(name)
    if d is not None:
        return {**d, "version": 1, "objective": "ce"}
    return parse_das2(name)


def das_rank(name: str) -> Optional[int]:
    d = das_spec(name)
    return d["k"] if d else None


# Estimators that need the model (gradients at the patched layer) and are cached like DAS rotations (P7)
GRAD_ESTIMATORS = ("grad_pooled", "grad_err")
# Estimators fitted on task labels (P5): running-sum value -> ridge line and Fourier planes of the value
LABEL_ESTIMATORS = ("ridge_value", "fourier_T2", "fourier_T5", "fourier_T10")
NEW_ESTIMATORS = GRAD_ESTIMATORS + LABEL_ESTIMATORS


def is_estimator(name: str) -> bool:
    return name in ESTIMATORS or name in NEW_ESTIMATORS or das_spec(name) is not None


def is_u_condition(name: str) -> bool:
    return name in U_CONDITIONS or is_steer(name)
ROLES = ("confirmatory", "exploratory")


@dataclass
class SAESpec:
    width: str = "16k"
    l0: Optional[int] = None          # explicit average_l0 variant
    rule: str = "canonical"           # canonical (L0 closest to 100 = the pilot's choice) | explicit
    # breadth pilot (harness/dicts.py): a registry dictionary (configs/dict_registry.yaml) or 'trained:<name>'
    # (scripts/train_dictionary.py), and/or an add-on fitted on the generic set ('lin' | 'pcx<r>')
    dict: Optional[str] = None
    addon: Optional[str] = None

    def tag(self) -> tuple:
        return (self.width, self.l0, self.dict or "", self.addon or "")

    def resolve(self, layer: int, inventory: dict) -> int:
        from harness.sae import canonical_l0

        variants = inventory[str(layer)][self.width]
        if self.l0 is not None:
            if self.l0 not in variants:
                raise ValueError(f"layer {layer} {self.width}: l0 {self.l0} not in {variants}")
            return self.l0
        return canonical_l0(variants)


@dataclass
class Cell:
    task: str
    layer: int
    sae: SAESpec = field(default_factory=SAESpec)
    role: str = "exploratory"
    # E3 (fresh templates): family groups ('pilot', 'fresh' or family ids from harness.families). fit_families: what
    # train pairs fitted objects (directions, mean errors, covariance noise) use; eval_families: the test pairs
    # (n_test per family). None = every row of the prompt set (release-1 behaviour; refused for sets with fresh rows).
    fit_families: Optional[List[str]] = None
    eval_families: Optional[List[str]] = None

    def fit_tag(self) -> str:
        return "+".join(self.fit_families) if self.fit_families else "pilot"

    def tag(self) -> str:
        """'' for release-1 cells; else a suffix that keeps cells over the same (task, layer, SAE) apart."""
        if not (self.fit_families or self.eval_families):
            return ""
        return f"__fit-{self.fit_tag()}__eval-{'+'.join(self.eval_families or ['all'])}"


@dataclass
class DirSpec:
    conditions: List[str] = field(default_factory=lambda: ["repair"])
    ranks: List[int] = field(default_factory=lambda: [0, 1, 2])
    estimators: List[str] = field(default_factory=lambda: ["pca_pooled", "answer_sep"])
    sources: List[str] = field(default_factory=lambda: ["same_task"])   # same_task | other_task | generic | layer:<L>
    random_seeds: int = 0             # Haar-random U per rank (null distribution)
    r_max: int = 32


@dataclass
class SubthreshSpec:
    """E5: add back A's JumpReLU sub-threshold features (entries with alpha*theta <= z < theta, z > 0).

    alphas: grid; each alpha gives `subthresh` with every slot error. slot_errors: 'B' (x_hat_A + add-back + eps_B)
    and/or 'meanPos' (x_hat_A + add-back + eps_bar_pos). greedy_ks: sizes of a greedily selected latent set, fitted on
    train pairs at greedy_alpha (default min(alphas)) and evaluated at every alpha; [] = off."""
    alphas: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.8, 0.9, 0.95])
    slot_errors: List[str] = field(default_factory=lambda: ["B", "meanPos"])
    greedy_ks: List[int] = field(default_factory=list)
    greedy_alpha: Optional[float] = None

    @property
    def g_alpha(self) -> float:
        return self.greedy_alpha if self.greedy_alpha is not None else min(self.alphas)

    @property
    def alpha_min(self) -> float:
        """Smallest alpha any condition needs: stage 2 keeps sparse entries down to it."""
        return min(self.alphas + [self.g_alpha])


@dataclass
class ExpConfig:
    exp_id: str
    name: str
    description: str = ""
    kind: str = "patch_grid"
    model: str = "gemma-2-2b"         # harness.model.MODEL_REPOS key (breadth pilot P8: gemma-3-1b-pt)
    model_revision: Optional[str] = None
    prompt_sets: Dict[str, str] = field(default_factory=dict)       # task -> path relative to data_root
    cells: List[Cell] = field(default_factory=list)
    n_train: int = 300
    n_test: int = 150
    filters: List[str] = field(default_factory=lambda: ["f_pilot"])
    fit_filter: str = "f_pilot"
    conditions: List[str] = field(default_factory=list)
    noise_seeds: List[int] = field(default_factory=lambda: [0])
    directions: Optional[DirSpec] = None
    generic: Optional[dict] = None    # {set: path relative to data_root, skip_positions: 4}
    generic_l0: bool = False          # E4: report every cell SAE's realized L0 / FVU on the generic set
    subthresh: Optional[SubthreshSpec] = None
    bootstrap: int = 1000            # run-side quick summary; DEV analysis uses 10 000
    seed: int = 0
    smoke_pairs: int = 10
    pilot_reference: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ load / validate
    @classmethod
    def load(cls, path) -> "ExpConfig":
        d = yaml.safe_load(Path(path).read_text())
        cells = [Cell(**{**c, "sae": SAESpec(**c.get("sae", {}))}) for c in d.pop("cells", [])]
        dirs = d.pop("directions", None)
        sub = d.pop("subthresh", None)
        cfg = cls(**d, cells=cells, directions=DirSpec(**dirs) if dirs else None,
                  subthresh=SubthreshSpec(**sub) if sub else None)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        errs = []
        if self.kind != "patch_grid":
            errs.append(f"unknown kind {self.kind}")
        for c in self.cells:
            if c.task not in self.prompt_sets:
                errs.append(f"cell task {c.task} has no prompt set")
            if c.role not in ROLES:
                errs.append(f"cell role {c.role}")
            if bool(c.fit_families) != bool(c.eval_families):
                errs.append(f"cell {c.task} L{c.layer}: set both fit_families and eval_families, or neither")
            for fams in (c.fit_families, c.eval_families):
                if fams:
                    from harness.families import expand
                    try:
                        expand(c.task, fams)
                    except ValueError as e:
                        errs.append(str(e))
        tags = [(c.task, c.layer, c.sae.tag(), c.tag()) for c in self.cells]
        if len(set(tags)) != len(tags):
            errs.append("two cells share (task, layer, SAE, fit/eval families)")
        if any(c.sae.addon for c in self.cells) and not self.generic:
            errs.append("an SAE add-on is fitted on the generic set: add a `generic:` block")
        for f in [*self.filters, self.fit_filter]:
            if f not in FILTERS:
                errs.append(f"unknown filter {f}")
        for n in self.conditions:
            if n not in CONDITIONS or is_u_condition(n) or n == "subthresh":
                errs.append(f"condition {n} is unknown or belongs under directions")
        if self.directions:
            for n in self.directions.conditions:
                if not is_u_condition(n):
                    errs.append(f"direction condition {n}")
            for e in self.directions.estimators:
                if not is_estimator(e):
                    errs.append(f"estimator {e}")
            for s in self.directions.sources:
                if s not in ("same_task", "other_task", "generic") and not s.startswith("layer:"):
                    errs.append(f"source {s}")
            if "generic" in self.directions.sources and not self.generic:
                errs.append("source generic needs a `generic:` block")
            if max(self.directions.ranks) > self.directions.r_max:
                errs.append("rank above r_max")
            for e in self.directions.estimators:
                if e in LABEL_ESTIMATORS and any(c.task != "add_chain" for c in self.cells):
                    errs.append(f"estimator {e} needs numeric states (add_chain cells only)")
        if self.generic_l0 and not self.generic:
            errs.append("generic_l0 needs a `generic:` block")
        if self.subthresh:
            sp = self.subthresh
            if not sp.alphas or any(not 0 < a <= 1 for a in sp.alphas + [sp.g_alpha]):
                errs.append("subthresh alphas must lie in (0, 1]")
            if not sp.slot_errors or not set(sp.slot_errors) <= {"B", "meanPos"}:
                errs.append("subthresh slot_errors must be a non-empty subset of [B, meanPos]")
            if any(k < 1 or k > 10 for k in sp.greedy_ks):
                errs.append("subthresh greedy_ks must lie in 1..10")
        if errs:
            raise ValueError(f"{self.exp_id}: " + "; ".join(errs))

    def to_dict(self) -> dict:
        return asdict(self)

    def layers_by_task(self) -> Dict[str, List[int]]:
        out: Dict[str, set] = {}
        for c in self.cells:
            out.setdefault(c.task, set()).add(c.layer)
        if self.directions:
            for s in self.directions.sources:
                if s.startswith("layer:"):
                    for t in out:
                        out[t].add(int(s.split(":")[1]))
            if "other_task" in self.directions.sources:          # other tasks need the same layers
                allL = set().union(*out.values())
                out = {t: set(allL) for t in out}
        return {t: sorted(v) for t, v in out.items()}
