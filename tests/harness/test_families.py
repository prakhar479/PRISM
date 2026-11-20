"""E3 fresh template families: generators, slots, family-aware selection, committed sets, config, tiny runner."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from harness import families as F
from harness import tasks as T
from harness.config import ExpConfig
from src.data import synthetic_tasks as st

ROOT = Path(__file__).resolve().parents[2]
PILOT_SLOTS = {"entity_track": {"obj_1", "box_1", "obj_2", "box_2", "obj_3", "box_3", "move_1_obj", "move_1_box",
                                "move_2_obj", "move_2_box", "query_obj"},
               "add_chain": {"start_num", "add_1", "add_2", "add_3", "query"}}


@pytest.mark.parametrize("task,family", [(t, f) for t in F.FAMILIES for f in F.families_for(t)])
def test_family_pairs_valid_and_disjoint_from_pilot(task, family):
    s = T.task_settings(T.pilot_cfg(), task)
    pairs = F.generate_pairs(task, family, 200, 1, gen_kwargs=s["gen_kwargs"])
    assert len(pairs) == 200 and len({p.pair_id for p in pairs}) == 200
    pilot_prompts = {p.clean_prompt for p in st.generate_pairs(task, 200, 1, gen_kwargs=s["gen_kwargs"])}
    for p in pairs:
        assert p.clean_target_str != p.corrupt_target_str and p.meta["family"] == family
        assert p.clean_prompt not in pilot_prompts
        assert {n for n, _, _ in F.slot_spans(task, family, p.clean_prompt)} == PILOT_SLOTS[task]
        if task == "entity_track":
            assert p.clean_target_str.strip() in F.OT_FAMILIES[family].labels
            assert not set(p.clean_prompt.replace(".", " ").split()) & set(st.OBJS)      # no pilot nouns
        else:
            assert p.clean_state_id == int(p.clean_target_str) <= 9


@pytest.mark.parametrize("family", list(F.OT_FAMILIES))
def test_ot_answer_is_the_queried_objects_final_box(family):
    f = F.OT_FAMILIES[family]
    rng = np.random.default_rng(0)
    n = 0
    while n < 50:
        p = F._gen_ot(f, rng)
        if p is None:
            continue
        n += 1
        for prompt, target in ((p.clean_prompt, p.clean_target_str), (p.corrupt_prompt, p.corrupt_target_str)):
            q = re.search(f.query_re, prompt).group(1)
            loc = dict(re.findall(f.init_re, prompt))
            for o, b in re.findall(f.move_re, prompt):
                loc[o] = b
            assert f" {loc[q]}" == target


def test_select_groups_pilot_unchanged_round_robin_and_per_family():
    rows = ([{"split": "train", "rank": k, "family": "pilot_calc" if k % 2 else "pilot_story"} for k in range(6)]
            + [{"split": "train", "rank": k, "family": fam} for fam in ("rs_plus", "rs_story") for k in range(5)])
    fl = {"f_pilot": np.ones(len(rows), bool)}
    fl["f_pilot"][7] = False                                                 # rs_plus rank 1 fails the filter
    assert T.select(rows[:6], {"f_pilot": fl["f_pilot"][:6]}, "train", "f_pilot", 4) == [0, 1, 2, 3]
    assert T.select(rows, fl, "train", "f_pilot", 4, groups=["pilot"]) == [0, 1, 2, 3]
    rr = T.select(rows, fl, "train", "f_pilot", 4, groups=["rs_plus", "rs_story"])
    assert [rows[i]["family"] for i in rr] == ["rs_plus", "rs_story", "rs_plus", "rs_story"]
    assert [rows[i]["rank"] for i in rr] == [0, 0, 2, 1]
    pg = T.select(rows, fl, "train", "f_pilot", 3, groups=["rs_plus", "rs_story"], per_group=True)
    assert len(pg) == 6 and sum(rows[i]["family"] == "rs_plus" for i in pg) == 3


def test_expand_and_config_validation(tmp_path):
    assert F.expand("entity_track", ["fresh"]) == F.families_for("entity_track")
    assert F.expand("add_chain", ["pilot", "rs_plus"]) == ["pilot", "rs_plus"]
    with pytest.raises(ValueError):
        F.expand("entity_track", ["rs_plus"])
    d = yaml.safe_load((ROOT / "configs" / "experiments" / "E3.yaml").read_text())
    d["cells"][0]["eval_families"] = None
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="both fit_families and eval_families"):
        ExpConfig.load(p)


def test_committed_e3_sets_contain_the_pilot_set_unchanged():
    for task in F.FAMILIES:
        rows, meta = T.load_set(ROOT / "data" / "prompt_sets" / f"{task}_e3.jsonl.gz")
        pilot, _ = T.load_set(ROOT / "data" / "prompt_sets" / f"{task}_pilot.jsonl.gz")
        assert [r for r in rows if F.is_pilot(r["family"])] == pilot
        assert not T.check_invariants(rows)
        for fam in F.families_for(task):
            fr = [r for r in rows if r["family"] == fam]
            tr = {r["clean_prompt"] for r in fr if r["split"] == "train"} | {r["corrupt_prompt"] for r in fr if r["split"] == "train"}
            te = {r["clean_prompt"] for r in fr if r["split"] == "test"}
            assert tr and te and not tr & te                                                   # prompt-disjoint
            assert all(PILOT_SLOTS[task] <= set(r["slots"]) for r in fr)


@pytest.mark.slow
def test_tiny_e3_runs_with_fit_and_eval_families(tmp_path):
    from harness import run as R
    from harness.schema import validate_per_pair

    d = yaml.safe_load((ROOT / "configs" / "experiments" / "E3.yaml").read_text())
    d.update(name="tiny_test", cells=[c for c in d["cells"] if c["task"] == "add_chain"][:2], bootstrap=20,
             smoke_pairs=4, conditions=["feat_A_errB", "feat_A_meanPos"], noise_seeds=[0])
    d["prompt_sets"] = {"add_chain": d["prompt_sets"]["add_chain"]}
    d["directions"].update(conditions=["repair"], ranks=[0, 1], random_seeds=1)
    p = tmp_path / "E3_tiny.yaml"
    p.write_text(yaml.safe_dump(d))
    assert R.main(["--config", str(p), "--profile", "dev", "--tiny", "--smoke"]) == 0
    run_dir = sorted((ROOT / "tests" / "_tiny_runs" / "E3").glob("*/manifest.json"), key=lambda q: q.stat().st_mtime)[-1].parent
    pp = pd.read_parquet(run_dir / "per_pair.parquet")
    assert not validate_per_pair(pp)
    assert set(pp.fit_families) == {"pilot", "fresh"}
    assert set(pp.family) == set(F.families_for("add_chain"))                     # both cells evaluate fresh
    per = pp[pp.condition == "clean_B"].groupby(["fit_families", "family"]).pair_id.nunique()
    assert (per == 4).all()                                                         # smoke_pairs per family
    s = json.loads((run_dir / "summary.json").read_text())
    assert all(c["max_abs_dD"] < 1e-3 for c in s["hook_checks"])
    assert {q.name.split("__fit-")[1] for q in (run_dir / "directions").glob("*__fit-*.json")} == {"pilot.json", "fresh.json"}
