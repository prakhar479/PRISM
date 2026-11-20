"""Tasks: pilot pools, invariants, slots, filter flags (handoff tests 3, 12)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from harness import tasks as T
from src.data.synthetic_tasks import FallbackTokenizer, build_task_splits

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ftok():
    return FallbackTokenizer()


@pytest.mark.parametrize("task", T.PILOT_TASKS)
def test_pool_invariants_with_stand_in_tokenizer(task, ftok):
    rows = T.build_pilot_candidates(task, ftok, n_train=30, n_test=20)
    assert rows and not T.check_invariants(rows)
    for r in rows[:20]:
        assert len(r["slots"]) == r["T"] and r["slots"][-1] == "final"
        diff = [i for i, (a, b) in enumerate(zip(r["clean_ids"], r["corrupt_ids"])) if a != b]
        assert diff[0] == r["first_diff"]                         # patched positions = [first_diff, T-1]
    assert {r["split"] for r in rows} == {"train", "test"}
    tr = {r["clean_prompt"] for r in rows if r["split"] == "train"} | {r["corrupt_prompt"] for r in rows if r["split"] == "train"}
    te = {r["clean_prompt"] for r in rows if r["split"] == "test"} | {r["corrupt_prompt"] for r in rows if r["split"] == "test"}
    assert not tr & te


def test_unfiltered_pool_equals_pilot_without_filter(ftok):
    """With no model filter the pilot keeps the first n tokenized candidates: our pool order must agree."""
    cfg = T.pilot_cfg()
    rows = T.build_pilot_candidates("entity_track", ftok, cfg, n_train=30, n_test=20)
    s = T.task_settings(cfg, "entity_track")
    from src.experiments.common import task_cfg
    c = task_cfg(cfg, "entity_track")
    pil = build_task_splits("entity_track", ftok, 30 * c["candidate_multiplier"], 20 * c["candidate_multiplier"], c["seed"],
                            model=None, n_shots=c["n_shots"], answer_format=c["answer_format"], gen_kwargs=c["gen_kwargs"],
                            candidate_multiplier=1)
    for split in ("train", "test"):
        ours = [r["pair_id"] for r in rows if r["split"] == split]
        assert ours == [it.pair.pair_id for it in pil[split]][: len(ours)]


def test_slot_labels_entity_track(ftok):
    p = "The key is in box B. The cup is in box E. The ring is in box A. The key moves to box C. The cup moves to box D. The key is in box"
    ids = ftok.encode(p)
    lab = T.token_slots("entity_track", ftok, p, len(ids))
    toks = [ftok.itos[i] for i in ids]
    assert lab[toks.index(" key")] == "obj_1" and lab[toks.index(" B")] == "box_1"
    assert lab[-1] == "final" and lab[-4] == "query_obj"
    assert "move_2_box" in lab and "move_1_obj" in lab


def test_filter_flags_and_select():
    rows = [{"pair_id": f"t-test-{i}", "split": "test", "rank": i, "ans_A": 1, "ans_B": 2} for i in range(6)]
    top1_A = np.array([1, 1, 0, 1, 1, 1])
    top1_B = np.array([2, 1, 2, 2, 2, 2])
    D_A = np.array([3.0, 3.0, 3.0, 1.0, 3.0, 3.0])
    D_B = np.array([-3.0, 0.5, -3.0, -3.0, 0.0, -3.0])
    f = T.filter_flags(rows, top1_A, top1_B, D_A, D_B, "strict")
    assert f["f_A_correct"].tolist() == [1, 1, 0, 1, 1, 1]
    assert f["f_both_correct"].tolist() == [1, 0, 0, 1, 1, 1]
    assert f["f_pilot"].tolist() == [1, 0, 0, 0, 1, 1]            # strict also needs D_A > 2
    g = T.filter_flags(rows, top1_A, top1_B, D_A, D_B, "clean_only")
    assert g["f_pilot"].tolist() == [1, 1, 0, 1, 1, 1]            # A correct and D_A - D_B > 2 (pair 3: 1 - (-3) = 4)
    assert T.select(rows, f, "test", "f_pilot", 2) == [0, 4]


def test_committed_prompt_sets_hash_and_invariants():
    for task in T.PILOT_TASKS:
        rows, meta = T.load_set(ROOT / "data" / "prompt_sets" / f"{task}_pilot.jsonl.gz")
        assert meta["n"] == len(rows) and not T.check_invariants(rows)
        assert meta["tokenizer"]["name"] == "google/gemma-2-2b"


@pytest.mark.realtokenizer
def test_real_tokenizer_matches_committed_sets():
    try:
        from transformers import AutoTokenizer
        rows, meta = T.load_set(ROOT / "data" / "prompt_sets" / "ioi_control_pilot.jsonl.gz")
        tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", revision=meta["tokenizer"]["revision"])
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"real tokenizer unavailable: {type(e).__name__}")
    again = T.build_pilot_candidates("ioi_control", tok)
    assert T.set_hash(again) == meta["hash"]
