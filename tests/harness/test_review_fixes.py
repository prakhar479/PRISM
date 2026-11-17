"""Regression tests for the issues found by the 2026-11-17 code review (one test per issue)."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harness.queue import Job, Queue, commit_free, done_valid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ---- queue: .done markers are commit-stamped; stage 3 / finalize / scripts / bundle are redone after a re-tag
def test_done_markers_follow_the_commit(tmp_path):
    s3 = Job(id="E9_s3", config="c.yaml", stage="3", workers="auto")
    s1 = Job(id="E9_s1", config="c.yaml", stage="1")
    s2 = Job(id="E9_s2", config="c.yaml", stage="2", slices="auto")
    pub = Job(id="publish", kind="script", script="scripts/publish_results.py")
    f = tmp_path / "x.done"
    f.write_text(json.dumps({"commit": "old", "n": 3}))
    assert not done_valid(f, s3, 3, "new") and not done_valid(f, pub, 1, "new")       # re-tag: redo
    assert done_valid(f, s3, 3, "old")                                                  # same commit: keep
    assert done_valid(f, s1, 1, "new")                                                  # content-addressed: keep
    assert done_valid(f, s2, 3, "new") and not done_valid(f, s2, 4, "new")             # slice i/N with another N: redo
    f.write_text("2026-11-14T12:00:00+0530")                                            # legacy marker
    assert done_valid(f, s1, 1, "new") and not done_valid(f, s3, 3, "new")
    assert commit_free(Job(id="prefetch", kind="script", script="scripts/run_prefetch.py"))


# ---- scheduler: a queue paused on a full disk with nothing running gives up instead of sleeping until the time limit
def test_scheduler_gives_up_after_hold_timeout(tmp_path):
    from harness.scheduler import Scheduler, expand
    q = Queue("t", "dev", [Job(id="a", kind="script", script="x.py")])
    units = expand(q, 1)
    logs = []
    sch = Scheduler(q, units, ["0"], tmp_path, lambda u: ["true"], {}, {}, tmp_path, log=logs.append, poll_s=0.01,
                    hold=lambda: "disk full", hold_timeout_s=0.05)
    import time
    t = time.time()
    sch.run(lambda: None)
    assert time.time() - t < 30 and units[0].status == "skipped" and "gave up" in units[0].note


# ---- preflight: a queue of script jobs only (release 3) takes the model revision from its prefetch block
def test_preflight_revision_for_script_only_queue():
    pf = load_script("run_preflight")
    assert pf.queue_revision(Queue.load(ROOT / "queue" / "release3-server.yaml")) == "c5ebcd40d208330abc697524c919956e692655cf"
    assert pf.queue_revision(Queue.load(ROOT / "queue" / "complete-server.yaml")) == "c5ebcd40d208330abc697524c919956e692655cf"


# ---- publish: exactly the runs a queue produces, at any commit
def test_publish_scope_follows_the_queue():
    br = load_script("bundle_results")
    sc = br.queue_scope(ROOT / "queue" / "complete-server.yaml")
    assert ("E12", "fidelity_by_language") in sc and ("E13", "behaviour_by_language") in sc and ("E9", "illusion_checks_das") in sc
    assert ("E4", "width_l0_matched") not in sc                                    # release 2's E4 is not republished


# ---- cache: non-blocking writer lock
def test_try_writer_lock_reports_a_held_lock(tmp_path):
    from harness.cache import Cache
    e = Cache(tmp_path).entry("das", k=1)
    with e.writer_lock():
        with e.try_writer_lock() as held:
            assert held is False                                                   # another open file description holds it
    with e.try_writer_lock() as held:
        assert held is True


# ---- runner: finalize never trains DAS
def test_finalize_refuses_to_train_das(tmp_path):
    from harness.cache import Cache
    from harness.run import Runner
    fake = types.SimpleNamespace(tiny=True, smoke=True, finalizing=True, cache=Cache(tmp_path), cfg=types.SimpleNamespace(seed=0),
                                 spec=types.SimpleNamespace(key=lambda: {"m": 1}), DAS_VERSION=1,
                                 spans_entry=lambda t, l: Cache(tmp_path).entry("spans", t=t, l=l))
    fs = types.SimpleNamespace(digest="abc")
    with pytest.raises(RuntimeError, match="never during finalize"):
        Runner.das_direction(fake, "add_chain", 18, [], [0, 1], None, fs, 2)


# ---- analysis: spec tables carry the config role; E9 / Q2 decision columns are NA when not evaluable
def test_config_role_and_na_decisions():
    from harness import final as FN
    m = {"config": {"cells": [{"task": "add_chain", "layer": 18, "sae": {"width": "16k"}, "role": "confirmatory"},
                              {"task": "entity_track", "layer": 12, "sae": {"width": "16k"}, "role": "exploratory"}]}}
    assert FN.config_role(m, "add_chain", 18, "L18_16k_l074", "pilot", "pilot") == "confirmatory"
    assert FN.config_role(m, "entity_track", 12, "L12_16k_l082", "pilot", "pilot") == "exploratory"   # not E8's default
    mf = load_script("make_final")
    t = pd.DataFrame({"run_hash": "h", "task": "add_chain", "layer": 18, "sae": "L18_16k_l074", "fit_families": "pilot",
                      "family": "pilot", "filter": "f_pilot", "estimator": ["pca_pooled", "das1", "pca_pooled"], "r": [1, 1, 2],
                      "condition": "full_along_U", "flip_rate": [0.9, 0.9, 0.9], "flip_rate_lo": [0.8, 0.8, 0.8],
                      "random_p95": [0.5, 0.5, np.nan], "random_n": [20, 20, 0], "das_train_acc": [np.nan, 0.2, np.nan]})
    out = mf.e9_rules(t, [], [])
    eff = dict(zip(zip(out.estimator, out.r), out.effective))
    assert eff[("pca_pooled", 1)] is True or eff[("pca_pooled", 1)] == True                       # noqa: E712
    assert pd.isna(eff[("das1", 1)])                     # invalid DAS fit (acc 0.2 < 0.5): not evidence
    assert pd.isna(eff[("pca_pooled", 2)])               # no random null: not evaluable


def test_e8_correct_ranks_per_family_and_bh_within_task():
    from harness import e8 as E
    base = dict(layer=12, sae="L12_16k_l082", fit_families="ot_en", filter="f_pilot", metric="flip_rate", estimator="pca_pooled",
                r=1, n_random_seeds=20, controls="random_p95", p_margin=0.01, base_gap_resolved=True, role="exploratory")
    rows = [dict(base, task="entity_track", family=f) for f in ("ot_en", "ot_hi", "ot_fr")] + \
           [dict(base, task="add_chain", family="rs_en", sae="L18_16k_l074", layer=18)]
    out = E.correct(pd.DataFrame(rows))
    assert out.canonical.all()                           # one canonical row per evaluation family
    assert out[out.task == "entity_track"].p_bh.notna().all()
