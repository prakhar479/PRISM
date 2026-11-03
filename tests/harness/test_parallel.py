"""Multi-process safety: cache writer locks, atomic writes, crash recovery (plan step 1-2)."""
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import torch

from harness.cache import Cache, atomic_write_text, lock_many


def _fill(root: str, tag: str, delay: float, log: str) -> None:
    """Fill a 4-shard entry under its writer lock, recording when this process held the lock."""
    c = Cache(Path(root))
    e = c.entry("demo", tag=tag)
    with e.writer_lock():
        e.open()
        with open(log, "a") as fh:
            fh.write(f"start {os.getpid()} {time.time()}\n")
        for i in range(4):
            if e.shard_done(i):
                continue
            time.sleep(delay)
            e.write_shard(i, {"x": torch.arange(10, dtype=torch.float32) + i})
        if not e.complete:
            e.mark_complete({"by": os.getpid()})
        with open(log, "a") as fh:
            fh.write(f"end {os.getpid()} {time.time()}\n")


def test_two_writers_on_one_entry_are_serialised_and_agree(tmp_path):
    ctx = mp.get_context("spawn")
    log = tmp_path / "log.txt"
    ps = [ctx.Process(target=_fill, args=(str(tmp_path / "c"), "same", 0.15, str(log))) for _ in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(60)
        assert p.exitcode == 0
    ev = [l.split() for l in log.read_text().splitlines()]
    assert [k for k, _, _ in ev] == ["start", "end", "start", "end"]          # never interleaved
    e = Cache(tmp_path / "c").entry("demo", tag="same")
    assert e.complete and e.shards() == [0, 1, 2, 3]
    assert torch.equal(e.read_all(["x"])["x"], torch.cat([torch.arange(10.) + i for i in range(4)]))
    assert not list(e.dir.glob("*.tmp*"))


def test_writer_killed_mid_fill_then_refill_completes(tmp_path):
    ctx = mp.get_context("spawn")
    log = tmp_path / "log.txt"
    p = ctx.Process(target=_fill, args=(str(tmp_path / "c"), "crash", 1.0, str(log)))
    p.start()
    e = Cache(tmp_path / "c").entry("demo", tag="crash")
    t0 = time.time()
    while not e.shard_done(0) and time.time() - t0 < 30:
        time.sleep(0.05)
    os.kill(p.pid, signal.SIGKILL)                       # dies holding the lock, mid-fill
    p.join(10)
    assert not e.complete and e.shard_done(0)
    _fill(str(tmp_path / "c"), "crash", 0.0, str(log))  # the lock died with the process; refill resumes
    assert e.complete and e.shards() == [0, 1, 2, 3]
    assert torch.equal(e.read_shard(3)["x"], torch.arange(10.) + 3)


def test_atomic_write_and_lock_many_order(tmp_path):
    p = tmp_path / "f.json"
    atomic_write_text(p, "{}")
    assert p.read_text() == "{}" and not list(tmp_path.glob("*.tmp.*"))
    c = Cache(tmp_path / "c")
    a, b = c.entry("k", n=1), c.entry("k", n=2)
    with lock_many([b, a, a]):                           # duplicates and any order are fine
        assert (a.dir / ".lock").exists() and (b.dir / ".lock").exists()


# --------------------------------------------------------------------------- stage-3 workers + finalize (plan step 5)
import json as _json
import subprocess
import sys

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _cfg(tmp_path, name):
    d = yaml.safe_load((ROOT / "configs" / "experiments" / "E2.yaml").read_text())
    d.update(name=name, cells=d["cells"][:1], bootstrap=20, smoke_pairs=6,
             conditions=["feat_A_errB", "feat_A_meanPos", "feat_B_noise"], noise_seeds=[0, 1])
    d["directions"].update(conditions=["repair"], ranks=[0, 1, 2], random_seeds=3)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def _run(cfg, *extra, env=None):
    cmd = [sys.executable, "-m", "harness.run", "--config", str(cfg), "--profile", "dev", "--tiny", "--smoke", *extra]
    return subprocess.Popen(cmd, cwd=ROOT, env={**__import__("os").environ, **(env or {})},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _run_dir(name):
    hits = []
    for m in (ROOT / "tests" / "_tiny_runs" / "E2").glob("*/manifest*.json"):
        if _json.loads(m.read_text())["config"]["name"] == name:
            hits.append(m.parent)
    assert len(set(hits)) == 1, hits
    return hits[0]


KEY = ["pair_id", "task", "layer", "sae", "condition", "estimator", "source", "r", "seed"]


@pytest.mark.slow
def test_concurrent_workers_plus_finalize_equal_a_sequential_run(tmp_path):
    seq, par = _cfg(tmp_path, "par_test_seq"), _cfg(tmp_path, "par_test_par")
    p = _run(seq)
    assert p.wait(600) == 0, p.stdout.read()
    for stage in ("1", "2"):                                   # warm-up, as the scheduler does
        q = _run(par, "--stage", stage)
        assert q.wait(600) == 0, q.stdout.read()
    env = {"HL_CONCURRENT_WORKERS": "3"}
    ws = [_run(par, "--stage", "3", "--worker", f"w{i}", env=env) for i in range(3)]
    for w in ws:
        assert w.wait(600) == 0, w.stdout.read()
    f = _run(par, "--finalize-only")
    assert f.wait(600) == 0, f.stdout.read()
    a = pd.read_parquet(_run_dir("par_test_seq") / "per_pair.parquet").sort_values(KEY).reset_index(drop=True)
    rd = _run_dir("par_test_par")
    b = pd.read_parquet(rd / "per_pair.parquet").sort_values(KEY).reset_index(drop=True)
    assert len(a) == len(b) and not b.duplicated(KEY).any()
    cols = ["logit_A", "logit_B", "lse", "top1_id"]
    pd.testing.assert_frame_equal(a[KEY + cols], b[KEY + cols])
    dones = [_json.loads(p.read_text()) for p in (rd / "parts").glob("*__0*.done")]
    assert {d["worker"] for d in dones} >= {"w0"} and not list((rd / "parts").glob("*.claim"))
    m = _json.loads((rd / "manifest.json").read_text())
    assert m["status"] == "complete" and m["cells"]                      # only finalize writes manifest.json
    assert {p.name for p in rd.glob("manifest.w*.json")} == {f"manifest.w{w}.json" for w in ("w0", "w1", "w2")}
    assert all(_json.loads((rd / f"manifest.w{w}.json").read_text())["status"] == "worker_done" for w in ("w0", "w1", "w2"))


@pytest.mark.slow
def test_finalize_refuses_unfinished_units(tmp_path):
    par = _cfg(tmp_path, "par_test_unfinished")
    for stage in ("1", "2"):
        q = _run(par, "--stage", stage)
        assert q.wait(600) == 0, q.stdout.read()
    f = _run(par, "--finalize-only")
    out = f.stdout.read()
    assert f.wait(600) == 1 and "not done" in out


def _hold_ram(root, nbytes, seconds, log):
    import os as _os
    _os.environ["HL_CONCURRENT_WORKERS"] = "2"
    from harness import parallel as P
    with P.reserve_ram(Path(root), nbytes, "test", poll_s=0.05):
        with open(log, "a") as fh:
            fh.write(f"in {time.time()}\n")
        time.sleep(seconds)
        with open(log, "a") as fh:
            fh.write(f"out {time.time()}\n")


def test_claims_exclusive_and_dead_claims_taken_over(tmp_path):
    from harness import parallel as P
    c = tmp_path / "u.claim"
    assert P.claim(c, "a") and not P.claim(c, "b")             # the live owner (this process) keeps it
    c.write_text(_json.dumps({"worker": "x", "pid": 2 ** 22 + 12345, "host": P.HOST, "t": 0}))   # a dead pid
    assert P.stale(c) and P.claim(c, "b") and _json.loads(c.read_text())["worker"] == "b"
    c.write_text(_json.dumps({"worker": "x", "pid": 1, "host": "another-host", "t": 0}))
    assert not P.stale(c) and not P.claim(c, "b")             # never steal another machine's claim
    (tmp_path / "d.claim").write_text(_json.dumps({"pid": 2 ** 22 + 1, "host": P.HOST}))
    assert P.clear_stale_claims(tmp_path) == 1


def test_ram_budget_makes_a_second_reservation_wait(tmp_path, monkeypatch):
    monkeypatch.setenv("HL_RAM_BUDGET_BYTES", str(100))
    ctx = mp.get_context("spawn")
    log = tmp_path / "ram.txt"
    ps = [ctx.Process(target=_hold_ram, args=(str(tmp_path), 70, 0.6, str(log))) for _ in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(60)
        assert p.exitcode == 0
    ev = [l.split()[0] for l in log.read_text().splitlines()]
    assert ev == ["in", "out", "in", "out"]                     # 70 + 70 > 100: never held together
    assert not list((tmp_path / ".ram").glob("*.json"))
