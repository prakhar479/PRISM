"""Tiered cache: fast root first, spill to the bulk root when the fast one is nearly full; one entry, one tier."""
from pathlib import Path

import harness.cache as hc
from harness import storage
from harness.cache import Cache
from harness.profile import load_profile


def fake_free(monkeypatch, table):
    monkeypatch.setattr(hc, "free_gib", lambda p: next(v for k, v in table.items() if str(p).startswith(str(k))))


def test_new_entries_use_fast_root_until_it_is_low_then_spill(tmp_path, monkeypatch):
    fast, bulk = tmp_path / "fast", tmp_path / "bulk"
    free = {fast: 50.0, bulk: 500.0}
    fake_free(monkeypatch, free)
    c = Cache(fast, [bulk], min_free_gb=4)
    a = c.entry("spans", k=1).open(); a.mark_complete()
    assert a.dir.is_relative_to(fast)
    free[fast] = 1.0                                         # SSD nearly full
    b = c.entry("spans", k=2).open()
    assert b.dir.is_relative_to(bulk) and b.rel() == f"spans/{b.key}"
    # every process finds both again, wherever they live
    assert Cache(fast, [bulk], 4).entry("spans", k=1).dir == a.dir
    assert Cache(fast, [bulk], 4).entry("spans", k=2).dir == b.dir
    assert c.where() == {str(fast): 1, str(bulk): 0}


def test_complete_copy_wins_and_partial_without_room_moves_on(tmp_path, monkeypatch):
    fast, bulk = tmp_path / "fast", tmp_path / "bulk"
    free = {fast: 50.0, bulk: 500.0}
    fake_free(monkeypatch, free)
    c = Cache(fast, [bulk], 4)
    e = c.entry("sae_out", k=1).open()
    free[fast] = 0.5                                         # partial entry on a disk that is now full
    e2 = c.entry("sae_out", k=1)
    assert e2.dir.is_relative_to(bulk)                       # restart elsewhere instead of failing on ENOSPC
    e2.open(); e2.mark_complete()
    free[fast] = 50.0
    assert c.entry("sae_out", k=1).dir == e2.dir             # the complete copy is the one used


def test_single_root_is_unchanged(tmp_path):
    c = Cache(tmp_path)
    assert c.entry("spans", k=1).dir.parent.parent == tmp_path


def test_unwritable_tier_is_skipped(tmp_path, monkeypatch):
    fast = tmp_path / "fast"
    blocker = tmp_path / "file"; blocker.write_text("x")
    fake_free(monkeypatch, {fast: 0.1, blocker: 900.0})
    c = Cache(fast, [blocker / "bulk"], 4)                   # mkdir under a regular file fails
    assert c.entry("spans", k=1).dir.is_relative_to(fast)    # falls back to the roomiest usable root


def test_profile_reads_spill_from_local_yaml(tmp_path):
    (tmp_path / "x.yaml").write_text("device: cpu\ncache_root: /a/cache\n")
    (tmp_path / "local.yaml").write_text("cache_spill: [/b/cache]\nhf_home: /b/hf\n")
    p = load_profile("x", env={}, profile_dir=tmp_path)
    assert [str(r) for r in p.spill_roots()] == ["/b/cache"] and p.cache().roots == [Path("/a/cache"), Path("/b/cache")]


def test_disk_info_and_usage(tmp_path):
    (tmp_path / "spans" / "k").mkdir(parents=True)
    (tmp_path / "spans" / "k" / "f").write_bytes(b"x" * 1000)
    d = storage.disk_info(tmp_path)
    assert d["free_gib"] > 0 and d["kind"] in ("SSD", "HDD", "network/unknown")
    assert storage.cache_usage(tmp_path) == {"spans": 1000}
    assert "write_mb_s" in storage.speed(tmp_path, mb=2)
