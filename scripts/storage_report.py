#!/usr/bin/env python
"""Where is the space, and where should it go? Read-only by default.

    python scripts/storage_report.py                       # disks, what the harness keeps on each, advice
    python scripts/storage_report.py --speed               # also a 256 MB write/read test per disk (a few seconds)
    python scripts/storage_report.py --watch 60            # repeat every 60 s while a queue runs
    python scripts/storage_report.py --write-profile --fast /ssd_scratch/$USER/prism --bulk /scratch/$USER/prism

Roles: FAST (SSD) holds the venv, the uv cache and the hot activation caches (read again and again by every stage-3
worker). BULK (HDD) holds the Hugging Face download (read once per process at model load, then served from RAM by the
page cache) and takes any activation cache entry that no longer fits on the SSD (Profile.cache_spill). The repo
directory (home) holds only small things: finished runs, queue state, results.
--write-profile writes profiles/local.yaml with that layout; it moves nothing (the report prints the move commands).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from harness import storage
from harness.profile import REPO_ROOT, load_profile

G = 2 ** 30


def human(n):
    return f"{n / G:.1f} GiB" if n >= G / 10 else f"{n / 2**20:.0f} MiB"


def collect(prof, do_speed):
    cache_roots = [prof.root("cache_root"), *prof.spill_roots()]
    disks = {}
    items = [("cache", p) for p in cache_roots] + [("runs", prof.root("runs_root"))]
    if prof.hf_home:
        items.append(("hf", Path(prof.hf_home)))
    work = os.environ.get("PRISM_WORK")
    if work:
        items.append(("venv + uv cache", Path(work)))
    for role, p in items:
        d = storage.disk_info(p)
        disks.setdefault(d["mount"] or d["path"], {**d, "roles": []})["roles"].append(role)
    if do_speed:
        for d in disks.values():
            d["speed"] = storage.speed(Path(d["path"]))
    use = []
    for r in cache_roots:
        for kind, b in storage.cache_usage(r).items():
            use.append((f"cache/{kind}", str(r), b))
    if prof.hf_home and Path(prof.hf_home).exists():
        use.append(("huggingface", str(prof.hf_home), storage.tree_bytes(Path(prof.hf_home))))
    use.append(("runs", str(prof.root("runs_root")), storage.tree_bytes(prof.root("runs_root"))))
    return disks, use


def advice(prof, disks):
    out = []
    ssd = [d for d in disks.values() if d["kind"] == "SSD"]
    hdd = [d for d in disks.values() if d["kind"] == "HDD"]
    hf = Path(prof.hf_home).resolve() if prof.hf_home else None
    if hf and hdd and ssd:
        on = next((d for d in disks.values() if "hf" in d["roles"]), None)
        if on and on["kind"] == "SSD":
            out.append(f"The Hugging Face download is on the SSD ({on['mount']}). It is read once per process, so it does not "
                       f"need the SSD: move it to the HDD to free that space for the caches:\n"
                       f"    rsync -a {hf}/ <bulk>/hf/ && rm -rf {hf}     # then re-run setup (PRISM_BULK=<bulk>) or --write-profile")
    if not prof.cache_spill and hdd:
        out.append("No cache spill tier is set: when the cache disk fills the queue pauses. Add the HDD with --write-profile --bulk <dir>.")
    for d in disks.values():
        if d["free_gib"] < 5:
            out.append(f"{d['mount']} has only {d['free_gib']} GiB free.")
    if not out:
        out.append("Layout looks right.")
    return out


def show(prof, do_speed):
    disks, use = collect(prof, do_speed)
    print(f"profile {prof.name}; cache tiers: {[str(prof.root('cache_root')), *map(str, prof.spill_roots())]}\n")
    print(f"{'mount':<22}{'kind':<17}{'fs':<8}{'total':>9}{'free':>9}  roles" + ("   write/read MB/s" if do_speed else ""))
    for d in disks.values():
        sp = d.get("speed", {})
        tail = f"   {sp.get('write_mb_s', '?')}/{sp.get('read_mb_s', '?')}" if do_speed else ""
        print(f"{d['mount']:<22}{d['kind']:<17}{d['fs']:<8}{d['total_gib']:>8.0f}G{d['free_gib']:>8.0f}G  {', '.join(d['roles'])}{tail}")
    print("\nwhat the harness keeps:")
    for name, where, b in sorted(use, key=lambda x: -x[2]):
        print(f"  {name:<28}{human(b):>10}  {where}")
    try:
        print("\ncache entries per tier:", json.dumps(prof.cache().where()))
    except OSError:
        pass
    print("\nadvice:")
    for a in advice(prof, disks):
        print("  -", a)
    return disks


def write_profile(fast, bulk):
    fast = Path(fast).resolve()
    lines = ["# written by setup_server.sh (scripts/storage_report.py --write-profile): fast disk first, spill to the bulk disk",
             f"cache_root: {fast}/cache"]
    if bulk:
        bulk = Path(bulk).resolve()
        lines += [f"cache_spill: [{bulk}/cache]", f"hf_home: {bulk}/hf"]
    else:
        lines += [f"hf_home: {fast}/hf"]
    for d in (fast / "cache", *( [Path(bulk) / "cache", Path(bulk) / "hf"] if bulk else [fast / "hf"])):
        d.mkdir(parents=True, exist_ok=True)
    dst = REPO_ROOT / "profiles" / "local.yaml"
    dst.write_text("\n".join(lines) + "\n")
    print(f"wrote {dst}:\n" + dst.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=os.environ.get("HL_PROFILE", "server"))
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--watch", type=float, default=0, help="repeat every N seconds")
    ap.add_argument("--write-profile", action="store_true")
    ap.add_argument("--fast", help="SSD work directory (cache + venv)")
    ap.add_argument("--bulk", help="HDD work directory (Hugging Face download + cache spill)")
    a = ap.parse_args()
    if a.write_profile:
        if not a.fast:
            ap.error("--write-profile needs --fast")
        write_profile(a.fast, a.bulk)
        return 0
    prof = load_profile(a.profile)
    while True:
        show(prof, a.speed)
        if not a.watch:
            return 0
        time.sleep(a.watch)
        print("\n" + "=" * 70 + f"\n{time.strftime('%F %T')}")


if __name__ == "__main__":
    sys.exit(main())
