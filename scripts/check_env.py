#!/usr/bin/env python
"""Compare installed packages with requirements/harness.lock. Fails (exit 1) on a major.minor mismatch of torch,
transformers, transformer-lens or numpy; reports other differences as warnings.

    python scripts/check_env.py [--lock requirements/harness.lock] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRICT = ("torch", "transformers", "transformer-lens", "numpy")


def parse_lock(path: Path) -> dict:
    pins = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)", line.strip())
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def base(v: str) -> str:
    return v.split("+")[0]


def majmin(v: str) -> tuple:
    return tuple(base(v).split(".")[:2])


def check(pins: dict) -> dict:
    out = {"errors": [], "warnings": [], "checked": {}}
    for name, want in sorted(pins.items()):
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            (out["errors"] if name in STRICT else out["warnings"]).append(f"{name}: not installed (lock {want})")
            continue
        out["checked"][name] = {"lock": want, "installed": have}
        if base(have) == base(want):
            continue
        msg = f"{name}: installed {have}, lock {want}"
        (out["errors"] if name in STRICT and majmin(have) != majmin(want) else out["warnings"]).append(msg)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default=str(ROOT / "requirements" / "harness.lock"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = check(parse_lock(Path(a.lock)))
    res["python"] = sys.version.split()[0]
    for e in res["errors"]:
        print("ERROR  ", e)
    for w in res["warnings"]:
        print("warning", w)
    print(f"check_env: {len(res['errors'])} errors, {len(res['warnings'])} warnings, {len(res['checked'])} packages checked")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
