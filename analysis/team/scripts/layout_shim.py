"""Maps the analysis scratch layout these archived scripts were written against onto this repository.

The scripts were developed in a scratch workspace whose root (`SP`) held: hw/ (the code checkout), ana_runs/ (merged run
outputs), r5/tables/ (final tables), team/ and deep/ (analysis outputs and scripts). `root()` builds `.layout/` at the repository
root with symbolic links that recreate that layout, so the scripts run unchanged. Run them from the repository root.
deep/hf (two public Gemma Scope 16k decoders) is linked to external/hf; fetch with scripts/fetch_decoders.sh."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LINKS = {"hw": ".", "ana_runs": "results/runs", "r5/tables": "tables/final", "team": "analysis/team"}


def root() -> Path:
    lay = REPO / ".layout"
    for name, target in LINKS.items():
        p = lay / name
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.relpath(REPO / target, p.parent), p)
    deep = lay / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    for entry in (REPO / "analysis" / "deep").iterdir():
        p = deep / entry.name
        if not p.exists():
            os.symlink(os.path.relpath(entry, deep), p)
    (REPO / "external" / "hf").mkdir(parents=True, exist_ok=True)
    if not (deep / "hf").exists():
        os.symlink(os.path.relpath(REPO / "external" / "hf", deep), deep / "hf")
    return lay
