"""PREREGISTRATION.md and its addendum were frozen by the humans on 2026-11-16 (DECISIONS.md). Any edit fails this test:
a change needs a new, dated addendum file, never an edit of a frozen one."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = {"PREREGISTRATION.md": "2d336914948c92039aed961c7f591902cc93f570323d81bfe732928d3e3672cf",
          "PREREGISTRATION_ADDENDUM.md": "4ba48a08594af286adc099fbf038e18ec92d512827ea960a8285c095a41f4aaa"}


def test_frozen_documents_are_unchanged():
    for name, digest in FROZEN.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"{name} was edited after the freeze"


def test_decision_constants_match_the_addendum():
    from harness import final as FN
    assert FN.EFFECT_PTS == 0.10 and FN.ALIGNED == 0.5 and FN.DAS_VALID_ACC == 0.5 and FN.M3_MARGIN == 0.10
    assert FN.HAAR_P99 == {1: 0.052, 2: 0.072, 4: 0.097, 8: 0.131}
