"""Language template packs: tokenizer-side invariants (skipped without the cached Gemma tokenizer) and structure."""
import glob
import unicodedata
from pathlib import Path

import pytest

from harness import families as F
from harness import langfamilies as LF

TOK = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/tokenizer.json"))


def test_every_pack_has_all_three_tasks_and_languages():
    for task in LF.FAMILIES:
        assert set(LF.FAMILIES[task]) == {LF.family_id(task, l) for l in LF.LANGUAGES}
    assert set(LF.LANG_OF) == {f for t in LF.FAMILIES.values() for f in t}


def test_fresh_family_selection_for_e3_is_unchanged():
    assert F.families_for("entity_track") == ["ot_placed", "ot_bin", "ot_drawer"]
    assert F.families_for("add_chain") == ["rs_plus", "rs_symbolic", "rs_story"]
    assert F.families_for("people_cities") == []
    assert F.expand("entity_track", ["pilot", "fresh"]) == ["pilot", "ot_placed", "ot_bin", "ot_drawer"]


@pytest.mark.parametrize("task", list(LF.FAMILIES))
@pytest.mark.parametrize("lang", LF.LANGUAGES)
def test_generation_slots_and_pair_semantics(task, lang):
    fid = LF.family_id(task, lang)
    pairs = F.generate_pairs(task, fid, 20, 0, split="t")
    assert len(pairs) == 20 and len({p.clean_prompt for p in pairs}) == 20
    for p in pairs:
        assert p.clean_prompt != p.corrupt_prompt and p.clean_target_str != p.corrupt_target_str
        line = p.clean_prompt.split("\n")[-1]
        slots = {n for n, _, _ in F.slot_spans(task, fid, line)}
        want = {"entity_track": {"obj_1", "box_1", "move_1_obj", "move_1_box", "query_obj"},
                "add_chain": {"start_num", "add_1", "query"}, "people_cities": {"person_1", "city_1", "query_person"}}[task]
        assert want <= slots, (fid, line, slots)
    for s in (v for v in vars(LF.FAMILIES[task][fid]).values() if isinstance(v, str)):
        assert unicodedata.normalize("NFC", s) == s


def test_people_cities_corrupt_differs_in_exactly_one_binding():
    for lang in LF.LANGUAGES:
        f = LF.PC[LF.family_id("people_cities", lang)]
        for p in F.generate_pairs("people_cities", f.id, 30, 1, split="t"):
            a, b = p.clean_prompt.split(". ") if False else (p.clean_prompt, p.corrupt_prompt)
            wa, wb = a.split(), b.split()
            assert len(wa) == len(wb) and sum(x != y for x, y in zip(wa, wb)) == 1


@pytest.mark.skipif(not TOK, reason="Gemma tokenizer not cached")
def test_validator_passes_on_the_real_tokenizer():
    import importlib.util
    spec = importlib.util.spec_from_file_location("vt", Path(__file__).resolve().parents[2] / "scripts" / "validate_templates.py")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.check(m.tokenizer()) == []
