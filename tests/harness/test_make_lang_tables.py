"""make_e12 / make_e13 on hand-written CSVs: comparisons against English, script partners, gates."""
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def fid(name, layer, rec, lo, hi, fvu=0.1, flo=0.09, fhi=0.11):
    return {"set": name, "lang": "x", "script": "s", "genre": "g", "licence": "l", "layer": layer, "recovered_mean": rec,
            "recovered_mean_lo": lo, "recovered_mean_hi": hi, "fvu": fvu, "fvu_lo": flo, "fvu_hi": fhi}


def test_e12_table_flags_and_script_partners(tmp_path):
    m = load("make_e12")
    write(tmp_path / "E12" / "aaa" / "fidelity.csv", [
        fid("en_pile", 12, 0.98, 0.975, 0.985), fid("hi_deva_dakshina", 12, 0.90, 0.89, 0.91, fvu=0.2, flo=0.19, fhi=0.21),
        fid("hi_latn_dakshina", 12, 0.97, 0.96, 0.98), fid("fr_web", 12, 0.978, 0.97, 0.986)])
    t = {r["set"]: r for r in m.build(m.gather(tmp_path))}
    assert t["hi_deva_dakshina"]["recovered_mean_below_en_ci"] is True and t["hi_deva_dakshina"]["fvu_above_en_ci"] is True
    assert t["fr_web"]["recovered_mean_below_en_ci"] is False
    assert t["en_pile"]["recovered_mean_below_en_ci"] == ""
    assert t["hi_deva_dakshina"]["script_partner"] == "hi_latn_dakshina"
    assert t["hi_deva_dakshina"]["recovered_mean_minus_script_partner"] == -0.07
    assert t["fr_web"]["recovered_mean_minus_en"] == -0.002


def test_e12_later_run_replaces_earlier_for_the_same_set_and_layer(tmp_path):
    m = load("make_e12")
    write(tmp_path / "E12" / "old" / "fidelity.csv", [fid("en_pile", 12, 0.5, 0.4, 0.6)])
    p = tmp_path / "E12" / "new" / "fidelity.csv"
    write(p, [fid("en_pile", 12, 0.98, 0.97, 0.99)])
    import os
    os.utime(tmp_path / "E12" / "old" / "fidelity.csv", (1, 1))
    assert m.gather(tmp_path)[("en_pile", 12)]["recovered_mean"] == "0.98"


def beh(task, family, lang, acc, keep, viable):
    return {"task": task, "family": family, "lang": lang, "script": "s", "validated": "False", "split": "test",
            "length_matching_success": 1.0, "n": 100, "acc_clean": acc, "acc_corrupt": 0.5, "keep_f_A_correct": acc,
            "keep_f_both_correct": 0.4, "keep_pilot_rule": keep, "keep_clean_only": keep, "median_D_clean": 3.0,
            "median_D_corrupt": -1.0, "chance_acc": 0.1, "gate_A_viable": viable}


def test_e13_gates_are_separate_columns(tmp_path):
    m = load("make_e13")
    write(tmp_path / "E13" / "r1" / "behaviour.csv", [
        beh("add_chain", "rs_en", "en", 0.90, 0.6, True), beh("add_chain", "rs_hi", "hi", 0.75, 0.5, True),
        beh("add_chain", "rs_fr", "fr", 0.5, 0.1, False)])
    out = tmp_path / "tables"
    import io, contextlib
    sys.argv = ["make_e13", "--runs", str(tmp_path), "--out", str(out)]
    with contextlib.redirect_stdout(io.StringIO()):
        assert m.main() == 0
    rows = {r["family"]: r for r in csv.DictReader((out / "E13_behaviour_lang.csv").open())}
    assert rows["rs_en"]["gate_B_within_10pts"] == "n/a (English)"
    assert rows["rs_hi"]["gate_A_viable"] == "True" and rows["rs_hi"]["gate_B_within_10pts"] == "False"   # passes A, fails B (gap 15 points)
    assert rows["rs_fr"]["gate_A_viable"] == "False"
