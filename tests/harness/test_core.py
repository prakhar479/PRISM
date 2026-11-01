"""Profiles, devices, tolerances, check_env (handoff test 17, part)."""
import json
from pathlib import Path

import pytest
import torch

from harness import device as dv
from harness.profile import load_profile


def test_profiles_load_and_env_overrides(tmp_path):
    p = load_profile("dev", env={})
    assert p.name == "dev" and p.device == "auto" and p.root("cache_root").is_absolute()
    q = load_profile("run", env={"HL_CACHE_ROOT": str(tmp_path / "c"), "HL_DEVICE": "cpu"})
    assert q.cache_root == str(tmp_path / "c") and q.device == "cpu" and "env:HL_CACHE_ROOT" in q.sources
    assert q.model_dtype == "bfloat16"
    assert load_profile(env={"HL_PROFILE": "run"}).name == "run"


def test_local_override_and_unknown_keys(tmp_path):
    d = tmp_path / "profiles"
    d.mkdir()
    (d / "dev.yaml").write_text("name: dev\nbatch_size: 4\n")
    (d / "local.yaml").write_text("batch_size: 2\n")
    assert load_profile("dev", env={}, profile_dir=d).batch_size == 2
    (d / "local.yaml").write_text("bogus: 1\n")
    with pytest.raises(ValueError):
        load_profile("dev", env={}, profile_dir=d)


def test_experiment_allowlist():
    p = load_profile("dev", env={})
    p.allowed_experiments = ["E0"]
    p.check_experiment("E0")
    with pytest.raises(PermissionError):
        p.check_experiment("E9")


def test_device_resolution_cpu_and_dtypes():
    assert dv.resolve_device("cpu").type == "cpu"
    assert dv.resolve_device("auto").type in ("cpu", "cuda", "mps")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            dv.resolve_device("cuda")
    assert dv.dtype_of("bf16") is torch.bfloat16 and dv.dtype_name(torch.float32) == "float32"
    assert dv.free_bytes(dv.cpu()) is None and dv.is_oom(dv.FakeOOM("x"))


def test_no_device_literals_in_harness():
    """Only harness/device.py may name concrete accelerator devices."""
    root = Path(__file__).resolve().parents[2] / "harness"
    bad = []
    for f in root.rglob("*.py"):
        if f.name == "device.py":
            continue
        s = f.read_text()
        for pat in ('"cuda"', "'cuda'", ".cuda(", '"mps"', "torch.cuda."):
            if pat in s:
                bad.append(f"{f.relative_to(root)}: {pat}")
    assert not bad, bad


def test_check_env_parses_lock(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_env", Path(__file__).resolve().parents[2] / "scripts" / "check_env.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    lock = tmp_path / "l.txt"
    lock.write_text("torch==0.1.0+cpu\n    # via x\nnumpy==99.0\nfoo-bar==1.0\n")
    pins = m.parse_lock(lock)
    assert pins == {"torch": "0.1.0+cpu", "numpy": "99.0", "foo-bar": "1.0"}
    res = m.check(pins)
    assert any("torch" in e for e in res["errors"]) and any("foo-bar" in w for w in res["warnings"])


def test_cap_memory_passes_a_device_index(monkeypatch):
    """set_per_process_memory_fraction rejects an index-less CUDA device (preflight smoke on RUN, 2026-11-01)."""
    got = {}

    class Props:
        total_memory = 8 * 2 ** 30

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: Props())

    def frac(f, i):
        got["args"] = (f, i)

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", frac)
    for spec, want in (("cuda", 0), ("cuda:1", 1)):
        dv.cap_memory(torch.device(spec), 4.0)
        assert isinstance(got["args"][1], int) and got["args"][1] == want and got["args"][0] == 0.5
