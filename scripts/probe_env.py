"""Environment probe: OS, Python, torch, devices, RAM, disk, Hugging Face access. Never downloads model weights.

    python scripts/probe_env.py                       # markdown to stdout
    python scripts/probe_env.py --out docs/dev_env.md # also write it to a file
    python scripts/probe_env.py --json probe.json     # machine-readable copy (for RUN_REQUESTS returns)
    python scripts/probe_env.py --no-network          # skip every Hugging Face call

HF checks: gemma-2-2b config + tokenizer files only (a few MB), and the *metadata* of one 16k Gemma Scope SAE
(file size via the hub API; params.npz is not downloaded). Works without torch / huggingface_hub installed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "google/gemma-2-2b"
SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_PROBE_DIR = "layer_12/width_16k"
PACKAGES = ["torch", "transformers", "transformer_lens", "sae_lens", "huggingface_hub", "safetensors", "numpy",
            "pandas", "pyarrow", "scipy", "sklearn", "yaml", "pytest", "datasets", "matplotlib"]
GiB = 2 ** 30


def _version(mod: str):
    try:
        m = importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001 - a broken install is still worth reporting
        return None if isinstance(e, ImportError) else f"import error: {type(e).__name__}"
    return getattr(m, "__version__", "unknown")


def _ram() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                out[k] = round(int(v.split()[0]) * 1024 / GiB, 1)
    except OSError:
        pass
    return out


def _cmd(args) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def devices() -> dict:
    d = {"cuda_available": False, "mps_available": False}
    try:
        import torch
    except ImportError:
        d["torch"] = None
        return d
    d["torch_cuda_build"] = torch.version.cuda
    d["cuda_available"] = torch.cuda.is_available()
    if d["cuda_available"]:
        d["gpus"] = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            d["gpus"].append({"name": p.name, "total_gb": round(total / GiB, 2), "free_gb": round(free / GiB, 2),
                              "capability": f"{p.major}.{p.minor}"})
        d["bf16_supported"] = torch.cuda.is_bf16_supported()
        d["nvidia_smi"] = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    mps = getattr(torch.backends, "mps", None)
    d["mps_available"] = bool(mps is not None and mps.is_available())
    d["torch_threads"] = torch.get_num_threads()
    return d


def hf_checks(cache_dir=None) -> dict:
    out = {"token_env": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
           "offline_env": os.environ.get("HF_HUB_OFFLINE", "")}
    try:
        import huggingface_hub as hh
    except ImportError:
        out["error"] = "huggingface_hub not installed"
        return out
    out["token_any"] = hh.get_token() is not None          # env var or `hf auth login` token file
    out["hf_home"] = os.environ.get("HF_HOME", "")
    api = hh.HfApi()
    try:
        out["whoami"] = api.whoami().get("name", "?") if out["token_any"] else None
    except Exception as e:  # noqa: BLE001
        out["whoami"] = f"failed: {type(e).__name__}: {str(e)[:160]}"
    # model: config + tokenizer only
    m = {}
    try:
        info = api.model_info(MODEL_REPO)
        m["revision"] = info.sha
        t0 = time.time()
        files = ["config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]
        got = {}
        for f in files:
            try:
                p = hh.hf_hub_download(MODEL_REPO, f, revision=info.sha, cache_dir=cache_dir)
                got[f] = os.path.getsize(p)
            except Exception as e:  # noqa: BLE001
                got[f] = f"failed: {type(e).__name__}"
        m["files"] = got
        m["seconds"] = round(time.time() - t0, 1)
        cfg = json.loads(Path(hh.hf_hub_download(MODEL_REPO, "config.json", revision=info.sha,
                                                 cache_dir=cache_dir)).read_text())
        m["config"] = {k: cfg.get(k) for k in ("num_hidden_layers", "hidden_size", "vocab_size", "final_logit_softcapping",
                                               "attn_logit_softcapping", "torch_dtype", "transformers_version")}
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=info.sha, cache_dir=cache_dir)
            m["tokenizer_ok"] = True
            m["tokenizer_example"] = {"When Mary": tok.encode("When Mary"), "Total = 8": tok.encode("Total = 8")}
        except Exception as e:  # noqa: BLE001
            m["tokenizer_ok"] = f"failed: {type(e).__name__}: {str(e)[:160]}"
    except Exception as e:  # noqa: BLE001
        m["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    out["model"] = m
    # one 16k SAE: list its L0 variants and stat params.npz without downloading it
    s = {}
    try:
        info = api.model_info(SAE_REPO)
        s["revision"] = info.sha
        entries = [e for e in api.list_repo_tree(SAE_REPO, path_in_repo=SAE_PROBE_DIR, revision=info.sha)]
        s["variants"] = sorted(e.path.split("/")[-1] for e in entries)
        if s["variants"]:
            p = f"{SAE_PROBE_DIR}/{s['variants'][0]}/params.npz"
            pi = list(api.get_paths_info(SAE_REPO, [p], revision=info.sha))
            s["probe_file"] = p
            s["probe_file_gb"] = round(pi[0].size / GiB, 3) if pi and getattr(pi[0], "size", None) else None
    except Exception as e:  # noqa: BLE001
        s["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    out["sae"] = s
    return out


def probe(network: bool = True) -> dict:
    disk = {}
    for label, p in (("repo", ROOT), ("home", Path.home()), ("hf_home", Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")))):
        try:
            u = shutil.disk_usage(p if p.exists() else p.parent)
            disk[label] = {"path_kind": label, "free_gb": round(u.free / GiB, 1), "total_gb": round(u.total / GiB, 1)}
        except OSError:
            pass
    rel = platform.release()
    info = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {rel}",
        "wsl": "microsoft" in rel.lower(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "python_executable_in_repo_venv": str(Path(sys.executable)).startswith(str(ROOT.parent)),
        "ram_gb": _ram(),
        "disk": disk,
        "packages": {p: _version(p) for p in PACKAGES},
        "devices": devices(),
        "git": {"head": _cmd(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
                "branch": _cmd(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"]),
                "dirty": bool(_cmd(["git", "-C", str(ROOT), "status", "--porcelain"]))},
    }
    info["hf"] = hf_checks() if network else {"skipped": "--no-network"}
    return info


def to_markdown(d: dict) -> str:
    dev = d["devices"]
    L = [f"# Environment probe ({d['hostname']}, {d['time']})", "",
         "Generated by `scripts/probe_env.py`. Do not edit by hand.", "",
         "| item | value |", "|---|---|",
         f"| OS | {d['os']}{' (WSL2)' if d['wsl'] else ''}, {d['machine']} |",
         f"| CPUs | {d['cpu_count']} |",
         f"| RAM | total {d['ram_gb'].get('MemTotal')} GiB, available {d['ram_gb'].get('MemAvailable')} GiB |",
         f"| Python | {d['python']} |",
         f"| torch | {d['packages'].get('torch')} (CUDA build {dev.get('torch_cuda_build')}) |",
         f"| CUDA available | {dev.get('cuda_available')} |",
         f"| MPS available | {dev.get('mps_available')} |"]
    for g in dev.get("gpus", []):
        L.append(f"| GPU | {g['name']}: {g['total_gb']} GiB total, {g['free_gb']} GiB free, sm {g['capability']} |")
    if dev.get("nvidia_smi"):
        L.append(f"| nvidia-smi | {dev['nvidia_smi']} |")
    for k, v in d["disk"].items():
        L.append(f"| disk ({k}) | {v['free_gb']} GiB free of {v['total_gb']} GiB |")
    L.append(f"| git | {d['git']['branch']} @ {d['git']['head']}{' (dirty)' if d['git']['dirty'] else ''} |")
    L += ["", "## Packages", "", "| package | version |", "|---|---|"]
    L += [f"| {k} | {v if v else 'not installed'} |" for k, v in d["packages"].items()]
    hf = d["hf"]
    L += ["", "## Hugging Face access", ""]
    if "skipped" in hf or "error" in hf:
        L.append(f"- {hf.get('skipped') or hf.get('error')}")
    else:
        L += [f"- token available: {hf['token_any']} (env var: {hf['token_env']}); whoami: {hf.get('whoami')}",
              f"- HF_HUB_OFFLINE: `{hf['offline_env'] or 'unset'}`"]
        m = hf.get("model", {})
        if "error" in m:
            L.append(f"- `{MODEL_REPO}`: **not accessible**: {m['error']}")
        else:
            L += [f"- `{MODEL_REPO}` revision `{m.get('revision')}`: config/tokenizer files {m.get('files')}",
                  f"  - config: {m.get('config')}",
                  f"  - tokenizer loads: {m.get('tokenizer_ok')}; example ids: {m.get('tokenizer_example')}"]
        s = hf.get("sae", {})
        if "error" in s:
            L.append(f"- `{SAE_REPO}`: **not accessible**: {s['error']}")
        else:
            L += [f"- `{SAE_REPO}` revision `{s.get('revision')}`: `{SAE_PROBE_DIR}` variants {s.get('variants')}",
                  f"  - `{s.get('probe_file')}`: {s.get('probe_file_gb')} GiB (metadata only, not downloaded)"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="write the markdown report here")
    ap.add_argument("--json", help="write the raw probe as JSON here")
    ap.add_argument("--no-network", action="store_true")
    a = ap.parse_args()
    d = probe(network=not a.no_network)
    md = to_markdown(d)
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(d, indent=2, default=str))


if __name__ == "__main__":
    main()
