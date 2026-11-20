#!/usr/bin/env bash
# Two public Gemma Scope 16k residual SAEs used by the exploratory geometry scripts (decoder atoms at L12 and L18), into
# external/hf in Hugging Face cache layout (git-ignored; ~580 MB). The model itself is not needed for this.
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["layer_12/width_16k/average_l0_82/params.npz", "layer_18/width_16k/average_l0_74/params.npz"]:
    print(hf_hub_download("google/gemma-scope-2b-pt-res", f, revision="fd571b47c1c64851e9b1989792367b9babb4af63",
                          cache_dir="external/hf"))
PY
