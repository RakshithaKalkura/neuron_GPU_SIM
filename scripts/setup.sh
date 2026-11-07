#!/usr/bin/env bash
# One-time setup: Python venv, Python packages, GeNN (optional), build, tests.
#   scripts/setup.sh  # everything (GeNN takes a few minutes)
#   SKIP_GENN=1 scripts/setup.sh  # without GeNN
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

echo "== checks"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: no NVIDIA driver?"; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc not found: set CUDA_PATH or add CUDA's bin/ to PATH"; exit 1; }
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
nvcc --version | tail -n1
echo "CUDA_PATH=$CUDA_PATH"

echo "== Python venv (.venv)"
if [ ! -x .venv/bin/python ]; then
    # reuse the system's CUDA PyTorch if there is one (saves a ~2 GB download)
    if python3 -c "import torch, sys; sys.exit(not torch.cuda.is_available())" 2>/dev/null; then
        python3 -m venv --system-site-packages .venv
    else
        python3 -m venv .venv
    fi
fi
PY=.venv/bin/python
$PY -m pip install -q -U pip
$PY -m pip install -q -r requirements.txt

if [ -z "${SKIP_GENN:-}" ]; then
    echo "== GeNN (built from source into the venv)"
    [ -d third_party/genn ] || git clone --depth 1 https://github.com/genn-team/genn.git third_party/genn
    # If this fails with a compiler error, retry with clang:
    #   CC=clang CXX=clang++ LDSHARED="clang++ -shared" .venv/bin/pip install ./third_party/genn
    $PY -m pip install -q ./third_party/genn
    $PY -c "import pygenn; print('pygenn', pygenn.__version__, 'ok')"
fi

echo "== build + tests"
make
make test
echo "setup done. Next: scripts/run_all.sh"
