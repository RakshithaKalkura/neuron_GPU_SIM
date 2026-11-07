# Shared environment for the scripts in this folder (sourced, not run).
# Finds CUDA, the Python interpreter and Nsight Compute, so nothing is hard-coded.

# CUDA: $CUDA_PATH if set, else /usr/local/cuda, else the folder that contains nvcc
CUDA_PATH=${CUDA_PATH:-}
if [ -z "$CUDA_PATH" ]; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        CUDA_PATH=/usr/local/cuda
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA_PATH=$(dirname "$(dirname "$(command -v nvcc)")")
    fi
fi
export CUDA_PATH
[ -n "$CUDA_PATH" ] && export PATH="$CUDA_PATH/bin:$PATH"

# Python: the project venv if it exists, else python3
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
export PY

# Nsight Compute binary (override with NCU=/path/to/ncu)
export NCU=${NCU:-ncu}
