#!/usr/bin/env bash
# Nsight profiling of all delivery strategies.
#   scripts/profile.sh [N] [K] [RATE_HZ]  (defaults: 100000 1000 20)
# Writes to results/profile/: nsys_<strat>.nsys-rep and .txt (kernel times),
# ncu_<strat>.ncu-rep (kernel metrics), summary.md (tables of both).
# ncu needs admin rights on many machines: sudo env "PATH=$PATH" scripts/profile.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
N=${1:-100000}
K=${2:-1000}
RATE=${3:-20}
OUT=results/profile
mkdir -p "$OUT"
ARGS="--N $N --K $K --fixed-rate $RATE"
command -v nsys >/dev/null || { echo "nsys (Nsight Systems) not found"; exit 1; }
command -v "$NCU" >/dev/null || echo "WARNING: $NCU not found: Nsight Compute part skipped"

for V in naive push pull push-opt; do
    S=${V%-opt} # push-opt -> push
    EXTRA=""
    [ "$V" = push-opt ] && EXTRA="--stateless-rng"

    echo "=== $V: nsys (100 ms simulated)"
    nsys profile --force-overwrite true --trace cuda -o "$OUT/nsys_$V" \
        ./build/snn --strategy $S $EXTRA $ARGS --T 100 > /dev/null
    nsys stats --force-export true --report cuda_gpu_kern_sum --format table \
        "$OUT/nsys_$V.nsys-rep" > "$OUT/nsys_$V.txt" 2>/dev/null

    # skip 400 launches (build + start-up), then profile 3 launches of each kernel
    command -v "$NCU" >/dev/null || continue
    echo "=== $V: ncu (skip warm-up, profile 3 launches of each kernel)"
    rm -f "$OUT/ncu_$V.ncu-rep"
    "$NCU" --set full --import-source yes -f -o "$OUT/ncu_$V" \
        --kernel-name regex:"k_lif_update|k_push|k_pull" --launch-skip 400 --launch-count 6 \
        ./build/snn --strategy $S $EXTRA $ARGS --T 30 > "$OUT/ncu_$V.log" 2>&1 || true
    if [ ! -f "$OUT/ncu_$V.ncu-rep" ]; then
        echo "WARNING: ncu produced no report for $V (see $OUT/ncu_$V.log). If it says"
        echo "ERR_NVGPUCTRPERM, rerun with sudo; if no kernels were profiled, use a newer ncu."
    fi
done
python3 python/profile_summary.py
