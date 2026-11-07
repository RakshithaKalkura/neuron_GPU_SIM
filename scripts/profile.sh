#!/usr/bin/env bash
# Nsight profiling of all delivery strategies.
#   scripts/profile.sh [N] [K] [RATE_HZ]       (defaults: 100000 1000 20)
# Outputs in results/profile/:
#   nsys_<strat>.nsys-rep  timeline (open with nsys-ui)
#   nsys_<strat>.txt       per-kernel time summary
#   ncu_<strat>.ncu-rep    full kernel metrics (open with ncu-ui)
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-100000}; K=${2:-1000}; RATE=${3:-20}
OUT=results/profile; mkdir -p "$OUT"
export PATH=/usr/local/cuda/bin:$PATH
ARGS="--N $N --K $K --fixed-rate $RATE"

for V in naive push pull push-opt; do
  S=${V%-opt}; EXTRA=""; [ "$V" = push-opt ] && EXTRA="--stateless-rng"
  echo "=== $V: nsys (100 ms simulated)"
  nsys profile --force-overwrite true --trace cuda -o "$OUT/nsys_$V" \
       ./build/snn --strategy $S $EXTRA $ARGS --T 100 > /dev/null
  nsys stats --force-export true --report cuda_gpu_kern_sum --format table \
       "$OUT/nsys_$V.nsys-rep" > "$OUT/nsys_$V.txt" 2>/dev/null
  echo "=== $V: ncu (skip warm-up, profile 3 launches of each kernel)"
  # skip the first 400 launches (network build + transient), then profile 3 of each kernel
  ncu --set full --import-source yes -f -o "$OUT/ncu_$V" \
      --kernel-name regex:"k_lif_update|k_push|k_pull" --launch-skip 400 --launch-count 6 \
      ./build/snn --strategy $S $EXTRA $ARGS --T 30 > /dev/null
done
python3 python/profile_summary.py
