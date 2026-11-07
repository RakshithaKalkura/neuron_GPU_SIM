#!/usr/bin/env bash
# Nsight profiling of all delivery strategies.
#   scripts/profile.sh [N] [K] [RATE_HZ]       (defaults: 100000 1000 20)
# Outputs in results/profile/:
#   nsys_<strat>.nsys-rep  timeline (open with nsys-ui)
#   nsys_<strat>.txt       per-kernel time summary
#   ncu_<strat>.ncu-rep    full kernel metrics (open with ncu-ui)
#   summary.md             tables of both (python/profile_summary.py)
# Nsight Compute (ncu) reads GPU performance counters, which many machines allow only
# for admins. If it is not permitted, run this script with sudo (keep your PATH:
# `sudo env "PATH=$PATH" scripts/profile.sh`); otherwise the ncu part is skipped with a warning.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
N=${1:-100000}; K=${2:-1000}; RATE=${3:-20}
OUT=results/profile; mkdir -p "$OUT"
ARGS="--N $N --K $K --fixed-rate $RATE"
command -v nsys >/dev/null || { echo "nsys (Nsight Systems) not found"; exit 1; }
command -v "$NCU" >/dev/null || echo "WARNING: $NCU not found: Nsight Compute part skipped"

for V in naive push pull push-opt; do
  S=${V%-opt}; EXTRA=""; [ "$V" = push-opt ] && EXTRA="--stateless-rng"
  echo "=== $V: nsys (100 ms simulated)"
  nsys profile --force-overwrite true --trace cuda -o "$OUT/nsys_$V" \
       ./build/snn --strategy $S $EXTRA $ARGS --T 100 > /dev/null
  nsys stats --force-export true --report cuda_gpu_kern_sum --format table \
       "$OUT/nsys_$V.nsys-rep" > "$OUT/nsys_$V.txt" 2>/dev/null

  command -v "$NCU" >/dev/null || continue
  echo "=== $V: ncu (skip warm-up, profile 3 launches of each kernel)"
  # skip the first 400 launches (network build + transient), then profile 3 of each kernel
  rm -f "$OUT/ncu_$V.ncu-rep"
  "$NCU" --set full --import-source yes -f -o "$OUT/ncu_$V" \
      --kernel-name regex:"k_lif_update|k_push|k_pull" --launch-skip 400 --launch-count 6 \
      ./build/snn --strategy $S $EXTRA $ARGS --T 30 > "$OUT/ncu_$V.log" 2>&1 || true
  if [ ! -f "$OUT/ncu_$V.ncu-rep" ]; then   # no report -> no permission or ncu too old for this GPU
      echo "WARNING: ncu produced no report for $V (see $OUT/ncu_$V.log). If it says"
      echo "         ERR_NVGPUCTRPERM, rerun with sudo; if no kernels were profiled, use a newer ncu."
  fi
done
python3 python/profile_summary.py
