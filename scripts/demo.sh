#!/usr/bin/env bash
# Live demo (~2 min of runtime). Run after scripts/setup.sh.
#   scripts/demo.sh  # press Enter before each part
#   NOPAUSE=1 scripts/demo.sh  # no pauses
#   CUDA_VISIBLE_DEVICES=2 scripts/demo.sh  # pick an idle GPU
cd "$(dirname "$0")/.."
source scripts/env.sh
B=$'\e[1m'; R=$'\e[0m'

part() { # heading, then wait for Enter unless NOPAUSE is set
    echo; echo "${B}==== $1${R}"
    [ -z "$NOPAUSE" ] && read -rp "(Enter) " _
}

run() { # run <label> <snn args>: one simulation, one line of output
    local label=$1; shift
    ./build/snn "$@" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
print(f'  {sys.argv[1]:<22s} {d[\"ms_per_step\"]*1e3:8.1f} us/step   rate {d[\"rate_hz\"]:5.1f} Hz   '
      f'{d[\"events_per_s\"]/1e9:6.2f} G syn-events/s   GPU mem {d[\"gpu_mem_used\"]/2**30:4.1f} GiB')" "$label"
}

echo "${B}GPU spiking network simulator: push vs. pull spike delivery${R}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1
make -s

part "1. Correctness tests (exact delays, all strategies agree)"
./build/test_snn | grep -vE "^\[" | tail -n 5

part "2. Main result: same network (N=100k, K=1000, 20 Hz), only the delivery strategy changes"
run "naive (thread/spike)" --strategy naive --N 100000 --K 1000 --T 200 --fixed-rate 20
run "pull (gather)" --strategy pull --N 100000 --K 1000 --T 200 --fixed-rate 20
run "push (warp/spike)" --strategy push --N 100000 --K 1000 --T 200 --fixed-rate 20
run "push + optimizations" --strategy push --N 100000 --K 1000 --T 200 --fixed-rate 20 --stateless-rng --graph 100

part "3. Firing rate sweep: push stays fast, pull costs the same at every rate"
for r in 1 20 200; do
    run "push, $r Hz" --strategy push --N 100000 --K 1000 --T 200 --fixed-rate $r
    run "pull, $r Hz" --strategy pull --N 100000 --K 1000 --T 200 --fixed-rate $r
done

part "4. Scale: 1 million neurons, 1 billion synapses (real Brunel dynamics)"
run "push + opt, N=1M" --strategy push --N 1000000 --K 1000 --T 200 --stateless-rng --graph 100

part "5. Validation vs. Brian2 and GeNN (N=12.5k, 1 s; from scripts/run_all.sh)"
if ls results/gpu_push.bin results/brian2.bin >/dev/null 2>&1; then
    $PY python/plot.py validate | grep -v wrote | sed "s/^/  /"
    echo "  rasters: results/validate.png"
else
    echo "  (no saved spike files: run scripts/run_all.sh first)"
fi

part "6. Why (Nsight Compute, results/profile/summary.md)"
grep -E "^\| (push|pull) \| k_(push|pull)" results/profile/summary.md |
    awk -F'|' '{gsub(/ /, "", $2); gsub(/ /, "", $3); gsub(/ /, "", $5);
                printf "  %-5s %-12s memory bandwidth used: %5s %%\n", $2, $3, $5}'
echo
echo "${B}done.${R} Plots: results/*.png   full results: results/RUN_RTX_6000_Ada.md"
