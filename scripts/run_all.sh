#!/usr/bin/env bash
# Reproduce every result in results/ (run scripts/setup.sh first). Use an idle GPU.
#   scripts/run_all.sh  # everything: ~15 min, plus GeNN compiles
#   SKIP_GENN=1 scripts/run_all.sh  # without GeNN
#   SKIP_PROFILE=1 scripts/run_all.sh  # without Nsight profiling
#   CUDA_VISIBLE_DEVICES=1 scripts/run_all.sh  # pick a GPU
# A failed step is logged and the next step still runs.
cd "$(dirname "$0")/.."
source scripts/env.sh
LOG=results/run_all.log; mkdir -p results; : > "$LOG"
START=$(date '+%Y-%m-%d %H:%M:%S')
FAILED=()

health() { # count kernel errors since the script started
    timeout 20 journalctl -k --since "$START" --no-pager 2>/dev/null |
        grep -ciE "bad page|soft lockup|machine check|hardware error|segfault" || true
}

step() { # step <name> <timeout_s> <command...>
    local name=$1 to=$2; shift 2
    echo "[$(date +%T)] >>> $name" | tee -a "$LOG"
    local t0=$SECONDS
    timeout -k 30 "$to" "$@" >> "$LOG" 2>&1
    local rc=$?
    [ $rc -ne 0 ] && FAILED+=("$name")
    echo "[$(date +%T)] <<< $name: exit=$rc, $((SECONDS - t0)) s, kernel errors so far: $(health)" | tee -a "$LOG"
}

nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | tee -a "$LOG"
step "build" 600 make
step "tests" 600 ./build/test_snn
step "validate: GPU push" 300 ./build/snn --strategy push --T 1000 --record 1000 --out results/gpu_push
step "validate: GPU pull" 300 ./build/snn --strategy pull --T 1000 --record 1000 --out results/gpu_pull
step "validate: Brian2" 1800 $PY python/brian2_ref.py --T 1000 --out results/brian2
step "sweep: rate" 3600 $PY python/sweep.py rate
step "sweep: indegree" 3600 $PY python/sweep.py indegree
step "sweep: size" 3600 $PY python/sweep.py size
step "sweep: dense" 1800 $PY python/sweep.py dense
step "ablation" 3600 $PY python/ablation.py
step "warp aggregation" 1800 $PY python/warp_agg.py
[ -z "$SKIP_PROFILE" ] && step "profile (Nsight)" 3600 scripts/profile.sh
if [ -z "$SKIP_GENN" ]; then
    step "validate: GeNN" 1800 $PY python/genn_ref.py --N 12500 --T 1000 --record 1000 --out results/genn
    step "sweep: GeNN" 7200 $PY python/sweep.py genn
fi
step "plots" 600 $PY python/plot.py

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "done: all steps passed; full log in $LOG" | tee -a "$LOG"
else
    echo "done: FAILED steps: ${FAILED[*]}; see $LOG" | tee -a "$LOG"
fi
