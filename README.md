# GPU Spiking Neural Network Simulator: Push vs. Pull Spike Delivery

A CUDA simulator for a **Brunel balanced network** (80% excitatory, 20% inhibitory
leaky integrate-and-fire neurons) with sparse random connectivity and synaptic delays.
The project compares ways of **delivering spikes** on the GPU and measures which is
fastest as firing rate, connectivity and network size change.

## Quick start (reproduce all results)

**Needs:** Linux, an NVIDIA GPU (about 10 GB of memory for the largest run), CUDA 12.x
(`nvcc`), Python 3.9+ and git.

```bash
scripts/setup.sh        # venv + packages + GeNN, build, tests (must end with ALL TESTS PASSED)
scripts/run_all.sh      # every experiment: ~15 min plus GeNN compiles
grep "<<<" results/run_all.log   # one line per step; every step should show exit=0
```

- Without GeNN: `SKIP_GENN=1 scripts/setup.sh` and `SKIP_GENN=1 scripts/run_all.sh`.
- Without Nsight profiling: `SKIP_PROFILE=1 scripts/run_all.sh`.
- Multi-GPU or shared machine: choose an idle GPU with `CUDA_VISIBLE_DEVICES=<index>` in front of
  both commands. Other jobs on the same GPU slow down the timings.
- Long runs: `nohup scripts/run_all.sh > run_all.out 2>&1 &`, then `tail -f results/run_all.log`.

`run_all.sh` overwrites `results/`. Its steps:

| Step | Output in `results/` |
|---|---|
| build, correctness tests | `run_all.log` (`ALL TESTS PASSED`) |
| validation: our GPU (push, pull), Brian2, GeNN | `gpu_*.bin`, `brian2.bin`, `genn.bin` → `validate.png` |
| firing-rate sweep (N = 100k) | `rate.jsonl` → `rate.png` |
| in-degree sweep | `indegree.jsonl` → `indegree.png` |
| network-size sweep (up to 1M neurons) | `size.jsonl` → `size.png` |
| dense PyTorch comparison | `dense.jsonl` → `dense.png` |
| optimization ablation | `ablation.md`, `ablation.jsonl` |
| Nsight Systems + Nsight Compute | `profile/summary.md` |

### Troubleshooting

| Problem | Fix |
|---|---|
| `nvcc` not found | `export CUDA_PATH=/path/to/cuda` (the folder that contains `bin/nvcc`) |
| `no kernel image is available` | Wrong GPU architecture: `make clean && make ARCH=sm_XX`. `make` normally detects it. |
| `cudaGraphInstantiate` compile error | nvcc is using headers from an older CUDA (e.g. 11.x in `/usr/include`). Point `CUDA_PATH` to a complete CUDA 12 install. |
| GeNN install fails with a compiler error | `cd third_party/genn && git clean -xfd && cd ../..`, then `CC=clang CXX=clang++ LDSHARED="clang++ -shared" .venv/bin/pip install ./third_party/genn` |
| `ERR_NVGPUCTRPERM` in `results/profile/ncu_*.log` | Reading GPU performance counters needs admin rights: `sudo env "PATH=$PATH" scripts/profile.sh` |
| ncu profiles no kernels | The installed Nsight Compute is older than the GPU; install a newer one, or set `NCU=/path/to/ncu` |
| `FAILED` for N = 1,000,000 in the size sweep | Not enough GPU memory; the other sizes are still measured |

## How it works

Each 0.1 ms timestep runs two kernels:

```
1. k_lif_update   one thread per neuron: read input -> update voltage -> spike? -> reset
                  spiking neurons are collected into a compact list (warp ballot)
2. delivery       send each spike's weight to its targets, d steps in the future
```

**Delays** use a ring buffer `I_ring[R][N]` (R = max delay + 1). A spike sent at step t
with delay d is added to slot `(t+d) % R` and is read by the neuron at step t+d.

**Delivery strategies** (the main comparison):

| Strategy | Mapping | Data layout | Pros / cons |
|---|---|---|---|
| `naive` | 1 thread per spike, walks its row | CSR (by source) | Baseline: uncoalesced loads |
| `push` | 1 warp per spike, lanes split the row | CSR (by source) | Work ∝ #spikes; uses `atomicAdd` |
| `pull` | 1 thread per target neuron, gathers its inputs | CSC (by target) + spike bitmasks | No atomics, deterministic; work ∝ N·K every step |

**Optimizations** (each has a flag so it can be measured alone):

| Flag | What it does |
|---|---|
| `--stateless-rng` | Recreate the random generator from (seed, step, neuron) instead of loading/storing a 64-byte state per neuron |
| `--block-compact` | One `atomicAdd` per block when collecting spikes (shared-memory prefix sum) |
| `--sort-rows` | Sort CSR rows by target so a warp's atomics hit nearby memory |
| `--graph N` | Record N steps as a CUDA graph and replay it (removes launch overhead) |
| `--opt-all` | All of the above |

## Running single experiments

```bash
make                 # builds build/snn and build/test_snn (GPU architecture auto-detected)
make test            # correctness tests

# one simulation: prints a JSON line with time/step, firing rate, events/s, ...
./build/snn --strategy push --N 12500 --K 1000 --T 1000
./build/snn --strategy pull --N 12500 --K 1000 --T 1000 --opt-all

# benchmark mode: every neuron fires at a fixed rate (Hz), so rate is a controlled variable
./build/snn --strategy push --N 100000 --K 1000 --T 200 --fixed-rate 20

./build/snn --help   # all options
```

Main options: `--N` neurons, `--K` inputs per neuron, `--T` simulated ms, `--eta` external
drive (sets the firing rate: 0.9/1.5/2/4 give about 4/27/43/105 Hz), `--delay` or
`--dmin/--dmax` delays in ms, `--record n --out prefix` to save spikes.

Each step of `run_all.sh` can also be run on its own (use `.venv/bin/python`):

```bash
.venv/bin/python python/sweep.py rate        # also: indegree, size, dense, genn
.venv/bin/python python/ablation.py
.venv/bin/python python/brian2_ref.py --T 1000 --out results/brian2         # CPU reference
.venv/bin/python python/genn_ref.py --N 12500 --T 1000 --record 1000 --out results/genn
.venv/bin/python python/dense_torch.py --N 12500 --T 200                    # dense s @ W
scripts/profile.sh
.venv/bin/python python/plot.py              # all plots (or: rate, size, indegree, dense, validate)
```

## Correctness

- **Delay test:** two neurons excite each other with known delays; spike times must be exact
  for every strategy, with and without optimizations.
- **Equivalence test:** with fixed-rate spiking, all strategies must give the same voltages
  (difference ≈ 1e-6, from the order of floating-point additions).
- **Against other simulators** (N = 12,500, K = 1,000, 1 s; RTX 6000 Ada run):

| | Rate | CV(ISI) |
|---|---|---|
| Ours: push / pull | 43.4 / 43.4 Hz | 0.355 / 0.353 |
| GeNN | 42.9 Hz | 0.352 |
| Brian2 (CPU) | 43.1 Hz | 0.349 |

Exact spike times can't be compared: float atomics change the summation order, and the
network is chaotic, so tiny differences grow.

## Results

Full results with every table: [`results/RUN_RTX_6000_Ada.md`](results/RUN_RTX_6000_Ada.md)
(NVIDIA RTX 6000 Ada, CUDA 12.9; the GPU was shared with other jobs). An earlier run on
an RTX 4090 gave very similar push timings; pull was about 20% faster there.

Time per step (ms), RTX 6000 Ada:

| | N=12.5k Brunel network | N=100k, 20 Hz |
|---|---|---|
| Dense PyTorch | 0.915 | (too large: N² weights) |
| GeNN (default, postsynaptic) | 0.021 | 0.058 |
| Ours, naive | 0.087 | 0.128 |
| Ours, push | 0.018 | 0.024 |
| **Ours, push + opt** | **0.014** | **0.015** |
| Ours, pull | 0.067 | 0.550 |

- **Push wins at every firing rate tested (0.5–200 Hz).** At dt = 0.1 ms, even at 200 Hz only
  2% of neurons spike per step, while pull reads all N·K connections every step.
- **Nsight Compute:** pull runs at about 94% of memory bandwidth (it reads 400 MB per step), so it
  is bandwidth-bound. Push uses only about 9%, so it is limited by latency and atomics. In the
  neuron kernel, most memory traffic was the per-neuron random state (fixed by `--stateless-rng`).
- **Optimizations:** `--stateless-rng` gives the biggest gain (push at N = 100k: 1.46×). Sorting
  rows makes things slightly slower (0.90×): 32 sorted targets out of 100k are still on different
  cache lines, so locality doesn't improve. CUDA graphs help only at small N, where launch
  overhead matters. Full table: `results/ablation.md`.
- **Scale:** at N = 1M, push + opt runs at 0.114 ms/step (37 billion synaptic events/s).

## Files

| Path | Contents |
|---|---|
| `src/snn.cuh` | All GPU code: network builder, neuron kernel, delivery kernels, `Simulator` class |
| `src/main.cu` | Command-line program |
| `tests/test_snn.cu` | Correctness tests |
| `python/` | Baselines (`dense_torch`, `brian2_ref`, `genn_ref`), sweeps, ablation, plots, profiling summary |
| `scripts/setup.sh` | One-time setup: venv, packages, GeNN, build, tests |
| `scripts/run_all.sh` | Runs every experiment and writes `results/` |
| `scripts/profile.sh` | Nsight profiling of all strategies |
| `scripts/env.sh` | Finds CUDA, Python and ncu (used by the other scripts) |

## Possible extensions

- **Batched pull:** no spike arrives sooner than d_min = 15 steps, so pull could gather inputs
  for 15 steps in one pass and read about 15× less memory.
- **Warp-aggregated push:** combine atomics to the same target (needs a different mapping of
  threads to work, since targets in one row rarely repeat).
- 16-bit source IDs; STDP (learning synapses).
