# GPU Spiking Neural Network Simulator: Push vs. Pull Spike Delivery

A CUDA simulator for a **Brunel balanced network** (80% excitatory, 20% inhibitory
leaky integrate-and-fire neurons) with sparse random connectivity and synaptic delays.
The project compares ways of **delivering spikes** on the GPU and measures which is
fastest as firing rate, connectivity, and network size change.

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

## Requirements

- NVIDIA GPU + CUDA toolkit (tested: RTX 4090, CUDA 12.6). For other GPUs set `ARCH`
  (A100: `make ARCH=sm_80`).
- Python 3 with `numpy`, `matplotlib`, `torch` (for the baselines and plots).
- Optional: Brian2 and PyGeNN, for the reference comparisons (see below).

## Running the code

```bash
make                 # builds build/snn and build/test_snn
make test            # correctness tests (must print ALL TESTS PASSED)

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

### Experiments

```bash
python3 python/sweep.py rate      && python3 python/plot.py rate   # push vs pull vs rate
python3 python/sweep.py indegree                                    # vary K
python3 python/sweep.py size                                        # vary N
python3 python/sweep.py dense                                       # vs dense PyTorch
python3 python/ablation.py                                          # effect of each optimization
scripts/profile.sh                                                  # Nsight Systems + Compute
```

Results go to `results/` (`*.jsonl`, `*.png`, `*.md`).

### Baselines

```bash
python3 python/dense_torch.py --N 12500 --T 200                     # dense s @ W in PyTorch

python3 -m venv --system-site-packages .venv && .venv/bin/pip install brian2
.venv/bin/python python/brian2_ref.py --T 1000 --out results/brian2  # CPU reference

# GeNN: build from source (github.com/genn-team/genn), then
CUDA_PATH=/usr/local/cuda .venv/bin/python python/genn_ref.py --N 12500 --T 1000 --record 1000 --out results/genn

# compare statistics (rate, CV of inter-spike intervals) and rasters
for s in push pull; do ./build/snn --strategy $s --record 1000 --out results/gpu_$s; done
python3 python/plot.py validate
```

If gcc crashes while Brian2 or GeNN compile, build GeNN with clang
(`CC=clang CXX=clang++ LDSHARED="clang++ -shared" pip install .`).

## Correctness

- **Delay test:** two neurons excite each other with known delays; spike times must be exact
  for every strategy, with and without optimizations.
- **Equivalence test:** with fixed-rate spiking, all strategies must give the same voltages
  (difference ≈ 1e-6, from the order of floating-point additions).
- **Against other simulators** (N=12,500, K=1,000, 1 s):

| | Rate | CV(ISI) |
|---|---|---|
| Ours: push / pull | 43.3 / 43.4 Hz | 0.353 / 0.353 |
| GeNN | 42.9 Hz | 0.352 |
| Brian2 (CPU) | 43.1 Hz | 0.349 |

Exact spike times can't be compared: float atomics change the summation order, and the
network is chaotic, so tiny differences grow.

## Results so far (RTX 4090; GPU shared with other jobs, so preliminary)

| | N=12.5k Brunel network | N=100k, 20 Hz |
|---|---|---|
| Dense PyTorch | 0.85 ms/step | (too large: N² weights) |
| GeNN (default mode) | 0.022–0.032 | 0.070–0.111 |
| Ours, naive | 0.142–0.148 | 0.19 |
| **Ours, push** | **0.026–0.028** | **0.028–0.053** |
| Ours, pull | 0.087–0.093 | 0.71–0.75 |

- **Push wins at every firing rate tested (0.5–200 Hz).** At dt = 0.1 ms, even at 200 Hz only
  2% of neurons spike per step, while pull reads all N·K connections every step.
- **Nsight Compute:** pull runs at 95% of memory bandwidth (it reads 400 MB per step), so it
  is bandwidth-bound. Push uses only about 20%, so it is latency-bound. In the neuron kernel,
  most memory traffic was the per-neuron random state (fixed by `--stateless-rng`).
- **Optimization speedups for push** (median of 5 runs, µs/step; full table in `results/ablation.md`):

| | N=12.5k Brunel network | N=100k, 20 Hz |
|---|---|---|
| baseline | 41.7 | 65.3 |
| `--stateless-rng` | 39.1 (1.07×) | **39.1 (1.67×)** |
| `--block-compact` | 41.9 (1.00×) | 65.9 (0.99×) |
| `--sort-rows` | 42.6 (0.98×) | 73.0 (0.89×, slower) |
| `--graph 100` | 38.8 (1.08×) | 64.0 (1.02×) |
| all | 38.1 (1.10×) | 42.9 (1.52×) |

  The stateless random generator gives the biggest gain, as the profile predicted. Sorting
  rows makes things slightly slower: 32 sorted targets out of 100k are still on different
  cache lines, so locality doesn't improve, and the atomics may crowd onto the same memory
  partitions. CUDA graphs help only at small N, where launch overhead matters.

## Files

| Path | Contents |
|---|---|
| `src/snn.cuh` | All GPU code: network builder, neuron kernel, delivery kernels, `Simulator` class |
| `src/main.cu` | Command-line program |
| `tests/test_snn.cu` | Correctness tests |
| `python/` | Baselines (`dense_torch`, `brian2_ref`, `genn_ref`), sweeps, ablation, plots, profiling summary |
| `scripts/profile.sh` | Nsight profiling of all strategies |

## Possible extensions

- **Batched pull:** no spike arrives sooner than d_min = 15 steps, so pull could gather inputs
  for 15 steps in one pass and read about 15× less memory.
- **Warp-aggregated push:** combine atomics to the same target (needs a different mapping of
  threads to work, since targets in one row rarely repeat).
- 16-bit source IDs; sweeps up to N = 10⁶; STDP (learning synapses).
