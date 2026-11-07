# GPU Spiking Neural Network Simulator: Push vs. Pull Spike Delivery

A CUDA simulator for a Brunel balanced network (80% excitatory, 20% inhibitory LIF neurons,
sparse random connections, synaptic delays). It compares ways of delivering spikes on the
GPU as firing rate, connectivity and network size change.

## Quick start

Needs: Linux, an NVIDIA GPU (~10 GB for the 1M-neuron run), CUDA 12.x, Python 3.9+, git.

```bash
scripts/setup.sh  # venv, packages, GeNN, build, tests (must end with ALL TESTS PASSED)
scripts/run_all.sh  # all experiments, ~15 min plus GeNN compiles; writes results/
scripts/demo.sh  # short live demo
```

- Skip GeNN: `SKIP_GENN=1` before both scripts. Skip profiling: `SKIP_PROFILE=1 scripts/run_all.sh`.
- Pick an idle GPU with `CUDA_VISIBLE_DEVICES=<index>`; other jobs on the GPU slow down timings.
- Check the run: `grep "<<<" results/run_all.log` (every step should show `exit=0`).

Single runs:

```bash
make && make test
./build/snn --strategy push --N 100000 --K 1000 --T 200 --fixed-rate 20
./build/snn --help
```

Troubleshooting:

| Problem | Fix |
|---|---|
| `nvcc` not found, or `cudaGraphInstantiate` compile error | `export CUDA_PATH=/path/to/cuda12` |
| `no kernel image is available` | `make clean && make ARCH=sm_XX` |
| `ERR_NVGPUCTRPERM` in `results/profile/ncu_*.log` | `sudo env "PATH=$PATH" scripts/profile.sh` |
| `FAILED` for N = 1M | Not enough GPU memory; other sizes still run |

## How it works

Each 0.1 ms step runs two kernels: `k_lif_update` (one thread per neuron: update voltage,
collect spikes) and a delivery kernel. A spike at step t with delay d is added to the ring
buffer `I_ring[(t+d) % R]`.

| Strategy | Mapping | Notes |
|---|---|---|
| `naive` | 1 thread per spike | Baseline, uncoalesced |
| `push` | 1 warp per spike | Work ∝ spikes, uses `atomicAdd` |
| `pull` | 1 thread per target neuron | No atomics, but reads all N·K synapses every step |

Optimization flags: `--stateless-rng`, `--block-compact`, `--sort-rows`, `--graph N`,
`--opt-all` (all four), and `--warp-agg` (experimental).

## Results (RTX 6000 Ada)

Time per step (ms):

| | N=12.5k Brunel | N=100k, 20 Hz |
|---|---|---|
| Dense PyTorch | 0.915 | too large |
| GeNN | 0.021 | 0.058 |
| naive | 0.087 | 0.128 |
| push | 0.018 | 0.024 |
| **push + opt** | **0.014** | **0.015** |
| pull | 0.067 | 0.550 |

- Push wins at every rate tested (0.5–200 Hz). Pull is memory-bound (94% of bandwidth).
- `--stateless-rng` is the best optimization (1.46×); `--sort-rows` is slightly slower.
- 1M neurons: 0.114 ms/step (37 billion synaptic events/s).
- Same statistics as Brian2 and GeNN: rate 43.4 Hz (ours) vs 43.1 / 42.9 Hz, CV(ISI) ≈ 0.35.
- Warp-aggregated atomics make push slower (up to 3×): targets almost never repeat within a
  warp (0.01% of atomics saved at N = 100k), and the atomics were not a bottleneck.

Full results: [`results/RUN_RTX_6000_Ada.md`](results/RUN_RTX_6000_Ada.md).

## Files

| Path | Contents |
|---|---|
| `src/snn.cuh` | Start here; includes the other headers |
| `src/common.cuh` | Parameters and options |
| `src/build_network.cuh` | Random network generation |
| `src/neuron_update.cuh` | Neuron kernel |
| `src/delivery.cuh` | Delivery kernels (naive, push, pull) |
| `src/simulator.cuh` | `Simulator` class |
| `src/main.cu` | Command-line program |
| `tests/test_snn.cu` | Correctness tests |
| `python/` | Sweeps, ablation, warp aggregation, Brian2/GeNN/PyTorch baselines, plots |
| `scripts/` | Setup, run all, profiling, demo |

## LLM usage

I used Claude (Anthropic) for debugging, polishing the code and documentation, and help
interpreting the results.
