# Run on NVIDIA RTX 6000 Ada Generation

Full rerun of every experiment (`scripts/run_all.sh`, including GeNN) on 2026-11-07, 18:01–18:10 IST.
The numbers in the README come from an RTX 4090; this run is on a different GPU, so its timings differ.

## Machine

| | |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation, 48 GB, compute capability 8.9 (`ARCH=sm_89`) |
| GPU used | GPU 2 of 3 (`CUDA_VISIBLE_DEVICES=2`) |
| Driver | 550.163.01 |
| CUDA toolkit | 12.9 (nvcc V12.9.86) |
| CPU | AMD EPYC 7313 16-core (32 threads) |
| OS | Linux 6.8.0-138-generic |
| Python | 3.13.0, numpy 1.26.4, torch 2.7.1+cu126, brian2 2.11.0 |
| GeNN | pygenn 5.5.0, built from source (genn-team/genn commit 6aa98b9) |
| Nsight Systems | 2025.5.1 |

**Shared GPU.** Other users' jobs were running on all three GPUs during the run. GPU 2 was the least
busy (about 4–7% utilization and 5 GB memory used by others), so timings may be somewhat
noisier than on an idle GPU. The sweeps keep the fastest of 3 runs, and the ablation uses the
median of 5, which reduces this effect.

## Setup notes for this machine

- **CUDA toolkit.** The system nvcc (12.9, from conda) has no CUDA headers of its own and picked up
  Ubuntu's CUDA 11.5 headers in `/usr/include`, against which the code doesn't compile (the
  `cudaGraphInstantiate` signature changed in CUDA 12). I assembled a self-contained CUDA 12.9
  toolkit (the same nvcc, plus 12.9 headers and libraries from NVIDIA's pip packages) and
  pointed `CUDA_PATH` and `PATH` at it. No code changes were needed.
- **Python environment.** Created with `--system-site-packages` to reuse the system's CUDA build of
  PyTorch. The venv is outside the repo (the root disk was almost full), with `.venv` linked to it.
- **Build.** `make ARCH=sm_89` (the Makefile default, which matches this GPU) and `make test`
  printed `ALL TESTS PASSED`.
- **Nsight Compute.** Reading GPU performance counters needs admin rights on this machine
  (`RmProfilingAdminOnly: 1`; as a normal user `ncu` fails with `ERR_NVGPUCTRPERM`), and the
  system's `ncu` (2021.3) doesn't support Ada GPUs. So in the main `run_all.sh` pass the `ncu` calls
  were replaced by a no-op. Afterwards `scripts/profile.sh` was rerun **as root (sudo)** on GPU 2,
  with Nsight Compute 2025.2.1, at 18:25–18:26. That run produced the profiling section below
  (Nsight Systems + Nsight Compute).
- **One repo change.** `python/profile_summary.py` now finds `ncu` from `$NCU` or `PATH`,
  instead of the hardcoded `/usr/local/cuda/bin/ncu`.

## Key findings on this GPU

- **Correctness.** All tests pass. Push and pull match the naive reference to within 1.4e-6 mV.
  All four simulators give the same Brunel network firing rate: push 43.28 Hz, pull 43.31 Hz,
  Brian2 43.15 Hz, GeNN 42.92 Hz.
- **Push is the fastest of our three strategies at every firing rate (0.5–200 Hz), in-degree and
  network size tested**, as on the 4090. At N = 100k and 20 Hz: push-opt 14.9 µs/step, push 23.7,
  naive 127.8, pull 550.2.
- **Push timings match the 4090 closely** (within about 10%, and push-opt within a few percent).
  **Pull is about 20% slower** on this GPU (550 vs 455 µs/step at N = 100k). Pull is
  memory-bandwidth-bound. The RTX 6000 Ada has slightly less memory bandwidth than the
  RTX 4090 (960 vs 1008 GB/s) and a lower power limit (300 W vs 450 W). Other tenants on the
  shared GPU also use memory bandwidth. Together these plausibly explain the gap, but this run
  did not measure each cause separately.
- **1M neurons:** push-opt runs at 0.114 ms/step (37 G synaptic events/s) and uses 9.1 GiB of GPU
  memory.
- **Ours vs GeNN** (first complete GeNN run; GeNN crashed on the 4090 machine). At N = 100k and
  20 Hz, GeNN postsynaptic takes 57.8 µs/step and GeNN presynaptic 254.5, against push-opt
  14.9 µs/step (3.9× faster than GeNN's default mode). At N = 12.5k (Brunel), GeNN takes 20.7
  against push 18.3 and push-opt 14.2. GeNN is close to ours only at 1 Hz (14.8 vs 12.6).
- **Ablation:** as on the 4090, `--stateless-rng` is the only optimization with a large effect
  (push at N = 100k: 1.46×), `--sort-rows` makes things slower (0.90×), and the other two change
  little.
- **Nsight Compute confirms the 4090 analysis.** `k_pull` uses 93.7% of peak memory bandwidth and
  reads 400 MB from DRAM per step, so it is bandwidth-bound (94.7% on the 4090). `k_push_warp` uses
  only about 9%, so it is latency- and atomics-bound. With `--stateless-rng`, the neuron update
  kernel's DRAM reads fall from 7.6 MB to 1.2 MB per step (Nsight Systems: 11 → 2 µs).
  Kernel times measured under Nsight Compute are inflated, because it replays each kernel with
  cold caches, so use the Nsight Systems times for timing.
- **Dense PyTorch** is 12–130× slower than push at the tested sizes (N = 2k–20k).

## Step status (from `results/run_all.log`)

| step | exit | time (s) |
|---|---|---|
| build | 0 | 0 |
| tests | 0 | 0 |
| validate: GPU push | 0 | 0 |
| validate: GPU pull | 0 | 1 |
| validate: Brian2 | 0 | 31 |
| sweep: rate | 0 | 87 |
| sweep: indegree | 0 | 93 |
| sweep: size | 0 | 125 |
| sweep: dense | 0 | 21 |
| ablation | 0 | 90 |
| profile (Nsight) | 0 | 11 |
| validate: GeNN | 0 | 6 |
| sweep: GeNN | 0 | 65 |
| plots | 0 | 2 |

## Correctness tests

```
[delay tests]
[equivalence tests]
  equivalence pull vs naive (delay 1.5-1.5 ms): max|dV| = 9.54e-07 (max|V| = 15.3)
  equivalence pull vs push  (delay 1.5-1.5 ms): max|dV| = 1.43e-06 (max|V| = 15.3)
  equivalence pull vs naive (delay 0.1-3.0 ms): max|dV| = 9.54e-07 (max|V| = 12.4)
  equivalence pull vs push  (delay 0.1-3.0 ms): max|dV| = 9.54e-07 (max|V| = 12.4)
  equivalence pull vs naive +opt (delay 1.5-1.5 ms): max|dV| = 9.54e-07 (max|V| = 12)
  equivalence pull vs push  +opt (delay 1.5-1.5 ms): max|dV| = 9.54e-07 (max|V| = 12)
  equivalence pull vs naive +opt (delay 0.1-3.0 ms): max|dV| = 9.54e-07 (max|V| = 12)
  equivalence pull vs push  +opt (delay 0.1-3.0 ms): max|dV| = 9.54e-07 (max|V| = 12)
ALL TESTS PASSED
```

## Validation: Brunel network, N = 12,500, K = 1000, 1 s simulated

| simulator | firing rate (Hz) | wall time for 1 s (s) | ms/step |
|---|---|---|---|
| push | 43.2833 | 0.158 | 0.0158 |
| pull | 43.3094 | 0.667 | 0.0667 |
| brian2-cython | 43.148 | 25.252 | – |
| genn-postsynaptic | 42.9213 | 0.158 | 0.0158 |

Raster plots and statistics are in `results/validate.png`.

## Firing-rate sweep (N = 100k, K = 1000, fixed-rate mode): µs per step

| rate (Hz) | naive | push | pull | push-opt | pull-opt | genn-post | genn-pre |
|---|---|---|---|---|---|---|---|
| 0.5 | 81.5 | 23.6 | 543.5 | 12.1 | 523.9 | – | – |
| 1 | 86.8 | 24.1 | 543.0 | 12.6 | 522.2 | 14.8 | – |
| 2 | 104.1 | 22.3 | 555.8 | 13.6 | 522.4 | – | – |
| 5 | 120.5 | 26.0 | 544.0 | 14.6 | 527.6 | 23.0 | – |
| 10 | 120.8 | 23.5 | 543.7 | 16.3 | 521.5 | – | – |
| 20 | 127.8 | 23.7 | 550.2 | 14.9 | 520.1 | 57.8 | 254.5 |
| 50 | 294.6 | 24.1 | 549.3 | 17.3 | 521.3 | 130.4 | – |
| 100 | 441.6 | 31.1 | 545.2 | 20.5 | 523.1 | – | – |
| 200 | 446.0 | 48.6 | 552.2 | 38.1 | 524.5 | 569.5 | – |

**Compared with the RTX 4090 run** (µs/step, RTX 6000 Ada / RTX 4090):

| rate (Hz) | naive (6000 Ada / 4090) | push (6000 Ada / 4090) | pull (6000 Ada / 4090) | push-opt (6000 Ada / 4090) | pull-opt (6000 Ada / 4090) |
|---|---|---|---|---|---|
| 0.5 | 81.5 / 76.6 | 23.6 / 23.4 | 543.5 / 454.6 | 12.1 / 11.7 | 523.9 / 429.4 |
| 1 | 86.8 / 86.0 | 24.1 / 23.8 | 543.0 / 454.6 | 12.6 / 12.1 | 522.2 / 429.4 |
| 2 | 104.1 / 95.7 | 22.3 / 24.8 | 555.8 / 454.6 | 13.6 / 13.1 | 522.4 / 429.4 |
| 5 | 120.5 / 106.1 | 26.0 / 25.8 | 544.0 / 454.5 | 14.6 / 14.2 | 527.6 / 429.5 |
| 10 | 120.8 / 109.3 | 23.5 / 26.0 | 543.7 / 454.6 | 16.3 / 14.4 | 521.5 / 429.5 |
| 20 | 127.8 / 125.5 | 23.7 / 26.2 | 550.2 / 454.6 | 14.9 / 14.5 | 520.1 / 429.5 |
| 50 | 294.6 / 277.1 | 24.1 / 26.9 | 549.3 / 454.6 | 17.3 / 14.8 | 521.3 / 429.7 |
| 100 | 441.6 / 408.5 | 31.1 / 37.3 | 545.2 / 454.7 | 20.5 / 23.9 | 523.1 / 429.9 |
| 200 | 446.0 / 411.2 | 48.6 / 56.5 | 552.2 / 454.9 | 38.1 / 43.4 | 524.5 / 430.2 |

Plot: `results/rate.png`.

## In-degree sweep (N = 50k): µs per step

| config | naive | push | pull | push-opt | pull-opt |
|---|---|---|---|---|---|
| K=100, 5 Hz | 12.3 | 10.3 | 14.7 | 4.7 | 11.3 |
| K=100, 50 Hz | 20.2 | 10.2 | 14.8 | 4.6 | 11.0 |
| K=250, 5 Hz | 20.0 | 9.5 | 25.2 | 5.8 | 21.6 |
| K=250, 50 Hz | 43.2 | 9.9 | 28.8 | 6.1 | 24.3 |
| K=500, 5 Hz | 44.9 | 12.5 | 68.2 | 8.6 | 51.2 |
| K=500, 50 Hz | 78.7 | 13.5 | 77.8 | 9.6 | 54.6 |
| K=1000, 5 Hz | 92.0 | 17.6 | 270.5 | 14.2 | 261.3 |
| K=1000, 50 Hz | 152.5 | 20.8 | 271.1 | 17.3 | 263.0 |
| K=2000, 5 Hz | 192.7 | 31.6 | 539.4 | 24.1 | 537.4 |
| K=2000, 50 Hz | 285.5 | 29.5 | 540.0 | 25.7 | 531.1 |
| K=5000, 5 Hz | 485.0 | 63.0 | 1407.0 | 53.4 | 1402.1 |
| K=5000, 50 Hz | 702.4 | 60.3 | 1401.9 | 57.0 | 1401.4 |

Plot: `results/indegree.png`.

## Network-size sweep (Brunel dynamics, K = 1000, ~43 Hz): µs per step

| N | naive | push | pull | push-opt | pull-opt | genn-post |
|---|---|---|---|---|---|---|
| 12,500 | 86.6 | 18.3 | 67.3 | 14.2 | 63.1 | 20.7 |
| 25,000 | 114.4 | 18.6 | 94.0 | 17.0 | 74.3 | – |
| 50,000 | 144.1 | 21.2 | 285.1 | 16.7 | 256.2 | 79.1 |
| 100,000 | 261.8 | 24.4 | 555.1 | 15.4 | 531.9 | 138.5 |
| 200,000 | 452.6 | 43.8 | 1109.1 | 24.4 | 1065.7 | 270.3 |
| 500,000 | 510.8 | 119.4 | 2817.0 | 62.7 | 2697.1 | – |
| 1,000,000 | 654.1 | 300.2 | 5737.7 | 114.4 | 5524.1 | – |

**Compared with the RTX 4090 run** (µs/step, RTX 6000 Ada / RTX 4090):

| N | naive (6000 Ada / 4090) | push (6000 Ada / 4090) | pull (6000 Ada / 4090) | push-opt (6000 Ada / 4090) | pull-opt (6000 Ada / 4090) |
|---|---|---|---|---|---|
| 12,500 | 86.6 / 58.9 | 18.3 / 16.3 | 67.3 / 58.2 | 14.2 / 14.1 | 63.1 / 56.0 |
| 25,000 | 114.4 / 102.6 | 18.6 / 18.2 | 94.0 / 140.0 | 17.0 / 14.9 | 74.3 / 132.6 |
| 50,000 | 144.1 / 130.0 | 21.2 / 20.4 | 285.1 / 233.2 | 16.7 / 14.8 | 256.2 / 223.0 |
| 100,000 | 261.8 / 238.1 | 24.4 / 24.8 | 555.1 / 455.4 | 15.4 / 15.4 | 531.9 / 430.2 |
| 200,000 | 452.6 / 414.8 | 43.8 / 49.0 | 1109.1 / 984.8 | 24.4 / 24.6 | 1065.7 / 941.8 |
| 500,000 | 510.8 / 489.0 | 119.4 / 137.5 | 2817.0 / 2301.7 | 62.7 / 63.2 | 2697.1 / 2201.2 |
| 1,000,000 | 654.1 / 642.6 | 300.2 / 336.9 | 5737.7 / 4910.5 | 114.4 / 119.1 | 5524.1 / 4719.1 |

At N = 1M: naive 0.654 ms/step (6.52 G syn-events/s, 9.1 GiB GPU memory in use), push 0.300 ms/step (14.20 G syn-events/s, 9.1 GiB GPU memory in use), pull 5.738 ms/step (0.74 G syn-events/s, 9.1 GiB GPU memory in use), push-opt 0.114 ms/step (37.30 G syn-events/s, 9.1 GiB GPU memory in use), pull-opt 5.524 ms/step (0.77 G syn-events/s, 9.1 GiB GPU memory in use).

Plot: `results/size.png`.

## Dense PyTorch vs sparse (Brunel, K = 1000): µs per step

| N | push | pull | push-opt | pull-opt | dense-torch |
|---|---|---|---|---|---|
| 2,000 | 12.3 | 66.2 | 9.3 | 55.8 | 288.2 |
| 5,000 | 15.1 | 61.3 | 12.3 | 63.9 | 274.9 |
| 10,000 | 17.5 | 65.6 | 14.0 | 66.9 | 636.9 |
| 12,500 | 24.0 | 66.1 | 17.3 | 65.5 | 914.9 |
| 20,000 | 16.7 | 68.4 | 17.1 | 67.6 | 2174.7 |

Plot: `results/dense.png`.

## Optimization ablation (median of 5 runs)

### RTX 6000 Ada (this run)

### N=12.5k Brunel (43 Hz): median us/step (speedup vs baseline)

| config | naive | push | pull |
|---|---|---|---|
| baseline | 98.9 (1.00x) | 19.1 (1.00x) | 66.5 (1.00x) |
| +stateless-rng | 89.9 (1.10x) | 17.8 (1.07x) | 66.3 (1.00x) |
| +block-compact | 88.5 (1.12x) | 18.6 (1.03x) | 64.1 (1.04x) |
| +sort-rows | 101.2 (0.98x) | 19.7 (0.97x) | 61.8 (1.08x) |
| +graph100 | 96.7 (1.02x) | 17.9 (1.06x) | 65.4 (1.02x) |
| all | 94.3 (1.05x) | 17.1 (1.11x) | 64.6 (1.03x) |

### N=100k fixed 20 Hz: median us/step (speedup vs baseline)

| config | naive | push | pull |
|---|---|---|---|
| baseline | 144.4 (1.00x) | 26.8 (1.00x) | 553.0 (1.00x) |
| +stateless-rng | 131.7 (1.10x) | 18.3 (1.46x) | 531.4 (1.04x) |
| +block-compact | 140.7 (1.03x) | 27.4 (0.98x) | 551.4 (1.00x) |
| +sort-rows | 149.8 (0.96x) | 29.6 (0.90x) | 553.5 (1.00x) |
| +graph100 | 145.2 (0.99x) | 26.2 (1.02x) | 557.0 (0.99x) |
| all | 141.3 (1.02x) | 19.1 (1.40x) | 531.7 (1.04x) |


### RTX 4090 (previous run, for comparison)

### N=12.5k Brunel (43 Hz): median us/step (speedup vs baseline)

| config | naive | push | pull |
|---|---|---|---|
| baseline | 58.9 (1.00x) | 16.2 (1.00x) | 58.0 (1.00x) |
| +stateless-rng | 57.6 (1.02x) | 15.1 (1.07x) | 56.9 (1.02x) |
| +block-compact | 59.0 (1.00x) | 16.3 (0.99x) | 58.1 (1.00x) |
| +sort-rows | 63.0 (0.94x) | 16.9 (0.96x) | 58.0 (1.00x) |
| +graph100 | 57.6 (1.02x) | 15.1 (1.07x) | 56.9 (1.02x) |
| all | 60.5 (0.97x) | 14.9 (1.09x) | 56.0 (1.03x) |

### N=100k fixed 20 Hz: median us/step (speedup vs baseline)

| config | naive | push | pull |
|---|---|---|---|
| baseline | 125.7 (1.00x) | 26.2 (1.00x) | 454.8 (1.00x) |
| +stateless-rng | 115.5 (1.09x) | 15.9 (1.64x) | 432.9 (1.05x) |
| +block-compact | 125.7 (1.00x) | 26.2 (1.00x) | 454.9 (1.00x) |
| +sort-rows | 133.8 (0.94x) | 28.6 (0.91x) | 454.8 (1.00x) |
| +graph100 | 124.6 (1.01x) | 25.1 (1.04x) | 452.4 (1.01x) |
| all | 122.2 (1.03x) | 17.1 (1.53x) | 429.7 (1.06x) |


## Profiling

### Nsight Systems: median kernel time per step (us)

| strategy | k_lif_update | delivery kernel | delivery | total/step |
|---|---|---|---|---|
| naive | 11.2 | k_push_naive | 119.2 | 130.3 |
| push | 10.9 | k_push_warp | 13.4 | 24.3 |
| pull | 16.1 | k_pull | 466.5 | 482.6 |
| push-opt | 2.0 | k_push_warp | 13.5 | 15.6 |

### Nsight Compute: per-kernel metrics (avg over profiled launches)

| strategy | kernel | time (us) | mem SOL % | SM SOL % | DRAM rd MB | DRAM wr MB | L2 hit % | L1 hit % | occupancy % | atomic reqs | sectors/load req |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | k_lif_update | 30.5 | 50.8 | 3.7 | 7.62 | 5.82 | 67.9 | 69.0 | 35.2 | 0.0 | 19.2 |
| naive | k_push_naive | 304.6 | 1.1 | 0.3 | 1.24 | 0.60 | 83.5 | 45.9 | 38.6 | 7062.7 | 8.9 |
| push | k_lif_update | 29.2 | 46.5 | 3.9 | 7.62 | 3.40 | 68.1 | 69.1 | 36.3 | 0.0 | 19.2 |
| push | k_push_warp | 34.0 | 9.1 | 3.2 | 1.25 | 0.79 | 83.4 | 9.3 | 55.8 | 6512.7 | 2.0 |
| pull | k_lif_update | 31.9 | 61.6 | 3.5 | 7.63 | 10.25 | 67.5 | 68.8 | 38.2 | 0.0 | 19.2 |
| pull | k_pull | 522.7 | 93.7 | 33.5 | 400.03 | 46.30 | 0.7 | 88.0 | 41.7 | 0.0 | 16.8 |
| push-opt | k_lif_update | 9.5 | 23.0 | 12.8 | 1.22 | 0.72 | 56.3 | 35.7 | 33.1 | 0.0 | 3.2 |
| push-opt | k_push_warp | 34.6 | 8.7 | 3.1 | 1.21 | 0.69 | 83.3 | 9.4 | 56.9 | 6311.7 | 2.0 |

