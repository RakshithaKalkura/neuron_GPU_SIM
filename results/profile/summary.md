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
