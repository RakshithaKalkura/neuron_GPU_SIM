### Nsight Systems: median kernel time per step (us)

| strategy | k_lif_update | delivery kernel | delivery | total/step |
|---|---|---|---|---|
| naive | 16.9 | k_push_naive | 113.4 | 130.3 |
| push | 12.5 | k_push_warp | 12.4 | 25.0 |
| pull | 17.6 | k_pull | 433.9 | 451.6 |

### Nsight Compute: per-kernel metrics (avg over profiled launches)

| strategy | kernel | time (us) | mem SOL % | SM SOL % | DRAM rd MB | DRAM wr MB | L2 hit % | L1 hit % | occupancy % | atomic reqs | sectors/load req |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | k_lif_update | 28.0 | 65.7 | 1.7 | 7.62 | 10.44 | 68.5 | 68.1 | 48.5 | 0.0 | 21.5 |
| naive | k_push_naive | 133.2 | 2.5 | 0.2 | 1.25 | 1.57 | 83.4 | 44.6 | 40.8 | 7065.3 | 14.0 |
| push | k_lif_update | 27.1 | 66.3 | 1.8 | 7.62 | 10.02 | 68.0 | 68.1 | 47.1 | 0.0 | 21.5 |
| push | k_push_warp | 16.2 | 20.3 | 2.2 | 1.24 | 1.56 | 83.3 | 5.7 | 58.8 | 6512.7 | 2.6 |
| pull | k_lif_update | 29.4 | 65.7 | 1.5 | 7.62 | 11.35 | 67.2 | 68.4 | 47.8 | 0.0 | 21.5 |
| pull | k_pull | 469.8 | 94.7 | 16.9 | 400.08 | 37.19 | 0.6 | 88.0 | 50.8 | 0.0 | 16.8 |
