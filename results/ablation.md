
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
