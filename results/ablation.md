
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
