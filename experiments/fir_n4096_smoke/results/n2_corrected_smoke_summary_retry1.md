# Corrected N2 n=4096 WSSR smoke results

| rank | warm | GPU used/total MiB | margin MiB | steady mean/median/std s | 25k h | 100k h | epoch-1 E | final E | final variance | acceptance | active rank | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 400 | 1 | 43195/81559 | 38364 | 0.404/0.404/0.002 | 2.80 | 11.22 | -109.5591125 | -109.4911346 | 1.02163 | 0.4670 | 2 | SCIENTIFICALLY_VALID_RUNNABLE |
| 800 | 2 | 43195/81559 | 38364 | 0.520/0.520/0.002 | 3.61 | 14.44 | -109.5591125 | -109.5056915 | 0.94772 | 0.4649 | 2 | SCIENTIFICALLY_VALID_RUNNABLE |
| 1600 | 2 | 51387/81559 | 30172 | 0.770/0.770/0.004 | 5.35 | 21.38 | -109.5591125 | -109.5088959 | 0.84287 | 0.4661 | 3 | SCIENTIFICALLY_VALID_RUNNABLE |

## Previous invalid fresh-walker smoke comparison

| rank | warm | old/corrected epoch-1 E | old/corrected acceptance | old/corrected variance | old/corrected steady median s | corrected/old timing |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 1 | -74.8377075/-109.5591125 | 0.2087/0.4676 | 14294.6826/1.9583 | nan/0.404 | nan |
| 800 | 2 | -74.8377075/-109.5591125 | 0.2087/0.4676 | 14294.6826/1.9583 | nan/0.520 | nan |
| 1600 | 2 | -74.8377075/-109.5591125 | 0.2087/0.4676 | 14294.6826/1.9583 | nan/0.770 | nan |
