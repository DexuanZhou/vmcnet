# C stage-wise SPRING -> WSSR result

The variance ratio is paired at the same total optimization epoch and frozen-evaluation seed. A value below one favors the WSSR switch.

| switch epoch | WSSR frozen E | SPRING frozen E | WSSR frozen var | SPRING frozen var | var ratio | conservative WSSR s/step |
|---:|---:|---:|---:|---:|---:|---:|
| 5000 | -37.844869063 | -37.844505119 | 0.02339315 | 0.01500384 | 1.5591 | 0.473600 |
| 20000 | -37.844791964 | -37.844939573 | 0.00749927 | 0.00703069 | 1.0666 | 0.418800 |
| 50000 | -37.844915487 | -37.844882333 | 0.00379218 | 0.00538494 | 0.7042 | existing run |
