# N2 n=4096 SPRING stability sweep

> This 200-epoch sweep assesses stability and per-epoch cost only; it does not select a winner by energy.

| lr | finite | first nonfinite | E@1/10/50/100/200 | Var@1/10/50/100/200 | acceptance@200 | median s/epoch | max GPU MiB | status |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 0.0005 | 54 | 55 | -109.5656/-111.4794/-70.1400/nan/nan | 2.017/31.886/398.476/nan/nan | nan | 0.477 | 21697 | NONFINITE |
| 0.0002 | 36 | 37 | -109.5656/-110.9378/nan/nan/nan | 2.017/22.437/nan/nan/nan | nan | 0.473 | 21693 | NONFINITE |
| 0.0001 | 36 | 37 | -109.5656/-110.5857/nan/nan/nan | 2.017/17.142/nan/nan/nan | nan | 0.472 | 21697 | NONFINITE |

Timing comparison:

| configuration | sec/epoch | ratio to SPRING lr=0.0005 |
|---|---:|---:|
| WSSR rank400/warm1 | 0.404 | 0.848 |
| WSSR rank800/warm2 | 0.520 | 1.091 |
| WSSR rank1600/warm2 | 0.770 | 1.615 |
