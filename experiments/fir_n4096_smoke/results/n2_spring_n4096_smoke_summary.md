# N2 SPRING n=4096 smoke timing

- status: **FAILED_OTHER**
- steady mean/median/std: 0.472/0.471/0.002 s/epoch
- estimated 25k / 100k: 3.27 / 13.09 h
- GPU maximum / total / margin: 21695 / 81559 / 59864 MiB
- epoch-1 / final energy: -109.5655518 / -92.5574493 Ha
- final variance / acceptance: 610.321960 / 0.52701

| WSSR | time ratio WSSR/SPRING | memory ratio WSSR/SPRING | per-epoch comparison |
|---|---:|---:|---|
| rank400/warm1 | 0.857 | 1.991 | faster than SPRING |
| rank800/warm2 | 1.104 | 1.991 | slower than SPRING |
| rank1600/warm2 | 1.634 | 2.369 | slower than SPRING |
