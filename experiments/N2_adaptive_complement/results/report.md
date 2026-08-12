# N2 adaptive-complement validation

All methods restart independently from the validated N2 R=2.068 Bohr, n=4096 KFAC-pre5000 checkpoint.

|variant|stable|frozen energy ± total uncertainty|frozen raw variance ± seed SD|IAT|ESS total|acceptance|train s/epoch|runtime ×|peak MiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|A_rank400_hard|True|-109.51153847 ± 0.00082|2.28489 ± 0.113|1.918|6415673|0.4674|0.4869|1.000|43193|
|B_rank400_beta01|True|-109.51224578 ± 0.001|2.28485 ± 0.223|1.928|6384947|0.4675|0.5083|1.044|43193|
|C_rank400_beta02|True|-109.51102040 ± 0.00081|2.11335 ± 0.104|1.889|6511481|0.4674|0.5122|1.052|43193|
|D_rank800_hard|True|-109.51272370 ± 0.00075|1.96374 ± 0.0528|1.875|6554317|0.4674|0.5641|1.159|51385|

## Conclusions

1. Adaptive complement generalizes to N2: yes.
2. beta=0.2 is consistently better than beta=0.1: yes.
3. Cap-active fractions: beta=0.1 99.9%; beta=0.2 99.1%.
4. Frozen raw variance changes, not merely clipped training variance: beta=0.1 0.00%, beta=0.2 7.51% versus rank400 hard.
5. rank400 beta=0.2 does not match rank800 frozen variance; runtime ratio beta0.2/rank800 = 0.908.
6. Complement dominance / constraint saturation: max cross-seed mean ratio 0.200; constraint-active fraction beta0.2 0.0%.
7. Recommended matched E50000 candidate: D_rank800_hard (not submitted).

Energy is used only as an internal relative diagnostic.
