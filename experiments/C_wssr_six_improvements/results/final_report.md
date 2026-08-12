# C WSSR six-improvement independent ablation

All training uses all-electron C `(4,2)`, `nchains=1000`, and the common KFAC-pre1000 checkpoint. No N2 job was run. Absolute C energies are not interpreted physically because the existing reference inconsistency remains unresolved.

## Direct answers

1. Largest full-reference error reduction: **projected iterative complement, 10 iterations**, mean error 0.213653; it is **training-ineligible** because complement/resolved reached 14.3 > 0.5. Best eligible method is **near_tail (128)**, error 0.976136.
2. Lowest frozen raw variance: **adaptive_complement**, 1.10051 ± 0.0782 across seeds.
3. Best measured speed–variance tradeoff: **baseline**.
4. Cluster-aware versus fixed near-tail: mean full errors 0.991088 versus 0.976136; fixed near-tail wins numerically.
5. Adaptive complement versus fixed `1e-4`: 0.984418 versus 0.984418. The selected adaptive beta reaches the nominal cap and therefore does not improve on the fixed case in Stage 1.
6. Smooth-cutoff E200 is STABLE; final-50 raw-variance SD=0.578554 and first-difference SD=0.805576, compared with baseline 1.35874/2.06351.
7. Force-aware is STABLE; its four-snapshot mean full error is 0.991202, versus baseline 0.991166. Injected-vector norms and overlaps are in Stage-1 raw data.
8. Projected iterative complement greatly improves numerical fidelity (0.213653 at 10 iterations) but all variants violate the 0.5 contribution gate (8.17, 10.8, 14.3); it does not justify training in its present isolated form.
9. Recommended next N2/4096 methods: **adaptive_complement (0.2)**, **smooth (385-464)**. This is a recommendation only; no N2 job was submitted.

## Stage 1 selections

```json
{
  "adaptive_complement": {
    "method": "adaptive_complement",
    "parameter": "0.2",
    "eligible": true,
    "ineligible_reason": "",
    "mean_full_error": 0.9844181835651398,
    "max_full_error": 0.9946489930152893,
    "mean_cosine": 0.16491103917360306,
    "mean_runtime": 0.1674648220068775,
    "max_complement_ratio": 0.10845993888601389,
    "mean_effective_rank": 400.0
  },
  "cluster": {
    "method": "cluster",
    "parameter": "0.005",
    "eligible": true,
    "ineligible_reason": "",
    "mean_full_error": 0.9910878837108612,
    "max_full_error": 0.9978706240653992,
    "mean_cosine": 0.11041775345802307,
    "mean_runtime": 0.2909363054932328,
    "max_complement_ratio": 0.0,
    "mean_effective_rank": 400.5
  },
  "force_aware": {
    "method": "force_aware",
    "parameter": "force+krylov",
    "eligible": true,
    "ineligible_reason": "",
    "mean_full_error": 0.99120232462883,
    "max_full_error": 0.9978746771812439,
    "mean_cosine": 0.10992217436432838,
    "mean_runtime": 0.6875490874808747,
    "max_complement_ratio": 0.0,
    "mean_effective_rank": 400.0
  },
  "near_tail": {
    "method": "near_tail",
    "parameter": "128",
    "eligible": true,
    "ineligible_reason": "",
    "mean_full_error": 0.9761357456445694,
    "max_full_error": 0.9872391819953918,
    "mean_cosine": 0.20614667236804962,
    "mean_runtime": 1.594186977497884,
    "max_complement_ratio": 0.0,
    "mean_effective_rank": 528.0
  },
  "smooth": {
    "method": "smooth",
    "parameter": "385-464",
    "eligible": true,
    "ineligible_reason": "",
    "mean_full_error": 0.9879424422979355,
    "max_full_error": 0.9952486753463745,
    "mean_cosine": 0.14017490297555923,
    "mean_runtime": 0.6475249457434984,
    "max_complement_ratio": 0.0,
    "mean_effective_rank": 464.0
  }
}
```

The planned seventh E200 method, projected iterative complement, was excluded by the user-specified eligibility gate. E200 therefore contains baseline plus five eligible independent improvements.

## Stage 2

|method|parameter|status|final-50 raw variance|mean full error|s/epoch|peak MiB|
|---|---|---|---:|---:|---:|---:|
|baseline|warm2|STABLE|1.18246|0.991102|0.1074|17721|
|cluster|0.005|STABLE|1.15407|0.99098|0.1205|34105|
|near_tail|128|STABLE|1.42339|0.983228|0.09764|34105|
|adaptive_complement|0.2|STABLE|1.17391|0.983919|0.08336|17721|
|smooth|385-464|STABLE|1.01316|0.988548|0.08296|34105|
|force_aware|force+krylov|STABLE|1.53245|0.990994|0.1501|17721|

## Stage 3 cross-seed frozen evaluation

|method|parameter|frozen energy mean±seed SD|raw variance mean±seed SD|s/epoch mean±seed SD|
|---|---|---:|---:|---:|
|baseline|warm2|-37.86756775 ± 0.000302|1.47747 ± 0.227|0.08713 ± 0.02|
|smooth|385-464|-37.86779419 ± 0.0016|1.4991 ± 0.325|0.1151 ± 0.0067|
|cluster|0.005|-37.86648519 ± 0.00041|1.76891 ± 0.199|0.1477 ± 0.033|
|adaptive_complement|0.2|-37.86655195 ± 0.00305|1.10051 ± 0.0782|0.1531 ± 0.026|

## Jobs and commands

- Complete CPU regression: `50150519` on `fc30561`, 123 passed, 0 failed,
  0 skipped in 487.81 s of pytest time (Slurm elapsed 00:10:05). This covers
  default-disabled invariance, all experimental modes, exact-first,
  residual-optimal complement, beta decay, near-tail, parser compatibility,
  and the existing WSSR suite.
- `stage2_array=49925325`
- `stage2_accuracy=49926055`
- `stage2_collector=49926056`
- `stage3_release=49926057`
- `stage3_array=49972432`
- `frozen_eval=49972433`
- `collector=49972434`

Exact commands are the versioned local scripts `stage1_array.sh`, `stage2_array.sh`, `stage2_accuracy_array.sh`, `stage3_array.sh`, and `stage3_frozen_eval.sh`. Stage-1 authoritative jobs were `49918208_[0-3]`; invariance tests were `49922369` (15 passed). Pre-correction jobs are preserved but excluded.

## Completion audit

- Default behavior: all experimental modes are disabled in the default
  configuration. The direct invariance test compares the baseline update and
  optimizer state with all experimental controls populated but
  `experimental_mode="none"` and requires exact array equality.
- Stage 1: all four requested snapshots (`initial`, `A500`, `B500`, `C500`)
  are present. Each has 19 frozen-operator rows, for 76 rows total, covering
  baseline, six independent improvements, all requested internal settings,
  and the fixed-complement reference.
- Stage 2: baseline plus five eligible improvements completed 200/200 finite
  epochs. The planned seventh trajectory, projected iterative complement, was
  not run because all three Stage-1 settings violated the predeclared
  `complement/resolved <= 0.5` training-eligibility rule by a large margin
  (8.17–14.26). This is a protocol-mandated exclusion rather than missing
  data.
- Stage 2 accuracy: exact-full comparisons exist at epochs 1, 50, 100, and
  200 for every trained method.
- Stage 3: 12/12 trajectories (baseline plus three selected improvements,
  three seeds each) completed 500/500 epochs with finite epoch-500
  checkpoints.
- Frozen evaluation: 12/12 checkpoints completed matched evaluation with
  4096 walkers, burn-in 10000, 5000 evaluation epochs, and saved raw local
  energies. Every statistics file is finite.
- Scope: all training systems are all-electron C `(4,2)` with `nchains=1000`
  and the same KFAC-pre1000 source checkpoint. No N2 job belongs to this
  experiment. No code was committed or pushed.

The complete regression log is
`/scratch/dexuan1/runs/logs/C-wssr6-tests-50150519.out`.

## Relevant local changes

- ` M tests/units/updates/test_wssr.py`
- ` M vmcnet/updates/wssr.py`
- `?? run_lih_wssr_R40_nou_smoke.sh`
- `?? run_wssr_tests.sh`
- `?? tests/units/updates/test_wssr_experimental.py`
- `?? vmcnet/updates/wssr_experimental.py`

WSSR update semantics were preserved when all new options are disabled; every experimental option is default-disabled. Nothing was committed or pushed.

## Artifacts

- Stage 1: `results/stage1/all_results.csv`, `aggregate.csv`, `report.md`
- Stage 2: `results/stage2/summary.csv`
- Stage 3: `results/stage3/per_seed.csv`, `cross_seed.csv`, `conclusions.json`
- Plots: `results/plots/*.svg`
