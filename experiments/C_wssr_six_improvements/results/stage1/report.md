# Stage 1 — fixed-snapshot numerical screening

All four C `(4,2)`, `nchains=1000` snapshots were replayed without moving walkers or applying an update. The primary reference is the full Tikhonov-regularized inverse on the identical frozen augmented operator. Oversampled late-state methods retain the saved rank-400 right-warm information and add a reproducible checkpoint-key orthogonal completion before normal q=2 SSI.

Jobs: authoritative array `49918208` (all four tasks completed); invariance tests `49922369` (15 passed). Earlier jobs `49913264_0`, `49915718_0`, and `49913264_1` were failed/cancelled infrastructure or pre-correction attempts and are excluded.

## Selected internal settings

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

Projected iterative complement has no training-eligible setting: its maximum complement/resolved ratios are 8.17, 10.78 and 14.26 for 3, 5 and 10 iterations, all above the predeclared 0.5 gate.

## Aggregate over four snapshots

|method|parameter|eligible|mean full error|max full error|mean cosine|max comp/resolved|mean effective rank|
|---|---:|:---:|---:|---:|---:|---:|---:|
|baseline|warm2|True|0.991166|0.997873|0.11008|0|400.0|
|cluster|0.001|True|0.991094|0.997871|0.11039|0|400.2|
|cluster|0.002|True|0.991094|0.997871|0.11039|0|400.2|
|cluster|0.005|True|0.991088|0.997871|0.11042|0|400.5|
|near_tail|32|True|0.988295|0.995854|0.13566|0|432.0|
|near_tail|64|True|0.983867|0.992329|0.16567|0|464.0|
|near_tail|128|True|0.976136|0.987239|0.20615|0|528.0|
|fixed_complement_reference|1e-4|True|0.984418|0.994649|0.16491|0.1085|400.0|
|adaptive_complement|0.05|True|0.987854|0.996134|0.13771|0.05|400.0|
|adaptive_complement|0.1|True|0.984699|0.994649|0.16300|0.1|400.0|
|adaptive_complement|0.2|True|0.984418|0.994649|0.16491|0.1085|400.0|
|smooth|385-432|True|0.989885|0.996966|0.12248|0|432.0|
|smooth|385-464|True|0.987942|0.995249|0.14017|0|464.0|
|smooth|369-464|True|0.988519|0.995746|0.13605|0|464.0|
|force_aware|force|True|0.991206|0.997874|0.10991|0|400.0|
|force_aware|force+krylov|True|0.991202|0.997875|0.10992|0|400.0|
|iterative_complement|3|False|0.631098|0.700082|0.79956|8.165|400.0|
|iterative_complement|5|False|0.452416|0.521114|0.89867|10.78|400.0|
|iterative_complement|10|False|0.213653|0.257154|0.97669|14.26|400.0|

Full per-snapshot results, constrained errors, residuals, principal angles, spectra, runtime and memory are in `all_results.csv`; the exact spectra are in each snapshot `metadata.json`.
