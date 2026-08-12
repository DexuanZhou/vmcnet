# N2 fixed-budget historical selection result

Date: 2026-08-09

## Configuration

The current rank-32 SSI block is protected. Five prior rank-32 blocks form a
160-direction pool. Historical components parallel to the current block are
removed, the novelty pool is orthonormalized, and 64 directions are selected
by current-batch predicted quadratic decrease. The final current-batch Ritz
solve has fixed capacity 96. No EF, complement, S averaging, or gradient
averaging is active.

## E1000 frozen comparison

| Method | Frozen energy (Ha) | s.e. (Ha) | Frozen variance |
|---|---:|---:|---:|
| H96: current 96 | -109.480021 | 0.001096 | 4.80512 |
| E96: current 32 + latest history 64 | -109.478416 | 0.001257 | 6.32302 |
| Selected96: current 32 + selected history 64 | -109.480436 | 0.001050 | 4.40989 |
| Existing hard current-only rank-200 | -109.483997 | 0.001275 | 3.11845 |

Selected96 reduces variance by 30.3% relative to E96 and by 8.23% relative
to H96. Its energy is statistically indistinguishable from H96. History is
therefore not intrinsically harmful; unconditional capacity allocation is.
Current-batch value selection recovers a small net benefit, but the candidate
still has 41.4% higher variance than hard rank-200.

The history pool retained numerical rank 160, 64 directions were selected,
and the final envelope retained rank 96. The tail-100 projected regularized
condition number was 3142. The steady training rate was approximately
0.176--0.178 s/step, so this candidate also fails the current Pareto gate
against SPRING (~0.091 s/step).

## Decision

Do not advance Selected96 to E5000 and do not tune its history score. Proceed
to the preregistered current-only d=200/384/512 timing and Ritz-spectrum audit
to determine whether additional current capacity has any viable cost window.

## Jobs and artifacts

- Training: 53864542
- Frozen evaluation: 53865709
- Run root: `/scratch/dexuan1/runs/N2_envelope_selected_history_20260809`
