# N2 rank-200 complement-recovery screen

This experiment starts every arm from the same N2 (R=2.016 Bohr)
KFAC-pre5000 checkpoint and tests whether useful information omitted by a
rank-200 WSSR subspace can be recovered.  The common matched protocol is 1000
walkers, 10 MCMC steps/update, mean-centred clipping at 5, no reburn,
Tikhonov lambda 1e-3, learning rate 0.002 with inverse-time decay 1e-4, norm
constraint 1e-3, and SSI 40/2.

The four paired arms are:

1. `A_hard_rank200`: rank-200 WSSR with a zero complement (control).
2. `B_isotropic_floor`: the same retained-space solve plus
   `0.0001/lambda * (I-UU^T)g`.  This is the proposed isotropic Tikhonov
   complement floor; the coefficient before the global learning-rate/norm
   controls is 0.1.
3. `C_rotating_EF`: rank-200 full-current Galerkin residual recurrence with
   decayed error feedback (`mu=decay=0.95`).  This is the distinct, implementable
   randomized-sketch compensation arm: fresh batches/SSI rotate the rank-200
   subspace and the omitted parameter-space conflict is carried until a future
   subspace captures it.  The formula in the original proposal that writes
   `eta_probe (I-UU^T)g` without an actual probe is algebraically the same as
   arm B, so it is not duplicated as a falsely independent treatment.
4. `D_momentum_floor`: arm B with gradient EMA
   `m_t=0.9 m_(t-1)+0.1 g_t`; both the retained solve and the small isotropic
   complement use `m_t`.

The primary endpoint is light frozen variance at epochs 1000 and 5000.  Every
arm also reports frozen energy and measured training seconds/step.  This is a
one-seed screen; no 100k job is launched automatically.

Source checkpoint:

`/scratch/dexuan1/runs/old/N2_R2016/e100000_followup/N2eq/N2eq_kfac_pre5000/checkpoints/5000.npz`
