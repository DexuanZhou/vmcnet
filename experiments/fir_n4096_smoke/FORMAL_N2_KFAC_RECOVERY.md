# Formal N2 R=2.068 KFAC preliminary configuration

- Config: `/scratch/dexuan1/runs/kfac_pre/N2eq_R2068_kfac_pre5000/config.json`
- Checkpoint: `/scratch/dexuan1/runs/kfac_pre/N2eq_R2068_kfac_pre5000/checkpoints/5000.npz`
- Geometry: `((0,0,-1.034),(0,0,1.034))` Bohr; charges `(7,7)`; electrons `(7,7)`
- Model: FermiNet; four backflow layers `[(256,16),(256,16),(256,16),(256)]`; 16 determinants; full determinant; isotropic decay; tanh; no cusp Jastrow
- KFAC: learning rate `0.05`; damping `0.001`; norm constraint `0.001`; inverse-time schedule; decay `1e-4`; Fisher-exact; curvature EMA `0.95`; inverse update every step; min damping `1e-4`; no L2
- Training: seed `0`; float32; non-distributed; 1000 chains; burn-in 5000; 5000 epochs; 10 MCMC steps/update
- Sampler: dynamically adjusted Gaussian proposal; initial `std_move=0.25`; width update every 100 moves
- Clipping: center `mean`, threshold `5.0`; `nan_safe=True`
- Checkpoints: every/final 5000; full parameters, optimizer, walker/sampler state, and RNG key

The corrected preliminary run loads this config without loading its checkpoint and changes only the output location and `nchains=4096` (plus Slurm/resources and disabled W&B logging).
