# N2 n=4096 KFAC checkpoint validation

- checkpoint: `/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary/checkpoints/5000.npz`
- stored epoch: 4999 (expected zero-based: 4999)
- geometry: R=2.068000000000 Bohr, positions=[[0.0, 0.0, -1.034], [0.0, 0.0, 1.034]]
- walker position shape: `[4096, 14, 3]`
- chains: 4096
- exact unique walkers: 4096
- exact duplicates: 0
- model leaf shapes match formal n=1000 KFAC: True
- frozen-eval epochs: 20
- energy mean: -109.51130905 Ha
- variance mean: 3.86721936 Ha^2
- acceptance mean: 0.464834
- finite: True
- result: **PASS**
