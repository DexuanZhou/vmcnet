# Pre-submission validation

Focused CPU regression command:

```bash
pytest -q tests/units/updates/test_wssr.py tests/units/utils/test_io.py \
  -k 'augmented_matrix_free_products or explicit_current_block_operator or optimizer_state'
```

Result: `3 passed, 142 deselected in 342.54s`.

The selected tests cover blockwise augmented products with active history,
the explicit-current block operator, and leaf-sharded optimizer-state
checkpoint round trips. Shell syntax checks also pass for all submission
scripts.

The queued 20-step GPU gate additionally writes and reloads a real rank-1600
checkpoint whose largest optimizer leaf exceeds 4 GiB. The 50k jobs depend on
this gate, so a large-checkpoint failure cannot consume a 50k training run.
