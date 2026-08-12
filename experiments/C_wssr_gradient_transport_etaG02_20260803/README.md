# C WSSR exponential-gradient transport test

The first gate is a read-only same-coordinate secant audit at epochs 1000 and
3000 of the existing exponential-S trajectory.  It compares both signs of
`S_current @ delta_theta` and `S_ema @ delta_theta` against the observed change
in the VMC gradient on unchanged electron coordinates.

If that gate passes, the matched C E5000 experiment uses

- `eta_S(t) = 0.95 * (1 - exp(-t / 1000))`
- `eta_g(t) = 0.20 * (1 - exp(-t / 1000))`

and compares gradient transport off versus on with every other setting fixed.
