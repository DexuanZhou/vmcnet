# N2 spectral-cluster envelope screen

Default-disabled, paired 200-step screen from the matched N2 equilibrium
KFAC-pre5000 checkpoint.  Arm A is the existing hard rank-200 WSSR control.
Arm B uses a current top-32 SSI basis plus the previous two top-32 bases as an
enclosing subspace.  It uses no S or gradient EMA.  The complement coefficient
is `alpha=0.2`; the unspecified `gamma` in the proposal is resolved adaptively
as `bar_lambda + lambda`, so the norm constraint cannot silently turn the arm
into an arbitrary fixed-gamma experiment.

The screen advances only if the last-100-step median enclosing-space numerical
rank is at least 48, current-basis novelty is nonzero, all updates are finite,
and both update components remain numerically active. No E5000 is submitted by
this script.
