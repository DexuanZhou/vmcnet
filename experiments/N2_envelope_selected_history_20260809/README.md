# N2 fixed-budget current-value historical selection

This is the single follow-up activated by the matched H96/E96 causal test.
The current rank-32 SSI block is protected. Five previous rank-32 blocks form
a 160-column history pool; after removing the current-span component and
orthonormalizing the historical novelty, the 64 directions with the largest
current-batch predicted quadratic decrease are retained. The final
Rayleigh--Ritz capacity is fixed at 96.

No complement, EF, S averaging, or gradient averaging is active. The run uses
the same equilibrium-N2 KFAC-pre5000 matched protocol and stops at E1000 for
the formal frozen evaluation.
