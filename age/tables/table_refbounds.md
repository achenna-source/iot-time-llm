Table: Reference policies on the main grid (8,640 probes). $ρ$: normalised regret at each read price (0 = optimal, 1 = the better fixed rule at that price). At $r = 0.6$ never re-reading is optimal, so $ρ$ is undefined and the excess expected cost over the optimum, in units of the wrong-command cost $L$, is given instead. $J$: staleness sensitivity index.
| Policy | Re-read rate | $J$ | $ρ$, $r = 0.02$ | $ρ$, $r = 0.1$ | $ρ$, $r = 0.3$ | Excess cost, $r = 0.6$ |
|---|---|---|---|---|---|---|
| Always re-read | 1.00 | +0.000 | 1.00 | 1.00 | 1.90 | 0.361 |
| Never re-read (act on the reading) | 0.00 | +0.000 | 69.16 | 5.70 | 1.00 | 0.000 |
| Best single time-to-live (hindsight) | 0.50 | +0.383 | 13.41 | 1.00 | 0.42 | 0.111 |
| Best time-to-live per read price | 0.50 | +0.169 | 1.00 | 0.48 | 0.40 | 0.000 |
| Best rule without the age (volatility and price known) | 0.55 | +0.110 | 1.00 | 0.85 | 0.66 | 0.000 |
| Exit-rate rule of Proposition 4 | 0.57 | +0.428 | 0.00 | 0.00 | 0.00 | 0.035 |
| Optimal policy (Theorem 1) | 0.47 | +0.314 | 0.00 | 0.00 | 0.00 | 0.000 |
| Best prompt-only threshold for J | 0.50 | +0.539 | 4.59 | 0.04 | 0.00 | 0.083 |
| Oracle (sees the realised change) | n/a | +1.000 | n/a | n/a | n/a | n/a |
Note: Fixed rules score $J = 0$ by construction. The time-to-live rules are chosen on the corpus itself, so they are optimistic for that family; the single time-to-live uses one age threshold for every price. The exit-rate rule ignores the time share of the remembered state and re-reads at $r = 0.6$, where re-reading is never optimal.
