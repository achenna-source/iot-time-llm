Table: Counterfactual timestamp-only error on measured event delays
| Delays | Oscillator | 17 min | 1 h | 1 day | 1 week | 30 days | Uncertifiable, 1 day |
|---|---|---|---|---|---|---|---|
| homes (CASAS) (3370 pairs) | XO ±20 ppm | 0.000 | 0.001 | 0.149 | 0.384 | 0.463 | 0.87 |
|  | TCXO ±2 ppm | 0.000 | 0.000 | 0.006 | 0.102 | 0.297 | 0.19 |
|  | disciplined, ±0.5 ppm | 0.000 | 0.000 | 0.000 | 0.012 | 0.113 | 0.04 |
| industrial testbed (HAI) (37647 pairs) | XO ±20 ppm | 0.000 | 0.000 | 0.011 | 0.137 | 0.269 | 0.28 |
|  | TCXO ±2 ppm | 0.000 | 0.000 | 0.000 | 0.006 | 0.059 | 0.00 |
|  | disciplined, ±0.5 ppm | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.00 |
Note: By oscillator and synchronisation interval; NTP residual 10 ms, uniform prior. The delays are measured; the clock offsets are simulated under Definition 1 with drift uniform over the tolerance (200,000 draws per cell). Uncertifiable: share of measured delays below $2δ$ for the hard bound $δ = 3σ_{s} + ρ_{max}T$. 17 min is the default maximum NTP polling interval ($2^{10}$ s); a disciplined clock keeps a residual frequency error after NTP has steered it. Largest Monte Carlo standard error in the table: 0.0011.
