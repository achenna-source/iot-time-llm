Table: Cost of the models' re-read decisions at each read price. $ρ$: normalised regret (0 = optimal, 1 = the better fixed rule at that price); in brackets, the excess expected cost over the optimum in percent of the wrong-command cost $L$. At $r = 0.6$ never re-reading is optimal and only the excess cost is defined. Best bias: the lowest regret reachable by adding one constant, chosen with hindsight, to the log-probability of the read option.
| Model | $r = 0.02$ | $r = 0.1$ | $r = 0.3$ | $r = 0.6$ (excess) | Best bias, $r = 0.1$ | Best bias, $r = 0.3$ |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 1.00 (0.3) | 1.00 (3.0) | 1.90 (12.9) | 36.1 | 1.00 | 1.90 |
| Qwen3-8B | 15 (4.7) | 2.02 (6.0) | 1.93 (13.1) | 33.6 | 1.00 | 1.90 |
| Llama-3.1-8B | 2.15 (0.7) | 1.09 (3.2) | 1.89 (12.8) | 35.5 | 1.09 | 1.89 |
| Granite-3.3-8B | 52 (16.7) | 4.97 (14.7) | 2.31 (15.6) | 24.8 | 1.02 | 1.88 |
| OLMo-2-7B | 22 (7.0) | 2.80 (8.3) | 2.35 (15.9) | 34.7 | 1.00 | 1.90 |
| Qwen2.5-14B (6 scenarios) | 2.06 (0.7) | 1.02 (3.0) | 1.80 (12.2) | 33.3 | 1.32 | 1.44 |
| Always re-read | 1.00 (0.3) | 1.00 (3.0) | 1.90 (12.9) | 36.1 | n/a | n/a |
| Never re-read | 69 (22.2) | 5.70 (16.8) | 1.00 (6.8) | 0.0 | n/a | n/a |
| Best single time-to-live | 13 (4.3) | 1.00 (2.9) | 0.42 (2.9) | 11.1 | n/a | n/a |
Note: At $r = 0.02$ the optimum and always re-reading differ by only 0.3% of $L$, so small excess costs give large $ρ$; the excess cost is the quantity to compare across prices.
