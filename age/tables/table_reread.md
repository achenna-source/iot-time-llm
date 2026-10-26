Table: Re-read decisions on the main grid
| Model | Letter mass | Re-read | Fresh reading | Acting optimal for any time share | At $r = 0.6$ | Condition holds / fails | $J$ [95% CI] |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 1.00 | 100 | 100 | 100 | 100 | 100 / 100 | +0.000 [+0.000, +0.000] |
| Qwen3-8B | 1.00 | 89 | 74 | 88 | 90 | 87 / 91 | +0.022 [+0.009, +0.038] |
| Llama-3.1-8B | 0.99 | 98 | 97 | 98 | 98 | 97 / 100 | −0.001 [−0.002, +0.000] |
| Granite-3.3-8B | 1.00 | 51 | 31 | 50 | 53 | 41 / 61 | +0.020 [+0.008, +0.032] |
| OLMo-2-7B | 1.00 | 85 | 87 | 84 | 84 | 85 / 85 | +0.009 [+0.000, +0.020] |
| Qwen2.5-14B (6 scenarios) | 1.00 | 96 | n/a | 94 | 94 | 97 / 95 | +0.019 [+0.010, +0.028] |
| Optimal policy | n/a | 47 | 0 | 0 | 0 | 47 / 47 | +0.314 |
Note: 8,640 probes per model; 4,320 for the Qwen2.5-14B scale check. Decisions are the most probable option; rates in percent. 95% intervals for $J$ from a bootstrap over the twelve scenarios. Letter mass: median probability that the first generated token is an option letter. Fresh reading: gate whose reading was taken in the current minute (288 probes). Acting optimal for any time share: the 3,744 probes on which acting on the reading is optimal whatever share of time the device spends in each state (Proposition 4). At $r = 0.6$ a read costs more than the expected loss of acting at any age in the corpus. Condition holds / fails: remembered state in which the instruction asks for the command, or for nothing.
