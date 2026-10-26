Table: Arbitration between timestamps and maintenance history
| Model | Letter mass | Accuracy | $A$ [95% CI] | $A$ within $π$ (0.7 / 0.9) | AUC | Position reliance | Timestamp gate | History gate |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 1.00 | 0.664 | −0.008 [−0.044, +0.030] | −0.012 / −0.022 | 0.52 | +0.124 | 0.58 | 0.76 |
| Qwen3-8B | 0.93 | 0.774 | +0.012 [−0.018, +0.042] | +0.023 / −0.040 | 0.53 | −0.002 | 0.57 | 0.88 |
| Llama-3.1-8B | 0.99 | 0.781 | −0.007 [−0.028, +0.014] | −0.020 / −0.008 | 0.53 | −0.013 | 0.60 | 0.95 |
| Granite-3.3-8B | 1.00 | 0.674 | +0.017 [−0.022, +0.058] | +0.002 / +0.018 | 0.51 | +0.104 | 0.58 | 0.69 |
| OLMo-2-7B | 1.00 | 0.748 | +0.031 [+0.002, +0.058] | +0.024 / +0.022 | 0.53 | +0.075 | 0.51 | 0.91 |
| Qwen2.5-14B | 1.00 | 0.777 | +0.013 [−0.026, +0.051] | +0.007 / +0.006 | 0.51 | +0.080 | 0.69 | 0.73 |
| Bayes attributor | n/a | 0.858 | +0.463 | +0.389 / +0.350 | 0.82 | +0.006 | 1.00 | 1.00 |
| Calibrated analyst | n/a | 0.804 | +0.313 | +0.226 / +0.263 | 0.82 | −0.000 | 0.77 | 0.80 |
Note: Main part, 1,728 probes. Decisions are the most probable option, with the assistant turn prefilled with "Answer:". $A$ with a 95% bootstrap interval; within $π$: $A$ computed separately for the history priors 0.7 and 0.9; AUC: how well the probability of following the timestamps separates the items on which they are right from those on which the history is right. Letter mass: median probability on the option letters. Position reliance: probability of naming the event whose line comes first, minus 1/2 (line order is declared uninformative). Timestamp gate: share of decisions that follow the earlier timestamp when the history is uninformative (432 probes). History gate: share that follow the history when the clocks are declared unsynchronised (96 probes). The calibrated analyst answers each event with its posterior probability; its gate columns are expected rates.
