Table: Extraction fidelity and free answers
| Model | Timestamps | Offset SD | Latency | History | $A$ extracted − declared [95% CI] | Flips | Free: one event named | Agreement | $κ$ |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 81 | 100 | 82 | 97 | −0.094 [−0.183, −0.018] | 11 | 96 | 83 | 0.60 |
| Qwen3-8B | 82 | 100 | 100 | 92 | −0.090 [−0.185, −0.010] | 14 | 100 | 88 | 0.76 |
| Llama-3.1-8B | 94 | 100 | 100 | 99 | −0.062 [−0.152, −0.008] | 11 | 100 | 91 | 0.81 |
| Granite-3.3-8B | 78 | 100 | 71 | 100 | −0.068 [−0.147, −0.007] | 8 | 100 | 83 | 0.65 |
| OLMo-2-7B | 29 | 100 | 62 | 71 | −0.463 [−0.645, −0.247] | 34 | 99 | 89 | 0.78 |
| Qwen2.5-14B | 100 | 100 | 99 | 100 | +0.000 [+0.000, +0.000] | 0 | 100 | 89 | 0.78 |
Note: Stratified random subsets. Extraction: share of 120 JSON outputs in which each field is exact; $A$ for the attributor fed with the extracted fields and for the Bayes attributor with the declared fields on the same items, with a paired bootstrap interval for the difference; flips: share of decisions that differ. Free text: 216 incidents answered without options; agreement and Cohen's $κ$ with the forced choice on the same incident. Rates in percent. Timestamps: the difference between the two extracted timestamps is exact to the millisecond.
