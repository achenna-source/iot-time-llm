Table: Comprehension and extraction
| Model | Comprehension | Parsed | Age | Change rate | Read cost | Wrong-command cost | Gate $ρ$ | Flips |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 69% | 100 | 100 | 100 | 100 | 100 | 0.00 | 0 |
| Qwen3-8B | 92% | 100 | 100 | 100 | 100 | 100 | 0.00 | 0 |
| Llama-3.1-8B | 67% | 100 | 100 | 100 | 100 | 100 | 0.00 | 0 |
| Granite-3.3-8B | 88% | 100 | 100 | 100 | 100 | 100 | 0.00 | 0 |
| OLMo-2-7B | 51% | 100 | 28 | 100 | 100 | 100 | 0.59 | 26 |
| Qwen2.5-14B (6 scenarios) | 99% | 100 | 100 | 100 | 100 | 100 | 0.00 | 0 |
Note: Comprehension: accuracy on the gate where re-reading is unavailable (144 probes, chance 50%). Extraction: share of 120 outputs, stratified by price, remembered state and volatility, in which each field is exact. Gate $ρ$: pooled normalised regret of the freshness gate fed with the extracted fields; flips: share of its decisions that differ from the gate fed with the declared fields. Field accuracies in percent.
