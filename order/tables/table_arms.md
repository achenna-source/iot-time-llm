Table: Arbitration when the model is helped
| Model | Arm | Items | Answered within budget | $A$ within $π$, arm | $A$ within $π$, corpus format | Accuracy, arm | Certification, arm |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | D stated | 864 | 100 | +0.019 | −0.016 | 0.793 | n/a |
|  | Reasoning | 192 | 43 | +0.049 | −0.075 | 0.781 | 0.56 |
| Qwen3-8B | D stated | 864 | 100 | −0.010 | −0.004 | 0.771 | n/a |
|  | Reasoning | 192 | 0 | −0.024 | −0.027 | 0.703 | 0.35 |
| Llama-3.1-8B | D stated | 864 | 100 | +0.051 | −0.017 | 0.794 | n/a |
|  | Reasoning | 192 | 81 | +0.026 | −0.014 | 0.750 | 0.48 |
| Granite-3.3-8B | D stated | 864 | 100 | +0.035 | +0.010 | 0.780 | n/a |
|  | Reasoning | 192 | 99 | −0.039 | +0.018 | 0.740 | 0.56 |
| OLMo-2-7B | D stated | 864 | 100 | +0.046 | +0.046 | 0.743 | n/a |
|  | Reasoning | 192 | 69 | −0.028 | −0.007 | 0.714 | 0.48 |
| Qwen2.5-14B | D stated | 864 | 100 | −0.018 | −0.003 | 0.767 | n/a |
|  | Reasoning | 192 | 71 | +0.023 | −0.029 | 0.786 | 0.50 |
| Bayes attributor | | | | +0.369 | | 0.858 | 1.00 |
Note: D stated: every second main item with the timestamp difference written out after the log (864 probes). Reasoning: every ninth main item and every fifth certification item, answered after step-by-step reasoning within a budget of 512 new tokens and parsed for a final "Answer: X"; when the budget ran out first, the answer was forced by appending "Answer:" to the model's own reasoning. $A$ within $π$ on the arm's items, for the arm and for the same items in the corpus format.
