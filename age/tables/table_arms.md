Table: Re-read decisions when the model is helped
| Model | Arm | Re-read, $r = 0.6$ | $ρ$, $r = 0.1$ | $ρ$, $r = 0.3$ | $J$ |
|---|---|---|---|---|---|
| Qwen2.5-7B | As in the corpus | 100 | 1.00 | 1.90 | +0.000 |
|  | Time share stated | 100 | 1.00 | 1.90 | +0.000 |
|  | Decision rule stated | 100 | 1.00 | 1.90 | +0.000 |
|  | Staleness probability stated | 100 | 1.00 | 1.90 | +0.000 |
|  | Step-by-step reasoning (92% answered within budget) | 67 | 5.64 | 1.94 | +0.097 |
| Qwen3-8B | As in the corpus | 91 | 1.54 | 1.83 | +0.044 |
|  | Time share stated | 74 | 2.16 | 1.75 | +0.098 |
|  | Decision rule stated | 96 | 1.46 | 1.97 | +0.015 |
|  | Staleness probability stated | 97 | 1.46 | 1.89 | +0.012 |
|  | Step-by-step reasoning (44% answered within budget) | 53 | 2.68 | 2.24 | +0.179 |
| Llama-3.1-8B | As in the corpus | 99 | 1.15 | 1.90 | −0.004 |
|  | Time share stated | 99 | 1.00 | 1.90 | −0.000 |
|  | Decision rule stated | 97 | 1.00 | 1.90 | +0.005 |
|  | Staleness probability stated | 100 | 1.00 | 1.90 | +0.000 |
|  | Step-by-step reasoning (85% answered within budget) | 50 | 4.36 | 2.31 | +0.054 |
| Granite-3.3-8B | As in the corpus | 48 | 3.49 | 2.00 | +0.028 |
|  | Time share stated | 43 | 4.75 | 1.75 | +0.026 |
|  | Decision rule stated | 76 | 3.59 | 1.96 | −0.006 |
|  | Staleness probability stated | 66 | 3.00 | 1.63 | +0.027 |
|  | Step-by-step reasoning (99% answered within budget) | 50 | 6.52 | 1.79 | +0.006 |
| OLMo-2-7B | As in the corpus | 78 | 2.22 | 2.42 | +0.044 |
|  | Time share stated | 72 | 2.90 | 2.89 | +0.019 |
|  | Decision rule stated | 67 | 5.99 | 3.56 | +0.000 |
|  | Staleness probability stated | 73 | 2.83 | 2.91 | +0.018 |
|  | Step-by-step reasoning (71% answered within budget) | 13 | 11 | 1.86 | −0.036 |
| Qwen2.5-14B (6 scenarios) | As in the corpus | 89 | 1.69 | 3.29 | +0.003 |
|  | Time share stated | 87 | 0.96 | 1.78 | +0.048 |
|  | Decision rule stated | 91 | 0.82 | 1.67 | +0.101 |
|  | Staleness probability stated | 97 | 0.93 | 1.82 | +0.022 |
|  | Step-by-step reasoning (85% answered within budget) | 77 | 0.89 | 1.77 | +0.047 |
| Optimal policy | | 0 | 0.00 | 0.00 | +0.314 |
Note: One probe per cell of the main grid (120 cells; three option rotations, except for reasoning, which uses one rotation and free generation within a budget of 512 new tokens, parsed for a final answer or, when the budget ran out, forced by appending "Answer:" to the model's own reasoning). Re-read rate at $r = 0.6$, where re-reading is never optimal, and normalised regret at $r = 0.1$ and $r = 0.3$.
