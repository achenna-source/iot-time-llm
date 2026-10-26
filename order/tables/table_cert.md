Table: Certification under a hard offset bound $δ$ (240 probes; prefilled scoring). Overclaim: the model asserts an order when $|D| < 2δ$; underclaim: it declines when $|D| > 2δ$; misorder: it asserts the order opposite to the timestamps.
| Model | Accuracy | Overclaim | Underclaim | Misorder |
|---|---|---|---|---|
| Qwen2.5-7B | 0.59 | 0.04 | 0.96 | 0.02 |
| Qwen3-8B | 0.33 | 0.74 | 0.29 | 0.27 |
| Llama-3.1-8B | 0.45 | 0.48 | 0.47 | 0.23 |
| Granite-3.3-8B | 0.60 | 0.00 | 1.00 | 0.00 |
| OLMo-2-7B | 0.48 | 0.33 | 0.59 | 0.19 |
| Qwen2.5-14B | 0.60 | 0.06 | 0.93 | 0.00 |
| Order certifier (Lemma 1) | 1.00 | 0.00 | 0.00 | 0.00 |
| Always "not established" | 0.60 | 0.00 | 1.00 | 0.00 |
| Always the timestamp order | 0.40 | 1.00 | 0.00 | 0.00 |
