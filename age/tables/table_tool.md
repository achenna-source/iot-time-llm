Table: First turn with native tools
| Model | Structured | Read only | Command only | Read and command | No call | Agreement | $κ$ |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 100 | 100 | 0 | 0 | 0 | 100 | n/a |
| Qwen3-8B | 90 | 84 | 6 | 0 | 10 | 81 | 0.27 |
| Llama-3.1-8B | 100 | 43 | 0 | 57 | 0 | 43 | 0.00 |
| Granite-3.3-8B | 0 | 6 | 38 | 2 | 54 | 37 | 0.08 |
| OLMo-2-7B | 0 | 22 | 3 | 0 | 75 | 22 | −0.04 |
| Qwen2.5-14B (6 scenarios) | 98 | 88 | 10 | 0 | 2 | 89 | 0.13 |
Note: 216 situations, no option list. Responses are parsed with each model's structured tool-call format, with a fallback for calls written as text. Read and command: both calls emitted in the same turn, so that the command does not wait for the read. Agreement and Cohen's $κ$ compare the first turn, collapsed to read, act on the reading or other, with the letter decision in the same situation. Rates in percent. $κ$ is undefined when one of the two classifications is constant.
