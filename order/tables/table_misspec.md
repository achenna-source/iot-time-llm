Table: The attributor under a misdeclared clock model
| Departure from the declared model | Attributor accuracy | Attributor $A$ | True Bayes accuracy | True Bayes $A$ | Earliest timestamp accuracy |
|---|---|---|---|---|---|
| Nominal | 0.869 | +0.283 | 0.869 | +0.284 | 0.773 |
| Offsets heavy-tailed (t, 3 degrees of freedom) | 0.876 | +0.223 | 0.883 | +0.246 | 0.821 |
| 5% offsets with 10× spread | 0.842 | +0.174 | 0.856 | +0.239 | 0.751 |
| Latency exponential, declared mean | 0.835 | +0.214 | 0.842 | +0.231 | 0.717 |
| Latency log-normal (cv 1), declared mean | 0.841 | +0.225 | 0.845 | +0.237 | 0.725 |
| Offset spread declared 0.5× | 0.829 | +0.173 | 0.869 | +0.280 | 0.773 |
| Offset spread declared 2× | 0.832 | +0.113 | 0.869 | +0.281 | 0.773 |
| Latency declared 0.5× | 0.857 | +0.228 | 0.869 | +0.284 | 0.774 |
| Latency declared 2× | 0.857 | +0.250 | 0.869 | +0.282 | 0.773 |
| Prior declared 0.9 when 0.7, 0.7 when 0.9 | 0.841 | +0.282 | 0.869 | +0.280 | 0.773 |
| Prior ignored (declared 1/2) | 0.773 | +0.000 | 0.869 | +0.283 | 0.773 |
Note: Realised accuracy and arbitration index $A$ over the main grid ($π$ of 0.7 or 0.9, $g/σ_{θ}$ of 0.5, 1 or 2; 200,000 simulated incidents per cell). True Bayes: the optimal rule for the process that actually generated the data. The maintenance-history rule has accuracy 0.800 and $A = 0$ in every row; the earliest-timestamp rule has $A = 0$ by construction. Ignoring the prior turns the attributor into the earliest-timestamp rule.
