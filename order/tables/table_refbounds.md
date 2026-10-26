Table: Reference policies on the main part
| Policy | Accuracy | acc($S_{ts}$) | acc($S_{hist}$) | $A$ | $A$ within $π$ (0.7 / 0.9) |
|---|---|---|---|---|---|
| Follow the earlier timestamp | 0.762 | 1.000 | 0.000 | +0.000 | +0.000 / +0.000 |
| Follow the maintenance history | 0.796 | 0.000 | 1.000 | +0.000 | +0.000 / +0.000 |
| Follow the first-listed line | 0.508 | 0.510 | 0.488 | −0.003 | −0.035 / +0.038 |
| Timestamps when $g = 2σ_{θ}$, history otherwise | 0.840 | 0.415 | 0.872 | +0.288 | +0.291 / +0.263 |
| Calibrated analyst (answers with the posterior probability) | 0.804 | 0.630 | 0.683 | +0.313 | +0.226 / +0.263 |
| Bayes attributor (Corollary 1) | 0.858 | 0.631 | 0.831 | +0.463 | +0.389 / +0.350 |
| Oracle (knows the true order) | 1.000 | 1.000 | 1.000 | +1.000 | n/a |
Note: 1,728 probes. $A$: arbitration index (Definition 2), with Rao-Blackwellised labels except for the oracle; within $π$: the index computed separately for the two history priors (0.7 and 0.9). The coarse rule follows the timestamps only in the cells where the latency is twice the offset spread. Certification part (240 probes): the order certifier of Lemma 1 is exact (accuracy 1.00), always reporting the timestamp order scores 0.40 and always abstaining 0.60.
