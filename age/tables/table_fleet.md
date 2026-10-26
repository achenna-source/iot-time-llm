Table: Re-read rules on real home sensors
| Fleet | Price $r$ | Single TTL | TTL per class | TTL per device state | Exit-rate rule | Two-parameter gate | Always re-read | Never re-read |
|---|---|---|---|---|---|---|---|---|
| Door contacts (6) | 0.02 | 0.38 | 0.38 | 0.09 | 0.55 | 0.17 | 1.00 | 1.91 |
|  | 0.1 | 1.00 | 1.00 | 0.10 | 3.03 | 0.61 | 6.95 | 1.00 |
|  | 0.3 | 1.00 | 1.00 | 0.03 | 27 | 0.22 | 84 | 1.00 |
| Motion sensors (133) | 0.02 | 1.17 | 1.17 | 0.37 | 2.38 | 0.32 | 2.42 | 1.00 |
|  | 0.1 | 1.00 | 1.00 | 0.57 | 17 | 1.10 | 19 | 1.00 |
|  | 0.3 | 1.00 | 1.00 | 0.47 | 61 | 0.60 | 79 | 1.00 |
| All devices (139) | 0.02 | 1.13 | 1.03 | 0.33 | 2.08 | 0.29 | 2.15 | 1.00 |
|  | 0.1 | 1.00 | 1.00 | 0.53 | 16 | 1.05 | 18 | 1.00 |
|  | 0.3 | 1.00 | 1.00 | 0.45 | 60 | 0.59 | 79 | 1.00 |
| All devices (139), uniform requests | 0.02 | 1.00 | 0.99 | 0.35 | 0.97 | 0.15 | 1.00 | 7.65 |
|  | 0.1 | 0.95 | 0.96 | 0.07 | 0.88 | 0.08 | 1.00 | 1.02 |
|  | 0.3 | 1.00 | 1.00 | 0.10 | 3.15 | 0.06 | 4.19 | 1.00 |
Note: Normalised regret on 139 CASAS devices. Every rule and every time-to-live is fitted on the first half of each device record and evaluated on the second half; a request finds the device in each state in proportion to the time it spends there, at ages from one minute to three days. 0 is the empirical optimum of the test half and 1 the better of the two fixed rules. The two-parameter gate applies Theorem 1 with the exit rate and time share fitted on the first half. The renewal rule of Proposition 4(ii), re-read when the age exceeds $r$ mean sojourns, is within 5% of the exit-rate rule in every cell. The last block weights every device, state and age equally. Office occupancy (UCI) is omitted because its recordings are too short to split.
