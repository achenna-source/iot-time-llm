# -*- coding: utf-8 -*-
"""Table 1 of the ORDER paper: reference policies on the main part (from reference_bounds.json)."""
import json
import sys

ref = json.load(open(sys.argv[1]))
order = [("earliest timestamp", "Follow the earlier timestamp"), ("maintenance history", "Follow the maintenance history"),
         ("first listed line", "Follow the first-listed line"),
         ("coarse rule", "Timestamps when $g = 2σ_{θ}$, history otherwise"),
         ("calibrated Bayes", "Calibrated analyst (answers with the posterior probability)"),
         ("Bayes attributor", "Bayes attributor (Corollary 1)"),
         ("oracle", "Oracle (knows the true order)")]
rows = ["Table: Reference policies on the main part",
        "| Policy | Accuracy | acc($S_{ts}$) | acc($S_{hist}$) | $A$ | $A$ within $π$ (0.7 / 0.9) |", "|---|---|---|---|---|---|"]
for key, lab in order:
    k = next(k for k in ref if k.startswith(key))
    v = ref[k]["realised"] if key == "oracle" else ref[k]["soft"]
    wp = ref[k].get("A_within_pi", [float("nan")] * 2)
    wps = "n/a" if key == "oracle" else "%+.3f / %+.3f" % tuple(wp)
    rows.append(("| %s | %.3f | %.3f | %.3f | %+.3f | %s |" % (lab, ref[k]["realised"]["acc"], v["acc_S_ts"], v["acc_S_mech"], v["A"], wps)).replace("-0.", "−0."))
cert = {k: v for k, v in ref.items() if k.startswith("cert")}
rows.append("Note: 1,728 probes. $A$: arbitration index (Definition 2), with Rao-Blackwellised labels except for the oracle; within $π$: the index computed separately for the two history priors (0.7 and 0.9). The coarse rule follows the timestamps only in the cells where the latency is twice the offset spread. Certification part (240 probes): the order certifier of Lemma 1 is exact (accuracy %.2f), always reporting the timestamp order scores %.2f and always abstaining %.2f."
            % (cert["cert: 2-delta order certifier"]["acc"], cert["cert: sign of D (never abstain)"]["acc"], cert["cert: always undetermined"]["acc"]))
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(rows) + "\n")
print("\n".join(rows))
