# -*- coding: utf-8 -*-
"""Reference policies on the ORDER corpus: cue-following rules, the Bayes attributor, the 2-delta order certifier."""
import json
import sys

import numpy as np

from metrics import arbitration, directions, position_reliance

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
main = [r for r in rows if r["part"] == "main"]
ts_A, mech_A = directions(main)
post = np.array([r["post_A"] for r in main])
la = np.array([r["list_a_first"] for r in main], float)

pol = {
    "earliest timestamp": ts_A.astype(float),
    "maintenance history": mech_A.astype(float),
    "first listed line": la,
    "Bayes attributor (posterior > 1/2)": (post > 0.5).astype(float),
    "oracle": np.array([1.0 if r["a_first"] else 0.0 for r in main]),
    "calibrated Bayes analyst (answers A with probability posterior)": post,
    "coarse rule (timestamps if g = 2 sigma, else history)": np.where(np.array([r["gamma"] for r in main]) == 2.0, ts_A, mech_A).astype(float),
}
out = {}
for k, P in pol.items():
    soft = arbitration(P, main, soft=True)
    real = arbitration(P, main, soft=False)
    pis = np.array([r["pi"] for r in main])
    wp = [arbitration(P[pis == v], [main[i] for i in np.where(pis == v)[0]], soft=True)["A"] for v in (0.7, 0.9)]
    out[k] = {"soft": soft, "realised": real, "position": position_reliance(P, main), "A_within_pi": wp}
    print("%-36s A_soft %+.3f  A_real %+.3f  acc %.3f  pos %+.3f" % (k, soft["A"], real["A"], real["acc"],
                                                                    out[k]["position"]))
for g in sorted(set(r["gamma"] for r in main)):
    m = [r for r in main if r["gamma"] == g]
    P = np.array([r["post_A"] > 0.5 for r in m], float)
    print("   gamma %.1f  Bayes A_soft %+.3f   n_S_ts %.0f  n_S_mech %.0f" % (
        g, arbitration(P, m)["A"], arbitration(P, m)["n_S_ts"], arbitration(P, m)["n_S_mech"]))
cert = [r for r in rows if r["part"] == "cert"]
rules = {"2-delta order certifier": lambda r: "undetermined" if abs(r["D"]) <= 2 * r["delta"] else ("A_first" if r["D"] > 0 else "B_first"),
         "sign of D (never abstain)": lambda r: "A_first" if r["D"] > 0 else "B_first",
         "always undetermined": lambda r: "undetermined"}
for k, f in rules.items():
    acc = np.mean([f(r) == r["correct"] for r in cert])
    out["cert: " + k] = {"acc": acc}
    print("cert %-28s acc %.3f" % (k, acc))
json.dump(out, open(sys.argv[2], "w"), indent=1)
