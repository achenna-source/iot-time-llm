# -*- coding: utf-8 -*-
"""Sensitivity of the freshness gate to misestimated fields (local, light; no language model involved).

On the main grid of ReRead-IoT (change rates 0.25 to 64 per day, six ages, four prices), the gate decides with
the DECLARED fields (change rate, time share 1/2, price) while the device follows a different process:
change rate 2x or 0.5x the declared one, remembered state rarer or commoner than declared (time share of the
other state 0.25 or 0.75), or a price misjudged by a factor of 2. Reported per price: normalised regret of the
gate against the optimum of the TRUE process, and of the two fixed rules.
python gate_misspec.py <probes.jsonl> <out.json>
"""
import json
import math
import sys

import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
main = [r for r in rows if r["part"] == "main"]
cells = sorted({(r["q_day"], r["age_s"], r["r"]) for r in main})


def p_true(q_day, age_s, psi):
    """binary Markov device with long-run change rate q_day per day; psi = time share of the other state."""
    pi_x = 1 - psi
    q_x = q_day / (2 * pi_x) / 86400.0          # pi_x q_x = (changes per unit time) / 2
    kap = q_x / psi
    return psi * (1 - math.exp(-kap * age_s))


SCEN = {"declared fields correct": {}, "change rate 2x the declared": {"qk": 2.0},
        "change rate 0.5x the declared": {"qk": 0.5}, "other state 75% of the time (declared 50%)": {"psi": 0.75},
        "other state 25% of the time (declared 50%)": {"psi": 0.25}, "price judged 2x too high": {"rk": 2.0},
        "price judged 2x too low": {"rk": 0.5}}
out = {}
for name, kw in SCEN.items():
    res = {}
    for r in sorted({c[2] for c in cells}):
        cc = [c for c in cells if c[2] == r]
        pt = np.array([p_true(q * kw.get("qk", 1.0), a, kw.get("psi", 0.5)) for q, a, _ in cc])
        pg = np.array([p_true(q, a, 0.5) for q, a, _ in cc])                    # what the gate believes
        read = pg > r * kw.get("rk", 1.0)
        cost = lambda rd: float(np.mean(np.where(rd, r, pt)))
        c_opt = float(np.mean(np.minimum(pt, r)))
        c_fix = min(r, float(np.mean(pt)))
        rho = (cost(read) - c_opt) / (c_fix - c_opt) if c_fix - c_opt > 1e-12 else float("nan")
        res[str(r)] = {"gate_rho": rho, "gate_cost": cost(read), "opt_cost": c_opt, "always": r,
                       "never": float(np.mean(pt)), "gate_read": float(read.mean()), "opt_read": float((pt > r).mean())}
    out[name] = res
    print("%-46s " % name + "  ".join("r=%s rho %.2f" % (k, v["gate_rho"]) for k, v in res.items()))
json.dump(out, open(sys.argv[2], "w"), indent=1)
