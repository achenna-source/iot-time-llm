# -*- coding: utf-8 -*-
"""Reference policies on the AGE main design: fixed rules, TTL rules, the exit-rate gate and the optimum."""
import json
import math
import sys

import numpy as np

from metrics import regret_summary, youden

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
main = [r for r in rows if r["part"] == "main"]
p = np.array([r["p_change"] for r in main])
rr = np.array([r["r"] for r in main])
age = np.array([r["age_s"] for r in main], float)
q = np.array([r["q_day"] for r in main]) / 86400.0
one, zero = np.ones_like(p), np.zeros_like(p)


def det(read):
    read = read.astype(float)
    out = regret_summary(read, 1 - read, zero, p, rr)
    pp = {}
    for v in sorted(set(rr)):
        m = rr == v
        s_ = regret_summary(read[m], 1 - read[m], zero[m], p[m], rr[m])
        gap = s_["cost_best_fixed"] - s_["cost_opt"]
        pp[str(v)] = {"rho": float(s_["rho"]) if gap > 1e-9 else None, "excess": float(s_["cost"] - s_["cost_opt"]),
                      "read": float(read[m].mean())}
    out["per_price"] = pp
    return out


pol = {}
pol["always read"] = det(one > 0)
pol["never read (act on memory)"] = det(zero > 0)
pol["optimal (read iff p > r)"] = det(p > rr)
pol["exit-rate gate (read iff qD > -ln(1-r))"] = det(q * age > -np.log(1 - rr))
# best single TTL: volatility-blind and cost-blind
best = None
for tau in sorted(set(age)) + [0, 1e9]:
    s = det(age > tau - 1e-9)
    if best is None or s["cost"] < best[1]["cost"]:
        best = (tau, s)
pol["best single TTL (tau = %g s)" % best[0]] = best[1]
# best TTL per cost level (cost-aware, volatility-blind)
read = np.zeros_like(p, bool)
for rv in sorted(set(rr)):
    m = rr == rv
    bt = min((np.mean(np.where(age[m] > t - 1e-9, rv, p[m])), t) for t in sorted(set(age[m])) + [1e9])
    read[m] = age[m] > bt[1] - 1e-9
pol["best TTL per cost level"] = det(read)
# timestamp-blind but volatility- and cost-aware: best constant decision per (q, r)
read = np.zeros_like(p, bool)
for qv in sorted(set(q)):
    for rv in sorted(set(rr)):
        m = (q == qv) & (rr == rv)
        read[m] = rv < p[m].mean()
pol["timestamp-blind (best per volatility and cost)"] = det(read)
# prompt-only ceiling of J: best threshold on the staleness probability (maximises J, not cost)
jbest = max((youden((p > c).astype(float), p), c) for c in np.r_[0, np.unique(p)])
pol["J ceiling for prompt-only policies (threshold %.3f)" % jbest[1]] = det(p > jbest[1])
pol["oracle (knows the realised change)"] = {"J": 1.0}
for k, v in pol.items():
    print("%-50s J %+.3f  rho %s  read %.2f" % (k, v["J"], ("%.3f" % v["rho"]) if "rho" in v else "  -  ",
                                               v.get("read_rate", float("nan"))))
json.dump(pol, open(sys.argv[2], "w"), indent=1)
