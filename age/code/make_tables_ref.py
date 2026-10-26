# -*- coding: utf-8 -*-
"""Table 1 of the AGE paper: reference policies on the main design (from reference_bounds.json)."""
import json
import sys

ref = json.load(open(sys.argv[1]))
order = [("always read", "Always re-read"), ("never read (act on memory)", "Never re-read (act on the reading)"),
         ("best single TTL", "Best single time-to-live (hindsight)"), ("best TTL per cost level", "Best time-to-live per read price"),
         ("timestamp-blind", "Best rule without the age (volatility and price known)"),
         ("exit-rate gate", "Exit-rate rule of Proposition 4"), ("optimal", "Optimal policy (Theorem 1)"),
         ("J ceiling", "Best prompt-only threshold for J"),
         ("oracle", "Oracle (sees the realised change)")]
rows = ["Table: Reference policies on the main grid",
        "| Policy | Re-read rate | $J$ | $ρ$, $r = 0.02$ | $ρ$, $r = 0.1$ | $ρ$, $r = 0.3$ | Excess cost, $r = 0.6$ |", "|---|---|---|---|---|---|---|"]


def f(v):
    return "n/a" if v is None else "%.2f" % v


for key, lab in order:
    k = next(k for k in ref if k.startswith(key))
    v = ref[k]
    if "rho" in v:
        pp = v["per_price"]
        rows.append("| %s | %.2f | %+.3f | %s | %s | %s | %.3f |" % (lab, v["read_rate"], v["J"], f(pp["0.02"]["rho"]), f(pp["0.1"]["rho"]),
                                                               f(pp["0.3"]["rho"]), pp["0.6"]["excess"]))
    else:
        rows.append("| %s | n/a | +1.000 | n/a | n/a | n/a | n/a |" % lab)
rows.append("Note: 8,640 probes. $ρ$: normalised regret at each read price (0 = optimal, 1 = the better fixed rule at that price). At $r = 0.6$ never re-reading is optimal, so $ρ$ is undefined and the excess expected cost over the optimum, in units of the wrong-command cost $L$, is given instead. $J$: staleness sensitivity index. Fixed rules score $J = 0$ by construction. The time-to-live rules are chosen on the corpus itself, so they are optimistic for that family; the single time-to-live uses one age threshold for every price. The exit-rate rule ignores the time share of the remembered state and re-reads at $r = 0.6$, where re-reading is never optimal.")
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(rows) + "\n")
print("\n".join(rows))
