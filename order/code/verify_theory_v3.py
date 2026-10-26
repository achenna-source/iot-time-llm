# -*- coding: utf-8 -*-
"""Checks of the ORDER results revised after the simulated review (local, CPU).

T1' I(H; D) <= E_{G,V}[ 1/2 ln(1 + 4 pi (1 - pi) G^2 / V) ] <= pi (1 - pi) kappa_eff^2, and <= H_b(pi)
T2' genie-aided floor: P_e* >= E_{G,V}[ P_e*(G, V; pi) ] >= 2 (1 - pi) E[Phi(-G / sqrt V)] >= 2 (1 - pi) Phi(-kappa_eff / sqrt2)
    with equality of the first inequality at pi = 1/2 for any latency law and any variance law
    (checked with heavy-tailed latencies and the drift model)
L1' with a known minimum latency g_min > 0, D > 2 delta - g_min certifies A before B
"""
import json
import math
import sys

import numpy as np
from scipy import stats

rng = np.random.default_rng(17)
Phi = stats.norm.cdf
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))


def pe_fixed(g, v, pi):
    """Bayes error for D ~ N(+g, v) vs N(-g, v) with prior pi on +g."""
    if pi == 0.5:
        return Phi(-g / math.sqrt(v))
    dstar = (v / (2 * g)) * math.log((1 - pi) / pi)
    return pi * Phi((dstar - g) / math.sqrt(v)) + (1 - pi) * (1 - Phi((dstar + g) / math.sqrt(v)))


def mixture_bayes(G, V, pi, n_grid=6001):
    span = 8 * math.sqrt(V.max()) + G.max()
    grid = np.linspace(-span, span, n_grid)
    dx = grid[1] - grid[0]
    pab = np.zeros_like(grid)
    for g, v in zip(G, V):
        pab += stats.norm.pdf(grid, g, math.sqrt(v))
    pab /= len(G)
    pba = pab[::-1]
    err = float(np.sum(np.minimum(pi * pab, (1 - pi) * pba)) * dx)
    mix = pi * pab + (1 - pi) * pba
    mi = float(np.sum(pi * pab * np.log(np.maximum(pab, 1e-300) / np.maximum(mix, 1e-300))
                      + (1 - pi) * pba * np.log(np.maximum(pba, 1e-300) / np.maximum(mix, 1e-300))) * dx)
    return err, mi


settings = {
    "lognormal latency, fixed V": (lambda n: rng.lognormal(0, 1.2, n), lambda n: np.full(n, 2 * 0.58 ** 2)),
    "lognormal + 5% tail to 20 s": (lambda n: np.where(rng.random(n) < 0.05, rng.uniform(5, 20, n), rng.lognormal(-0.2, 0.9, n)),
                                    lambda n: np.full(n, 2 * 0.58 ** 2)),
    "deterministic 1 s, drift model": (lambda n: np.full(n, 1.0),
                                       lambda n: 2 * 0.05 ** 2 + 1.0 ** 2 * (rng.uniform(0, 1, n) ** 2 + rng.uniform(0, 1, n) ** 2)),
    "exponential mean 0.5 s, drift model": (lambda n: rng.exponential(0.5, n),
                                            lambda n: 2 * 0.1 ** 2 + 1.2 ** 2 * (rng.uniform(0, 1, n) ** 2 + rng.uniform(0, 1, n) ** 2)),
}
rows = []
viol = 0
for name, (dg, dv) in settings.items():
    n = 2500
    G, V = dg(n), dv(n)
    k_eff = math.sqrt(2 * np.mean(G ** 2) * np.mean(1 / V))
    for pi in (0.5, 0.7, 0.9):
        err, mi = mixture_bayes(G, V, pi)
        genie = float(np.mean([pe_fixed(g, v, pi) for g, v in zip(G, V)]))
        mid = float(2 * (1 - pi) * np.mean(Phi(-G / np.sqrt(V))))
        closed = float(2 * (1 - pi) * Phi(-k_eff / math.sqrt(2)))
        mi_awgn = float(np.mean(0.5 * np.log(1 + 4 * pi * (1 - pi) * G ** 2 / V)))
        hb = -(pi * math.log(pi) + (1 - pi) * math.log(1 - pi))
        rows.append({"setting": name, "pi": pi, "bayes": err, "genie": genie, "mid": mid, "closed": closed,
                     "kappa_eff": k_eff, "MI": mi, "MI_awgn": mi_awgn, "MI_quad": pi * (1 - pi) * k_eff ** 2, "Hb": hb})
        if not (err >= genie - 3e-3 and genie >= mid - 1e-9 and mid >= closed - 1e-9):
            viol += 1
        if not (mi <= mi_awgn + 3e-3 and mi_awgn <= pi * (1 - pi) * k_eff ** 2 + 1e-9 and mi <= hb + 1e-9):
            viol += 1
        print("   %-38s pi %.1f  Bayes %.4f >= genie %.4f >= mid %.4f >= closed %.4f | MI %.4f <= %.4f <= %.4f"
              % (name, pi, err, genie, mid, closed, mi, mi_awgn, pi * (1 - pi) * k_eff ** 2))
check("T2' chain Bayes >= genie >= 2(1-pi)E[Phi] >= closed form, and T1' MI chain (4 settings x 3 priors)", viol == 0)
eq = [abs(r["bayes"] - r["genie"]) < 3e-3 for r in rows if r["pi"] == 0.5]
check("T2' genie-aided floor attained at pi = 1/2 in every setting (heavy tails, drift model)", all(eq))

# L1' minimum latency
viol = 0
delta, gmin = 0.5, 0.3
for _ in range(200000):
    a_first = rng.random() < 0.5
    G = gmin + rng.exponential(1.0)
    th = rng.uniform(-delta, delta, 2)
    D = (G if a_first else -G) + th[1] - th[0]
    if D > 2 * delta - gmin and not a_first:
        viol += 1
check("L1' D > 2 delta - g_min never certifies the wrong order", viol == 0)
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "theory_checks_v3.json", "w"), indent=1)
print("ALL PASS" if ok else "SOME CHECKS FAILED")
