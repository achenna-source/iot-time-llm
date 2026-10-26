# -*- coding: utf-8 -*-
"""Misspecification stress test for the relocated attributor (local, light; no language model involved).

The attributor of Section 4 decides from log-odds logit(pi_d) + g_d D / sigma_d^2 with DECLARED parameters
(pi_d, g_d, sigma_d). Here the true generative process departs from what is declared, and we measure the realised
accuracy and arbitration index over the main grid of the corpus (pi in {0.7, 0.9}, gamma = g / sigma in
{0.5, 1, 2}, favoured direction balanced), against the two cue-following rules and the correctly specified Bayes
rule. sigma = 1 without loss of generality (only gamma matters).
Truth departures: heavy-tailed offsets (Student t, 3 dof, same standard deviation), 5% of offsets with a 10x
spread (a missed synchronisation), random latency (exponential or log-normal with the declared mean).
Declaration errors: sigma declared 0.5x or 2x, g declared 0.5x or 2x, prior declared 0.9 when it is 0.7 and the
converse, prior ignored (declared 1/2).
python misspec_stress.py <out.json>
"""
import json
import math
import sys

import numpy as np

rng = np.random.default_rng(11)
N = 200000
PIS, GAMMAS = (0.7, 0.9), (0.5, 1.0, 2.0)


def offsets(kind, n):
    if kind == "t3":
        return rng.standard_t(3, n) / math.sqrt(3.0)           # unit standard deviation
    if kind == "outliers":
        s = np.where(rng.random(n) < 0.05, 10.0, 1.0)
        return rng.normal(0, 1, n) * s
    return rng.normal(0, 1, n)


def latency(kind, g, n):
    if kind == "exp":
        return rng.exponential(g, n)
    if kind == "lognormal":                                   # coefficient of variation 1, mean g
        s2 = math.log(2.0)
        return np.exp(rng.normal(math.log(g) - s2 / 2, math.sqrt(s2), n))
    return np.full(n, g)


def simulate(pi, gamma, off="gauss", lat="fixed", sig_x=1.0, g_x=1.0, pi_decl=None):
    g = gamma
    fav = rng.random(N) < 0.5                       # favoured event is A
    prior_A = np.where(fav, pi, 1 - pi)
    a_first = rng.random(N) < prior_A
    L = latency(lat, g, N)
    D = np.where(a_first, L, -L) + offsets(off, N) - offsets(off, N)
    pd = pi if pi_decl is None else pi_decl
    pdA = np.where(fav, pd, 1 - pd)
    lo_decl = np.log(pdA / (1 - pdA)) + (g * g_x) * D / (sig_x ** 2)
    lo_true = np.log(prior_A / (1 - prior_A)) + g * D            # Bayes under the nominal model
    # Bayes rule under the TRUE process: log-likelihood ratio from a histogram density of D given A first
    M = 4000000
    W = np.sort(latency(lat, g, M) + offsets(off, M) - offsets(off, M))
    edges = np.linspace(-40, 40, 8001)
    h = np.histogram(W, edges)[0] + 0.5
    lf = lambda x: np.log(h[np.clip(np.searchsorted(edges, x) - 1, 0, len(h) - 1)])
    lo_tb = np.log(prior_A / (1 - prior_A)) + lf(D) - lf(-D)
    dec = {"attributor (declared)": lo_decl > 0, "Bayes (nominal model)": lo_true > 0, "Bayes (true process)": lo_tb > 0,
           "earliest timestamp": D > 0, "maintenance history": fav}
    ts_A = D > 0
    dis = ts_A != fav
    s_ts = dis & (ts_A == a_first)
    s_me = dis & (fav == a_first)
    out = {}
    for k, d in dec.items():
        ok = d == a_first
        out[k] = {"acc": float(ok.mean()), "A": float(ok[s_ts].mean() + ok[s_me].mean() - 1)}
    return out


SCEN = [("nominal", {}),
        ("offsets heavy-tailed (t, 3 dof)", {"off": "t3"}),
        ("5% offsets with 10x spread", {"off": "outliers"}),
        ("latency exponential, declared mean", {"lat": "exp"}),
        ("latency log-normal (cv 1), declared mean", {"lat": "lognormal"}),
        ("sigma declared 0.5x", {"sig_x": 0.5}),
        ("sigma declared 2x", {"sig_x": 2.0}),
        ("latency declared 0.5x", {"g_x": 0.5}),
        ("latency declared 2x", {"g_x": 2.0}),
        ("prior declared 0.9 when 0.7, 0.7 when 0.9", {"pi_decl": "swap"}),
        ("prior ignored (declared 1/2)", {"pi_decl": 0.5})]

res = {}
for name, kw in SCEN:
    cells = []
    for pi in PIS:
        for gm in GAMMAS:
            k = dict(kw)
            if k.get("pi_decl") == "swap":
                k["pi_decl"] = 0.9 if pi == 0.7 else 0.7
            cells.append(simulate(pi, gm, **k))
    agg = {pol: {m: float(np.mean([c[pol][m] for c in cells])) for m in ("acc", "A")} for pol in cells[0]}
    agg["by_cell"] = cells
    res[name] = agg
    a = agg["attributor (declared)"]
    print("%-44s attributor acc %.3f A %+.3f | true Bayes acc %.3f A %+.3f | ts acc %.3f | hist acc %.3f" % (
        name, a["acc"], a["A"], agg["Bayes (true process)"]["acc"], agg["Bayes (true process)"]["A"],
        agg["earliest timestamp"]["acc"], agg["maintenance history"]["acc"]))
json.dump({"N_per_cell": N, "pis": PIS, "gammas": GAMMAS, "scenarios": res}, open(sys.argv[1], "w"), indent=1)
