# -*- coding: utf-8 -*-
"""Real-trace anchor for ORDER (local, light): measured propagation latencies against the clock offsets of
common synchronisation regimes, computed exactly under Definition 1 by Monte Carlo.

Offsets: theta = theta_s + rho U, theta_s ~ N(0, sigma_s^2), rho uniform on [-rho_max, rho_max] (fixed per
device), U uniform on [0, T]. The offset difference is symmetric and unimodal, so at pi = 1/2 the Bayes rule is
the sign of D and its error is P(G + N < 0). Also reported: the fraction of measured cause-effect pairs whose
latency is below 2 delta for the hard bound delta = 3 sigma_s + rho_max T (Lemma 1).
python realdata_order.py <realdata_order.json> <hai_latency.json or -> <out.json>
"""
import json
import math
import sys

import numpy as np

rng = np.random.default_rng(1)
lat = json.load(open(sys.argv[1]))
sources = {"homes (CASAS): door -> motion": np.array(lat["casas_door_to_motion_s"], float)}
hai = None
if len(sys.argv) > 3 and sys.argv[2] != "-":
    h = json.load(open(sys.argv[2]))
    hai = np.concatenate([np.array(v, float) for v in h.values()])
elif lat.get("hai_demand_to_position"):
    hai = np.concatenate([np.array(p["lags_s"], float) for p in lat["hai_demand_to_position"]])
if hai is not None and len(hai):
    sources["industrial testbed (HAI): valve demand -> position"] = np.maximum(hai, 0.5)   # 1 s sampling
OUT = sys.argv[-1]
CLOCKS = {"XO +-20 ppm": 20e-6, "TCXO +-2 ppm": 2e-6, "disciplined (+-0.5 ppm residual)": 0.5e-6}
SYNC = {"NTP (10 ms)": 0.010, "coarse (100 ms)": 0.100}
T_GRID = np.exp(np.linspace(math.log(600), math.log(60 * 86400), 40))
NAMED = {"17 min (NTP maxpoll)": 1024, "1 h": 3600, "1 day": 86400, "1 week": 7 * 86400, "30 days": 30 * 86400}


def error(g, rho_max, s_s, T, n=200000):
    G = rng.choice(g, n)
    U = rng.uniform(0, T, (n, 2))
    rho = rng.uniform(-rho_max, rho_max, (n, 2))
    th = rng.normal(0, s_s, (n, 2)) + rho * U
    e = np.mean(G + th[:, 1] - th[:, 0] < 0)
    return float(e)


rows, curves = [], {}
for src, g in sources.items():
    for ck, rho_max in CLOCKS.items():
        for sk, s_s in SYNC.items():
            curves["%s | %s | %s" % (src, ck, sk)] = [[float(T), error(g, rho_max, s_s, T, 60000)] for T in T_GRID]
            for ik, T in NAMED.items():
                delta = 3 * s_s + rho_max * T
                sig = math.sqrt(s_s ** 2 + (rho_max / math.sqrt(3)) ** 2 * T ** 2 / 3)
                be = error(g, rho_max, s_s, T)
                rows.append({"source": src, "clock": ck, "sync": sk, "interval": ik, "sigma_theta_s": sig,
                             "bayes_error": be, "mc_se": float(math.sqrt(be * (1 - be) / 200000)), "delta_s": delta,
                             "share_uncertifiable": float(np.mean(g <= 2 * delta)),
                             "median_latency_s": float(np.median(g))})
out = {"rows": rows, "curves": curves, "n": {k: int(len(v)) for k, v in sources.items()},
       "quantiles": {k: [float(x) for x in np.percentile(v, [10, 25, 50, 75, 90])] for k, v in sources.items()}}
json.dump(out, open(OUT, "w"), indent=1)
for r in rows:
    if r["sync"].startswith("NTP"):
        print("%-52s %-12s %-8s sigma %.3f s  Bayes err %.3f  uncertifiable %.2f" % (
            r["source"], r["clock"], r["interval"], r["sigma_theta_s"], r["bayes_error"], r["share_uncertifiable"]))
print(out["n"], out["quantiles"])
