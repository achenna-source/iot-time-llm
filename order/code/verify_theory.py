# -*- coding: utf-8 -*-
"""Numerical checks of every formal statement in the ORDER paper (runs locally, CPU, seconds).

L1  bounded skew |theta| <= delta: precedence certified iff |D| > 2 delta, and the bound is tight
T1  KL(P_AB || P_BA) <= E[G^2] / sigma^2 and I(H; D) <= 2 pi (1 - pi) E[G^2] / sigma^2
T2  Bayes error of ANY timestamp-based attribution >= (1 - pi)(1 - TV) >= (1 - pi)(1 - rho / sqrt2)
T2b exact Bayes error for deterministic latency and Gaussian offsets
T3  fan-out: P(earliest timestamp = root) = int phi(z) prod_j Phi(g_j/sigma - z) dz -> 0 as k grows
T4  offset variance with drift: sigma_theta^2(T) = sigma_s^2 + sigma_rho^2 T^2 / 3; maximal sync interval
Offsets theta ~ N(0, sigma^2) per device, independent; D = tau_B - tau_A.
"""
import json
import math
import sys

import numpy as np
from scipy import integrate, stats

rng = np.random.default_rng(11)
ok = True
out = {}


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))


# ---------------------------------------------------------------- L1 certifiability
delta = 0.5
viol = 0
for _ in range(20000):
    tA = 0.0
    tB = rng.uniform(-3, 3)
    th = rng.uniform(-delta, delta, 2)
    D = (tB + th[1]) - (tA + th[0])
    if abs(D) > 2 * delta and np.sign(D) != np.sign(tB - tA):
        viol += 1
# witness: any |D| < 2 delta is consistent with both orders (boundary |D| = 2 delta has measure zero)
wit = 0
for D in np.linspace(-2 * delta, 2 * delta, 43)[1:-1]:
    # order A before B: tB - tA = eps > 0, offsets thA, thB with thB - thA = D - eps, |th| <= delta
    eps = 1e-9
    need = D - eps
    okAB = abs(need) <= 2 * delta
    need2 = D + eps          # order B before A: tB - tA = -eps
    okBA = abs(need2) <= 2 * delta + 1e-12
    wit += okAB and okBA
check("L1 |D| > 2 delta never misorders; every |D| <= 2 delta is consistent with both orders",
      viol == 0 and wit == 41)

# ---------------------------------------------------------------- latency families with E[G^2]
sigma = 1.0
families = {
    "deterministic": (lambda g: None, None),
}


def mix_pdf(d, sampler_pdf, sign, s2):
    """density of sign*G + N(0, s2) for latency density sampler_pdf on (0, inf)."""
    f = lambda g: sampler_pdf(g) * stats.norm.pdf(d, sign * g, math.sqrt(s2))
    return integrate.quad(f, 0, np.inf, limit=200)[0]


lat = {
    "exponential(mean 1)": (lambda g: math.exp(-g), 2.0),
    "gamma(k=4, mean 1)": (lambda g: stats.gamma.pdf(g, 4, scale=0.25), 1.0 + 0.25),
    "uniform(0, 2)": (lambda g: 0.5 if g <= 2 else 0.0, 4.0 / 3.0),
    "lognormal(mu=0, s=0.5)": (lambda g: stats.lognorm.pdf(g, 0.5), math.exp(2 * 0.5 ** 2)),
}
s2 = 2 * sigma ** 2                       # Var(theta_B - theta_A)
grid = np.linspace(-12, 12, 481)
res_t1 = []
for name, (pdf, m2) in lat.items():
    pAB = np.array([mix_pdf(d, pdf, +1, s2) for d in grid])
    pBA = pAB[::-1]                        # symmetry: density of -G+N at d equals density of G+N at -d
    dx = grid[1] - grid[0]
    pAB /= pAB.sum() * dx
    pBA /= pBA.sum() * dx
    kl = np.sum(pAB * np.log(np.maximum(pAB, 1e-300) / np.maximum(pBA, 1e-300))) * dx
    tv = 0.5 * np.sum(np.abs(pAB - pBA)) * dx
    for pi in (0.5, 0.7, 0.9):
        mixd = pi * pAB + (1 - pi) * pBA
        mi = (pi * np.sum(pAB * np.log(np.maximum(pAB, 1e-300) / np.maximum(mixd, 1e-300))) * dx
              + (1 - pi) * np.sum(pBA * np.log(np.maximum(pBA, 1e-300) / np.maximum(mixd, 1e-300))) * dx)
        bayes_err = np.sum(np.minimum(pi * pAB, (1 - pi) * pBA)) * dx
        res_t1.append({"latency": name, "pi": pi, "m2": m2, "KL": kl, "KL_bound": m2 / sigma ** 2, "MI": mi,
                       "MI_bound": 2 * pi * (1 - pi) * m2 / sigma ** 2, "TV": tv, "bayes_err": bayes_err,
                       "floor_TV": (1 - pi) * (1 - tv),
                       "floor_rho": (1 - pi) * max(0.0, 1 - math.sqrt(m2 / (2 * sigma ** 2)))})
    print("   %-24s KL %.3f <= %.3f   TV %.3f" % (name, kl, m2 / sigma ** 2, tv))
check("T1 KL(P_AB||P_BA) <= E[G^2]/sigma^2 (4 latency families)", all(r["KL"] <= r["KL_bound"] + 1e-6 for r in res_t1))
check("T1 I(H;D) <= 2 pi(1-pi) E[G^2]/sigma^2 (pi = .5, .7, .9)", all(r["MI"] <= r["MI_bound"] + 1e-6 for r in res_t1))
check("T2 Bayes error >= (1-pi)(1-TV) >= (1-pi)(1-rho/sqrt2)",
      all(r["bayes_err"] >= r["floor_TV"] - 1e-6 and r["floor_TV"] >= r["floor_rho"] - 1e-6 for r in res_t1))
out["t1"] = res_t1

# ---------------------------------------------------------------- T2b exact deterministic latency
viol = 0
for g in (0.25, 0.5, 1, 2, 4):
    for pi in (0.5, 0.7, 0.9):
        dstar = (sigma ** 2 / g) * math.log((1 - pi) / pi)
        s = math.sqrt(2) * sigma
        exact = pi * stats.norm.cdf((dstar - g) / s) + (1 - pi) * (1 - stats.norm.cdf((dstar + g) / s))
        N = 400000
        H = rng.random(N) < pi
        D = np.where(H, g, -g) + rng.normal(0, s, N)
        llr = g * D / sigma ** 2
        dec = llr > math.log((1 - pi) / pi)
        mc = np.mean(dec != H)
        if abs(mc - exact) > 0.004:
            viol += 1
        floor = (1 - pi) * max(0.0, 1 - g / (math.sqrt(2) * sigma))
        if exact < floor - 1e-9:
            viol += 1
check("T2b exact Bayes error (LLR = g D / sigma^2) matches Monte Carlo and respects the floor", viol == 0)

# ---------------------------------------------------------------- T3 fan-out
def pk(gs, sigma=1.0):
    f = lambda z: stats.norm.pdf(z) * np.prod([stats.norm.cdf(g / sigma - z) for g in gs])
    return integrate.quad(f, -10, 10, limit=200)[0]


viol = 0
rows = []
for gamma in (0.5, 1, 2, 3):
    for k in (2, 3, 5, 10, 30, 100):
        gs = [gamma] * (k - 1)
        exact = pk(gs)
        th = rng.normal(0, 1, (200000, k))
        tt = np.concatenate([[0.0], gs])
        mc = np.mean(np.argmin(tt + th, axis=1) == 0)
        union = max(0.0, 1 - (k - 1) * stats.norm.cdf(-gamma / math.sqrt(2)))
        if abs(exact - mc) > 0.005 or exact < union - 1e-9 or exact > stats.norm.cdf(gamma / math.sqrt(2)) + 1e-9:
            viol += 1
        rows.append({"gamma": gamma, "k": k, "P_k": exact, "union_lb": union})
    print("   gamma %.1f: " % gamma + "  ".join("k=%d %.3f" % (r["k"], r["P_k"]) for r in rows if r["gamma"] == gamma))
check("T3 fan-out formula = Monte Carlo, union lower bound and Phi(gamma/sqrt2) upper bound hold", viol == 0)
check("T3 P_k decreasing in k for every gamma",
      all(np.all(np.diff([r["P_k"] for r in rows if r["gamma"] == g]) < 0) for g in (0.5, 1, 2, 3)))
out["fanout"] = rows

# ---------------------------------------------------------------- T4 sync interval
viol = 0
for T in (60, 3600, 86400):
    s_s, s_r = 0.01, 20e-6
    N = 400000
    th = rng.normal(0, s_s, N) + rng.normal(0, s_r, N) * rng.uniform(0, T, N)
    pred = s_s ** 2 + s_r ** 2 * T ** 2 / 3
    if abs(th.var() / pred - 1) > 0.01:
        viol += 1
check("T4 Var(theta) = sigma_s^2 + sigma_rho^2 T^2/3 (Monte Carlo, 3 sync intervals)", viol == 0)


def t_max(g, eps, s_s, s_r):
    z = stats.norm.ppf(1 - eps)
    budget = g ** 2 / (2 * z ** 2) - s_s ** 2
    return math.sqrt(3 * budget) / s_r if budget > 0 else 0.0


table = []
for clock, s_r in (("XO 20 ppm", 20e-6), ("TCXO 2 ppm", 2e-6)):
    for sync, s_s in (("NTP (10 ms)", 0.010), ("PTP (10 us)", 1e-5)):
        for g in (0.01, 0.1, 1.0, 10.0):
            table.append({"clock": clock, "sync": sync, "latency_s": g, "Tmax_s": t_max(g, 0.05, s_s, s_r)})
out["sync_table"] = table
for r in table:
    if r["latency_s"] in (0.1, 1.0):
        print("   %-11s %-12s g=%5.2f s  T_max = %10.0f s" % (r["clock"], r["sync"], r["latency_s"], r["Tmax_s"]))

json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "theory_checks.json", "w"), indent=1)
print("ALL PASS" if ok else "SOME CHECKS FAILED")
