# -*- coding: utf-8 -*-
"""Checks of the revised ORDER results (after the formal review). Runs locally, CPU, about a minute.

T1  conditionally Gaussian offsets N | V ~ N(0, V), V independent of G and of the order:
    KL(P_AB || P_BA) <= kappa_eff^2,  kappa_eff^2 = 2 E[G^2] E[1/V]   (fixed V = 2 sigma^2: kappa_eff = kappa)
T2  sharp floor: Bayes error >= 2 (1 - pi) Phi(-kappa_eff / sqrt2), attained at pi = 1/2 for a deterministic
    latency and fixed V
R1  the drift model of Definition 1 (Gaussian sync residual and drift, U uniform on [0, T]) is a Gaussian scale
    mixture; the floor written with kappa = sqrt(E G^2) / sigma_theta can FAIL there, the kappa_eff floor holds
C1  synchronisation budget: per-event guarantee T <= sqrt(g^2/(2 z^2) - sigma_s^2) / sigma_rho (= T_max / sqrt3)
P1  fan-out: exact ratio g/sigma needed for P_k >= 0.95 versus the union-bound form
"""
import json
import math
import sys

import numpy as np
from scipy import integrate, optimize, stats

rng = np.random.default_rng(5)
ok = True
out = {}


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))


Phi = stats.norm.cdf


def densities(g_samp, v_samp, grid):
    """P_AB and P_BA densities on grid for D = +-G + N, N | V ~ N(0, V), Monte Carlo over (G, V)."""
    pab = np.zeros_like(grid)
    for g, v in zip(g_samp, v_samp):
        pab += stats.norm.pdf(grid, g, math.sqrt(v))
    pab /= len(g_samp)
    pba = pab[::-1]                       # symmetric grid
    dx = grid[1] - grid[0]
    return pab / (pab.sum() * dx), pba / (pba.sum() * dx), dx


def bayes(pab, pba, dx, pi):
    return float(np.sum(np.minimum(pi * pab, (1 - pi) * pba)) * dx)


def kl(p, q, dx):
    return float(np.sum(p * np.log(np.maximum(p, 1e-300) / np.maximum(q, 1e-300))) * dx)


# ---------------- T1/T2 with fixed V (Gaussian offsets), several latency laws
sigma = 1.0
grid = np.linspace(-14, 14, 2801)
laws = {"deterministic 1": lambda n: np.full(n, 1.0), "exponential mean 1": lambda n: rng.exponential(1.0, n),
        "gamma k=4": lambda n: rng.gamma(4, 0.25, n), "lognormal s=0.5": lambda n: rng.lognormal(0, 0.5, n),
        "uniform(0,2)": lambda n: rng.uniform(0, 2, n), "deterministic 0.4": lambda n: np.full(n, 0.4)}
viol, rows = 0, []
for name, draw in laws.items():
    g = draw(3000)
    v = np.full_like(g, 2 * sigma ** 2)
    pab, pba, dx = densities(g, v, grid)
    k_eff = math.sqrt(2 * np.mean(g ** 2) * np.mean(1 / v))
    d = kl(pab, pba, dx)
    for pi in (0.5, 0.7, 0.9):
        be = bayes(pab, pba, dx, pi)
        floor = 2 * (1 - pi) * Phi(-k_eff / math.sqrt(2))
        rows.append({"law": name, "pi": pi, "kappa": k_eff, "bayes": be, "floor": floor, "KL": d})
        if be < floor - 2e-3 or d > k_eff ** 2 + 1e-3:
            viol += 1
check("T1/T2 fixed-variance offsets: KL <= kappa^2 and Bayes error >= 2(1-pi)Phi(-kappa/sqrt2)", viol == 0, "violations %d" % viol)
det = [r for r in rows if r["law"] == "deterministic 1" and r["pi"] == 0.5][0]
check("T2 tight at pi = 1/2 for deterministic latency", abs(det["bayes"] - det["floor"]) < 3e-3,
      "bayes %.4f floor %.4f" % (det["bayes"], det["floor"]))
out["fixed"] = rows
print("   remove at most: kappa=1 -> %.0f%%, kappa=0.5 -> %.0f%%" % (100 * (1 - 2 * Phi(-1 / math.sqrt(2))), 100 * (1 - 2 * Phi(-0.5 / math.sqrt(2)))))

# ---------------- R1 drift model (scale mixture): kappa vs kappa_eff floors
viol_eff, fail_naive, rows = 0, 0, []
for s_s_frac in (0.1, 0.3):
    for kap_target in (0.1, 0.25, 0.5, 1.0):
        T, s_r = 1.0, 1.0
        s_s = s_s_frac * s_r * T
        sig_theta = math.sqrt(s_s ** 2 + s_r ** 2 * T ** 2 / 3)
        g0 = kap_target * sig_theta                     # deterministic latency
        n = 6000
        Ua, Ub = rng.uniform(0, T, n), rng.uniform(0, T, n)
        v = 2 * s_s ** 2 + s_r ** 2 * (Ua ** 2 + Ub ** 2)
        g = np.full(n, g0)
        gr = np.linspace(-8 * sig_theta, 8 * sig_theta, 3201)
        pab, pba, dx = densities(g, v, gr)
        be = bayes(pab, pba, dx, 0.5)
        k_eff = math.sqrt(2 * g0 ** 2 * np.mean(1 / v))
        naive = 2 * 0.5 * Phi(-kap_target / math.sqrt(2))
        eff = 2 * 0.5 * Phi(-k_eff / math.sqrt(2))
        rows.append({"sigma_s/(sigma_rho T)": s_s_frac, "kappa": kap_target, "kappa_eff": k_eff, "bayes": be,
                     "floor_kappa": naive, "floor_kappa_eff": eff})
        viol_eff += be < eff - 2e-3
        fail_naive += be < naive - 2e-3
        print("   drift model s_s=%.1f kappa %.2f kappa_eff %.2f  Bayes %.4f  floor(kappa) %.4f  floor(kappa_eff) %.4f"
              % (s_s_frac, kap_target, k_eff, be, naive, eff))
check("R1 kappa_eff floor holds under the drift model of Definition 1", viol_eff == 0)
check("R1 the naive kappa floor fails somewhere under the drift model (reviewer counterexample reproduced)", fail_naive > 0,
      "failures %d" % fail_naive)
out["drift"] = rows

# ---------------- C1 per-event synchronisation guarantee
viol = 0
for g, s_s, s_r, eps in ((1.0, 0.01, 11.547e-6, 0.05), (0.1, 0.01, 11.547e-6, 0.05), (1.0, 0.01, 1.1547e-6, 0.01)):
    z = stats.norm.ppf(1 - eps)
    t_avg = math.sqrt(3 * (g ** 2 / (2 * z ** 2) - s_s ** 2)) / s_r
    t_worst = math.sqrt(g ** 2 / (2 * z ** 2) - s_s ** 2) / s_r
    n = 400000
    # both events at the end of the interval, Gaussian drift
    th = rng.normal(0, s_s, (n, 2)) + rng.normal(0, s_r, (n, 2)) * t_worst
    err_worst = np.mean(g + th[:, 1] - th[:, 0] < 0)
    # events at uniform times within an interval of length t_avg
    U = rng.uniform(0, t_avg, (n, 2))
    th2 = rng.normal(0, s_s, (n, 2)) + rng.normal(0, s_r, (n, 2)) * U
    err_avg = np.mean(g + th2[:, 1] - th2[:, 0] < 0)
    th3 = rng.normal(0, s_s, (n, 2)) + rng.normal(0, s_r, (n, 2)) * t_avg
    err_end = np.mean(g + th3[:, 1] - th3[:, 0] < 0)
    print("   g %.2f eps %.2f: T_worst %.0f s err %.4f | T_avg %.0f s averaged err %.4f, end-of-interval err %.4f"
          % (g, eps, t_worst, err_worst, t_avg, err_avg, err_end))
    viol += err_worst > eps + 0.002
check("C1 per-event budget T_worst keeps every event pair at or below eps", viol == 0)

# ---------------- P1 fan-out requirements
def pk(gam, k):
    f = lambda z: stats.norm.pdf(z) * stats.norm.cdf(gam - z) ** (k - 1)
    return integrate.quad(f, -10, 10)[0]


req = {}
for k in (2, 10, 100):
    exact = optimize.brentq(lambda gm: pk(gm, k) - 0.95, 0.1, 10)
    union = math.sqrt(2) * stats.norm.ppf(1 - 0.05 / (k - 1))
    loose = 2 * math.sqrt(math.log((k - 1) / 0.05))
    req[k] = (exact, union, loose)
    print("   k=%3d  exact %.2f  union %.2f  sqrt-log %.2f" % (k, exact, union, loose))
check("P1 union form is valid (>= exact) and tighter than the sqrt-log form", all(e <= u + 1e-9 <= l + 1e-9 for e, u, l in req.values()))
print("   offset spread must shrink by %.2fx from 1 to 99 effects" % (req[100][0] / req[2][0]))
out["fanout_req"] = {str(k): v for k, v in req.items()}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "theory_checks_v2.json", "w"), indent=1)
print("ALL PASS" if ok else "SOME CHECKS FAILED")
