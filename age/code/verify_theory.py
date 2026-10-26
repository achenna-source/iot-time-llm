# -*- coding: utf-8 -*-
"""Numerical checks of every formal statement in the AGE paper (runs locally, CPU, seconds).

P1  averaged Bayes risk is non-decreasing in age (garbling), checked on random chains
P2  reversible chain + change-sensitive command: p_x(D) = 1 - P_xx(D) non-decreasing
P3  cyclic (non-reversible) devices: p_x(D) can decrease, so the read set is not an age threshold
T1  binary device: p = psi (1 - exp(-u/psi)), u = q D, threshold u* = -psi ln(1 - r/psi)
T2  any finite chain, change-sensitive command: p_x(D) <= 1 - exp(-q_x D) <= q_x D
T3  TTL regret on a heterogeneous fleet is strictly positive; volatility-scaled gate has zero regret
"""
import json
import math
import sys

import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(7)
out = {}
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))


def random_generator(n, reversible=False):
    if reversible:
        pi = rng.dirichlet(np.ones(n))
        S = rng.exponential(1.0, (n, n))
        S = (S + S.T) / 2
        Q = S * pi[None, :]          # detailed balance: pi_i Q_ij = pi_i pi_j S_ij symmetric
        np.fill_diagonal(Q, 0)
        np.fill_diagonal(Q, -Q.sum(1))
        return Q, pi
    Q = rng.exponential(1.0, (n, n))
    np.fill_diagonal(Q, 0)
    np.fill_diagonal(Q, -Q.sum(1))
    w, v = np.linalg.eig(Q.T)
    pi = np.real(v[:, np.argmin(np.abs(w))])
    return Q, pi / pi.sum()


ages = np.linspace(0, 6, 241)

# ---------------------------------------------------------------- P1 averaged monotonicity
viol = 0
for trial in range(200):
    n = rng.integers(2, 6)
    Q, pi = random_generator(n)
    na = rng.integers(2, 4)
    loss = rng.uniform(0, 1, (na, n))           # loss[a, state]
    prev = -1
    for d in ages:
        P = expm(Q * d)                          # P[x, y] = P(X_t = y | X_{t-d} = x)
        joint = pi[:, None] * P                  # P(old = x, now = y)
        risk = sum(min(loss[a] @ joint[x] for a in range(na)) for x in range(n))
        if risk < prev - 1e-10:
            viol += 1
        prev = risk
check("P1 averaged Bayes risk non-decreasing in age (200 random chains, random losses)", viol == 0, "violations=%d" % viol)

# ---------------------------------------------------------------- P2 reversible conditional monotonicity
viol = 0
for trial in range(300):
    n = rng.integers(2, 7)
    Q, pi = random_generator(n, reversible=True)
    pxx = np.array([np.diag(expm(Q * d)) for d in ages])   # ages x states
    if np.any(np.diff(pxx, axis=0) > 1e-10):
        viol += 1
check("P2 reversible + change-sensitive: 1-P_xx(D) non-decreasing (300 chains)", viol == 0, "violations=%d" % viol)

# ---------------------------------------------------------------- P3 cyclic counterexamples
lam = 1.0
Q3 = lam * np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]], float)
pxx_num = np.array([expm(Q3 * d)[0, 0] for d in ages])
pxx_cf = 1 / 3 + (2 / 3) * np.exp(-1.5 * lam * ages) * np.cos(math.sqrt(3) / 2 * lam * ages)
check("P3a three-phase cycle closed form P_xx = 1/3 + 2/3 e^{-3 lam D/2} cos(sqrt3 lam D/2)",
      np.max(np.abs(pxx_num - pxx_cf)) < 1e-9, "max err %.1e" % np.max(np.abs(pxx_num - pxx_cf)))
check("P3b three-phase cycle: p_x(D) decreases somewhere", np.any(np.diff(1 - pxx_num) < -1e-12))


def duty_cycle(k_on, k_off, t_on, t_off):
    """Pump on a timer: ON for ~t_on (k_on Erlang phases), OFF for ~t_off; observable state ON/OFF."""
    n = k_on + k_off
    Q = np.zeros((n, n))
    for i in range(n):
        rate = k_on / t_on if i < k_on else k_off / t_off
        Q[i, (i + 1) % n] = rate
        Q[i, i] = -rate
    return Q


Qd = duty_cycle(8, 16, 10.0, 20.0)                      # minutes: 10 min ON, 20 min OFF
tgrid = np.linspace(0, 90, 361)
# reading "ON" taken at a uniformly random phase of the ON block: initial law uniform on ON phases
init = np.zeros(24)
init[:8] = 1 / 8
p_change = np.array([1 - (init @ expm(Qd * d))[:8].sum() for d in tgrid])
dec = np.where(np.diff(p_change) < -1e-9)[0]
peak = tgrid[np.argmax(p_change[tgrid < 45])]
trough = tgrid[(tgrid > 15) & (tgrid < 60)][np.argmin(p_change[(tgrid > 15) & (tgrid < 60)])]
check("P3c duty-cycled pump (10 min ON / 20 min OFF): P(changed | read ON) non-monotone",
      len(dec) > 0, "max %.3f at %.1f min, min %.3f at %.1f min" % (p_change.max(), peak, p_change[tgrid == trough][0], trough))
out["dutycycle"] = {"t": tgrid.tolist(), "p": p_change.tolist()}
r = 0.6   # between the trough (one full cycle later) and the peak (mid OFF block)
read_set = tgrid[p_change > r]
gaps = np.where(np.diff(read_set) > (tgrid[1] - tgrid[0]) * 1.5)[0]
check("P3d duty cycle with r = 0.6: optimal read set is a union of >= 2 disjoint age intervals", len(gaps) >= 1,
      "intervals=%d" % (len(gaps) + 1))

# ---------------------------------------------------------------- T1 binary closed form and threshold
viol = 0
for trial in range(500):
    a, b = rng.exponential(1, 2)
    Q = np.array([[-a, a], [b, -b]])
    kap = a + b
    for x, qx in ((0, a), (1, b)):
        psi = qx / kap
        for d in rng.uniform(0, 5, 5):
            num = 1 - expm(Q * d)[x, x]
            cf = psi * (1 - math.exp(-qx * d / psi))
            if abs(num - cf) > 1e-9:
                viol += 1
check("T1a binary: p = psi(1 - exp(-u/psi)), u = q_x D, psi = 1 - pi_x (500 chains)", viol == 0)
viol = 0
for psi in np.linspace(0.05, 0.95, 19):
    for rr in np.linspace(0.001, 0.9, 50):
        if rr >= psi:
            continue
        ustar = -psi * math.log(1 - rr / psi)
        p = lambda u: psi * (1 - math.exp(-u / psi))
        if not (p(ustar * 0.999) <= rr + 1e-12 and p(ustar * 1.001) >= rr - 1e-12):
            viol += 1
        if not (ustar >= rr):     # exit-rate rule never reads too early
            viol += 1
check("T1b threshold u* = -psi ln(1 - r/psi) and u* >= r", viol == 0)

# ---------------------------------------------------------------- T2 universal exit-rate bound
viol = 0
worst = 0
for trial in range(300):
    n = rng.integers(2, 7)
    Q, pi = random_generator(n, reversible=bool(trial % 2))
    for d in ages[1:]:
        P = expm(Q * d)
        for x in range(n):
            qx = -Q[x, x]
            p = 1 - P[x, x]
            if p > 1 - math.exp(-qx * d) + 1e-10 or 1 - math.exp(-qx * d) > qx * d + 1e-12:
                viol += 1
            worst = max(worst, p / (qx * d))
check("T2 p_x(D) <= 1 - exp(-q_x D) <= q_x D on 300 random chains (both kinds)", viol == 0, "max p/(qD) = %.3f" % worst)

# ---------------------------------------------------------------- T3 TTL regret on heterogeneous fleets
def fleet_regret(qs, ws, r, psi=0.5, age_lo=60, age_hi=3 * 86400, n=4000):
    """Ages log-uniform on [age_lo, age_hi] seconds; rates in changes per day; symmetric binary devices."""
    a = np.exp(np.linspace(math.log(age_lo), math.log(age_hi), n))
    p = np.array([psi * (1 - np.exp(-(q / 86400) * a / psi)) for q in qs])   # classes x ages
    cost_opt = np.minimum(p, r)                                             # in units of L
    J_opt = sum(w * c.mean() for w, c in zip(ws, cost_opt))
    best = (None, 1e9)
    for tau in np.r_[0.0, np.exp(np.linspace(math.log(age_lo), math.log(age_hi), 600)), np.inf]:
        read = a > tau
        J = sum(w * np.where(read, r, pc).mean() for w, pc in zip(ws, p))
        if J < best[1]:
            best = (tau, J)
    J_always = r
    J_never = sum(w * pc.mean() for w, pc in zip(ws, p))
    return J_opt, best[1], best[0], J_always, J_never


res = []
for ratio in (1, 4, 16, 64, 256):
    qs = (1.0, 1.0 * ratio)
    Jo, Jt, tau, Ja, Jn = fleet_regret(qs, (0.5, 0.5), 0.1)
    res.append({"ratio": ratio, "J_opt": Jo, "J_ttl": Jt, "tau_s": tau, "J_always": Ja, "J_never": Jn,
                "regret_ttl": Jt - Jo, "rel": (Jt - Jo) / (min(Ja, Jn) - Jo)})
    print("   ratio %4d  J_opt %.4f  best TTL %.4f (tau %.0f s)  always %.4f  never %.4f  rel.regret %.3f"
          % (ratio, Jo, Jt, tau, Ja, Jn, (Jt - Jo) / (min(Ja, Jn) - Jo)))
check("T3a homogeneous fleet: best TTL regret ~ 0", res[0]["regret_ttl"] < 1e-4)
check("T3b heterogeneous fleets: best TTL regret > 0 and increasing with rate ratio",
      all(x["regret_ttl"] > 1e-4 for x in res[1:]) and all(np.diff([x["regret_ttl"] for x in res]) > 0))
out["ttl_regret"] = res

json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "theory_checks.json", "w"), indent=1)
print("ALL PASS" if ok else "SOME CHECKS FAILED")
