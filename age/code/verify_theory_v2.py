# -*- coding: utf-8 -*-
"""Check of the renewal extension of Proposition 4 (after the simulated review), local, CPU.

For a stationary alternating-renewal device read at a time independent of its history, the staleness of a
reading x satisfies p_x(D) <= F_e,x(D) <= D / m_x, where F_e,x is the equilibrium residual-life distribution of
the sojourns in x and m_x their mean. Checked on the timer-driven pump of Example 1 (Erlang dwells) and on
alternating renewal devices with Weibull and lognormal dwells, by simulation.
"""
import math
import sys

import numpy as np
from scipy import stats
from scipy.linalg import expm

rng = np.random.default_rng(23)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))


# Example 1: Erlang(8) on-dwell with mean 10 min, read at a stationary time given "on"
k_on, k_off, t_on, t_off = 8, 16, 10.0, 20.0
n = k_on + k_off
Q = np.zeros((n, n))
for i in range(n):
    rate = k_on / t_on if i < k_on else k_off / t_off
    Q[i, (i + 1) % n] = rate
    Q[i, i] = -rate
init = np.zeros(n)
init[:k_on] = 1 / k_on
ts = np.linspace(0.25, 120, 480)
p = np.array([1 - (init @ expm(Q * d))[:k_on].sum() for d in ts])
Fe = np.array([stats.gamma.expect(lambda s: 1.0, args=(k_on,), scale=t_on / k_on, lb=0, ub=0) for _ in [0]])  # placeholder
surv = lambda s: stats.gamma.sf(s, k_on, scale=t_on / k_on)
from scipy import integrate  # noqa: E402
Fe = np.array([integrate.quad(surv, 0, d)[0] / t_on for d in ts])
check("Example 1: p_on(D) <= F_e(D) <= D/m on 0.25..120 min", np.all(p <= Fe + 1e-9) and np.all(Fe <= ts / t_on + 1e-12),
      "max p - Fe = %.2e" % np.max(p - Fe))


def simulate(draw_on, draw_off, m_on, horizon=2e5, n_reads=60000, ages=(0.2, 0.5, 1, 2, 4)):
    t, state, times, states = 0.0, 1, [0.0], [1]
    while t < horizon:
        t += draw_on() if state == 1 else draw_off()
        state = 1 - state
        times.append(t)
        states.append(state)
    times, states = np.array(times), np.array(states)
    tq = rng.uniform(0.1 * horizon, 0.8 * horizon, n_reads)
    idx = np.searchsorted(times, tq, side="right") - 1
    on = states[idx] == 1
    out = []
    for a in ages:
        idx2 = np.searchsorted(times, tq[on] + a, side="right") - 1
        out.append(np.mean(states[idx2] != 1))
    return np.array(out), np.array(ages)


viol = 0
for name, don, doff, m_on, sf in (
        ("Weibull k=0.6", lambda: rng.weibull(0.6) * 1.0, lambda: rng.weibull(0.6) * 2.0, math.gamma(1 + 1 / 0.6),
         lambda s: math.exp(-(s / 1.0) ** 0.6)),
        ("lognormal s=1", lambda: rng.lognormal(0, 1), lambda: rng.lognormal(0.5, 1), math.exp(0.5),
         lambda s: stats.lognorm.sf(s, 1)),
        ("Weibull k=3", lambda: rng.weibull(3) * 1.0, lambda: rng.weibull(3) * 1.5, math.gamma(1 + 1 / 3),
         lambda s: math.exp(-(s / 1.0) ** 3))):
    pe, ages = simulate(don, doff, m_on)
    fe = np.array([integrate.quad(sf, 0, a)[0] / m_on for a in ages])
    bad = np.sum(pe > fe + 0.01) + np.sum(fe > ages / m_on + 1e-9)
    viol += bad
    print("   %-14s p %s  F_e %s  D/m %s" % (name, np.round(pe, 3), np.round(fe, 3), np.round(ages / m_on, 3)))
check("renewal devices: p_x(D) <= F_e,x(D) <= D/m_x (simulation, 3 dwell laws)", viol == 0)
print("ALL PASS" if ok else "SOME CHECKS FAILED")
