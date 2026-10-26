# -*- coding: utf-8 -*-
"""AGE metrics shared by the reference-bound simulation and the model analysis.

For a probe i with staleness probability p_i and read cost ratio r_i (both in units of L):
  expected cost of READ = r_i, of acting on memory = p_i, of the opposite act = 1 - p_i.
Staleness discrimination index (Youden J of the read decision against the realised change):
  J = sum_i w_i P_i p_i / sum_i w_i p_i  -  sum_i w_i P_i (1 - p_i) / sum_i w_i (1 - p_i)
which is exactly 0 for any policy whose read probability does not vary across probes.
Normalised regret: rho = (C - C_opt) / (C_best_fixed - C_opt); 0 = optimal, 1 = no better than the
best fixed policy (always read, or always act on memory).
"""
import numpy as np


def youden(P_read, p, w=None):
    P_read, p = np.asarray(P_read, float), np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    return (np.sum(w * P_read * p) / np.sum(w * p)) - (np.sum(w * P_read * (1 - p)) / np.sum(w * (1 - p)))


def exp_cost(P_read, P_mem, P_opp, p, r):
    """expected cost per probe (units of L) for a stochastic policy."""
    return np.asarray(P_read) * r + np.asarray(P_mem) * p + np.asarray(P_opp) * (1 - p)


def regret_summary(P_read, P_mem, P_opp, p, r):
    p, r = np.asarray(p, float), np.asarray(r, float)
    C = exp_cost(P_read, P_mem, P_opp, p, r).mean()
    C_opt = np.minimum(p, r).mean()
    C_fix = min(r.mean(), p.mean())
    return {"cost": C, "cost_opt": C_opt, "cost_best_fixed": C_fix, "rho": (C - C_opt) / (C_fix - C_opt),
            "J": youden(P_read, p), "read_rate": float(np.mean(P_read))}
