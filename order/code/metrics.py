# -*- coding: utf-8 -*-
"""ORDER metrics shared by the reference-bound simulation and the model analysis.

Main part (forced choice between event A and event B):
  ts direction   = A if D > 0 else B           (earliest timestamp)
  mech direction = the event the maintenance history favours
  S_ts   = items where the timestamps point to the true first event and the history does not
  S_mech = items where the history points to the true first event and the timestamps do not
  arbitration index  A_ord = acc(S_ts) + acc(S_mech) - 1   (0 for any fixed-cue rule)
Two estimators: realised labels (truth drawn by the generator) and Rao-Blackwellised soft labels
(posterior P(A first | prompt), which only uses quantities declared in the prompt).
"""
import numpy as np


def directions(rows):
    ts_A = np.array([r["D"] > 0 for r in rows])
    mech_A = np.array([r["fav_fwd"] for r in rows])
    return ts_A, mech_A


def arbitration(P_A, rows, soft=True):
    """P_A: probability of answering event A per item."""
    P_A = np.asarray(P_A, float)
    ts_A, mech_A = directions(rows)
    if soft:
        tA = np.array([r["post_A"] for r in rows])
    else:
        tA = np.array([1.0 if r["a_first"] else 0.0 for r in rows])
    disagree = ts_A != mech_A
    acc = P_A * tA + (1 - P_A) * (1 - tA)                  # expected accuracy given truth probability
    # weight of each item in S_ts: P(truth = ts direction) on disagreeing items
    w_ts = np.where(disagree, np.where(ts_A, tA, 1 - tA), 0.0)
    w_me = np.where(disagree, np.where(mech_A, tA, 1 - tA), 0.0)
    acc_ts = np.sum(w_ts * np.where(ts_A, P_A, 1 - P_A)) / np.sum(w_ts)
    acc_me = np.sum(w_me * np.where(mech_A, P_A, 1 - P_A)) / np.sum(w_me)
    return {"A": acc_ts + acc_me - 1, "acc_S_ts": acc_ts, "acc_S_mech": acc_me, "acc": float(np.mean(acc)),
            "n_S_ts": float(np.sum(w_ts)), "n_S_mech": float(np.sum(w_me))}


def position_reliance(P_A, rows):
    """P(choose the event whose line comes first) - 1/2."""
    P_A = np.asarray(P_A, float)
    la = np.array([r["list_a_first"] for r in rows])
    return float(np.mean(np.where(la, P_A, 1 - P_A)) - 0.5)
