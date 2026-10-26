# -*- coding: utf-8 -*-
"""AGE real-trace re-analysis after the simulated review (Kaggle CPU; CASAS and UCI attached).

Changes from realdata.py:
- temporal split: every parameter and every TTL is fitted on the first half of each device record and evaluated
  on the second half;
- requests weighted by the time share of the remembered state (a request finds the device in state x with
  probability pi_x), with the uniform weighting kept as a sensitivity check;
- baselines: single TTL, TTL per device class, TTL per (device, state), exit-rate rule, renewal rule
  (D / m_x > r with m_x the mean measured dwell), two-parameter gate;
- the empirical optimum may also act on the opposite state when that is cheapest (cost 1 - p);
- all three prices, absolute costs, and per-class results.
Writes /kaggle/working/realdata_age_v2.json
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd

AGES = np.exp(np.linspace(math.log(60), math.log(3 * 86400), 61))
PRICES = (0.02, 0.1, 0.3)
rng = np.random.default_rng(5)


def find(p):
    return sorted(glob.glob("/kaggle/input/**/" + p, recursive=True))


def series(t, s):
    t, s = np.asarray(t, float), np.asarray(s, int)
    keep = np.r_[True, s[1:] != s[:-1]]
    return t[keep], s[keep]


def at(te, se, tq):
    i = np.searchsorted(te, tq, side="right") - 1
    return se[np.clip(i, 0, len(se) - 1)]


def fit(te, se, t0, t1):
    """exit rates (per second), time shares and mean dwells from events in [t0, t1]."""
    m = (te >= t0) & (te <= t1)
    te, se = te[m], se[m]
    if len(te) < 6:
        return None
    dur = np.diff(np.r_[te, t1])
    out = {}
    for x in (0, 1):
        tot = dur[se == x].sum()
        ex = np.sum(se[:-1] == x)
        out[x] = {"q": ex / tot if tot > 0 else np.nan, "pi": tot / dur.sum(),
                  "m": tot / max(np.sum(se == x), 1)}
    return out


def curve(te, se, t0, t1, x, n=3000):
    tq = rng.uniform(t0, t1 - 3 * 86400, n * 4)
    sq = at(te, se, tq)
    tq = tq[sq == x][:n]
    if len(tq) < 150:
        return None
    return np.array([np.mean(at(te, se, tq + a) != x) for a in AGES])


devices = []
for fn in find("casas-smart-home/*.txt"):
    home = os.path.basename(fn)[:-4]
    rows = []
    with open(fn, errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and p[2][0] in "DM" and p[3] in ("ON", "OFF", "OPEN", "CLOSE"):
                rows.append((p[0] + " " + p[1], p[2], 1 if p[3] in ("ON", "OPEN") else 0))
    df = pd.DataFrame(rows, columns=["ts", "sid", "v"])
    df["t"] = pd.to_datetime(df["ts"], errors="coerce", format="mixed").astype("int64") / 1e9
    df = df.dropna().sort_values("t")
    for sid, g in df.groupby("sid"):
        if len(g) >= 50:
            te, se = series(g["t"].values, g["v"].values)
            devices.append(("casas/%s/%s" % (home, sid), "door" if sid.startswith("D") else "motion", te, se))
for fn in find("occupancy-detection-data-set-uci/*.txt"):
    d = pd.read_csv(fn)
    te, se = series(pd.to_datetime(d["date"]).astype("int64").values / 1e9, d["Occupancy"].astype(int).values)
    devices.append(("uci/" + os.path.basename(fn), "occupancy", te, se))

cells = []          # one row per (device, state, age) on the TEST half
for name, kind, te, se in devices:
    t0, t1 = te[0], te[-1]
    if t1 - t0 < 8 * 86400:
        continue
    mid = t0 + (t1 - t0) / 2
    par = fit(te, se, t0, mid)
    if par is None:
        continue
    for x in (0, 1):
        tr = curve(te, se, t0, mid, x)
        ts = curve(te, se, mid, t1, x)
        if tr is None or ts is None or not np.isfinite(par[x]["q"]) or par[x]["q"] <= 0:
            continue
        kap = par[0]["q"] + par[1]["q"]
        psi = par[x]["q"] / kap
        pm = psi * (1 - np.exp(-kap * AGES))
        for j, a in enumerate(AGES):
            cells.append({"dev": name, "kind": kind, "x": x, "a": a, "w": par[x]["pi"], "p_train": tr[j], "p_test": ts[j],
                          "p_model": pm[j], "q": par[x]["q"], "m": par[x]["m"]})
C = pd.DataFrame(cells)
print("devices used", C["dev"].nunique(), "cells", len(C), flush=True)


def evaluate(D, r, weight):
    w = D["w"].values if weight == "share" else np.ones(len(D))
    w = w / w.sum()
    pt = D["p_test"].values
    a = D["a"].values
    res = {}
    res["oracle"] = float(np.sum(w * np.minimum(np.minimum(pt, r), 1 - pt)))
    res["oracle_read_or_act"] = float(np.sum(w * np.minimum(pt, r)))
    res["always_read"] = float(r)
    res["never_read"] = float(np.sum(w * pt))
    cost = lambda read: float(np.sum(w * np.where(read, r, pt)))
    taus = np.r_[0, AGES, 1e12]
    # single TTL fitted on the train half
    tr = D["p_train"].values
    ctr = lambda read: float(np.sum(w * np.where(read, r, tr)))
    best = min(taus, key=lambda tau: ctr(a > tau))
    res["ttl_single"] = cost(a > best)
    # TTL per class and per (device, state), fitted on train
    for key, col in (("ttl_class", ["kind"]), ("ttl_state", ["dev", "x"])):
        read = np.zeros(len(D), bool)
        for _, g in D.groupby(col):
            idx = g.index.values
            gi = D.index.get_indexer(idx)
            bt = min(taus, key=lambda tau: float(np.sum(w[gi] * np.where(a[gi] > tau, r, tr[gi]))))
            read[gi] = a[gi] > bt
        res[key] = cost(read)
    res["exit_rate_rule"] = cost(D["q"].values * a > -math.log(1 - r))
    res["renewal_rule"] = cost(a / D["m"].values > r)
    res["two_param_gate"] = cost(D["p_model"].values > r)
    fx = min(res["always_read"], res["never_read"])
    res["rho"] = {k: (v - res["oracle_read_or_act"]) / (fx - res["oracle_read_or_act"]) for k, v in res.items()
                  if k not in ("oracle", "oracle_read_or_act")}
    return res


out = {"n_devices": int(C["dev"].nunique()), "by": {}}
for kind in ("door", "motion", "occupancy", "all"):
    D = C if kind == "all" else C[C["kind"] == kind]
    D = D.reset_index(drop=True)
    if len(D) == 0:
        continue
    out["by"][kind] = {"n_devices": int(D["dev"].nunique())}
    for weight in ("share", "uniform"):
        for r in PRICES:
            out["by"][kind]["%s_r%s" % (weight, r)] = evaluate(D, r, weight)
            e = out["by"][kind]["%s_r%s" % (weight, r)]
            print(kind, weight, r, {k: round(v, 3) for k, v in e["rho"].items()}, flush=True)
json.dump(out, open("/kaggle/working/realdata_age_v2.json", "w"), indent=1)
print("done")
