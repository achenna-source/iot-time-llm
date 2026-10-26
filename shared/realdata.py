# -*- coding: utf-8 -*-
"""Real-trace anchors for both papers (Kaggle CPU; datasets attached as kernel sources).

AGE   binary device states from CASAS (door contacts, motion), UCI Occupancy and HAI (binary actuator tags):
      exit rates, empirical staleness curves P(changed | read state x, age D), deviation from the CTMC fit,
      and the cost of TTL rules versus the volatility-scaled gate evaluated on the empirical curves.
ORDER propagation latencies in the HAI testbed (controller demand -> actuator position -> process variable)
      and in CASAS homes (door contact -> first motion event), compared with the clock-offset spread of
      common IoT synchronisation regimes.
Writes /kaggle/working/realdata_age.json and realdata_order.json.
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd

OUT = "/kaggle/working"
AGES = np.exp(np.linspace(math.log(60), math.log(3 * 86400), 61))   # 1 min .. 3 days
rng = np.random.default_rng(3)


def find(pattern):
    hits = sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True))
    return hits


# ============================================================== binary state series
def series_from_events(times, states):
    """piecewise-constant binary series: times (seconds, sorted), states (0/1) after each event."""
    t = np.asarray(times, float)
    s = np.asarray(states, int)
    keep = np.r_[True, s[1:] != s[:-1]]
    return t[keep], s[keep]


def state_at(t_ev, s_ev, tq):
    i = np.searchsorted(t_ev, tq, side="right") - 1
    return s_ev[np.clip(i, 0, len(s_ev) - 1)]


def analyse_device(name, t_ev, s_ev, n_samples=20000):
    t0, t1 = t_ev[0], t_ev[-1]
    span = t1 - t0
    if span < 3 * 86400 or len(t_ev) < 10:
        return None
    dur = np.diff(np.r_[t_ev, t1])
    res = {"name": name, "span_days": span / 86400, "n_changes": int(len(t_ev) - 1)}
    rates = {}
    for x in (0, 1):
        tot = dur[s_ev == x].sum()
        exits = np.sum(s_ev[:-1] == x)
        rates[x] = exits / tot * 86400 if tot > 0 else float("nan")          # per day
        res["pi_%d" % x] = float(tot / dur.sum())
    res["q_day"] = {str(k): float(v) for k, v in rates.items()}
    res["changes_per_day"] = (len(t_ev) - 1) / (span / 86400)
    # empirical staleness curves, read times uniform over the record (excluding the last 3 days)
    tq = rng.uniform(t0, t1 - 3 * 86400, n_samples)
    sq = state_at(t_ev, s_ev, tq)
    curves = {}
    for x in (0, 1):
        m = sq == x
        if m.sum() < 200:
            continue
        emp = [float(np.mean(state_at(t_ev, s_ev, tq[m] + a) != x)) for a in AGES]
        kap = rates[0] + rates[1]
        model = [float(rates[x] / kap * (1 - math.exp(-kap * a / 86400))) for a in AGES]
        curves[str(x)] = {"emp": emp, "ctmc": model, "n": int(m.sum())}
    res["curves"] = curves
    return res


devices = []
# ---- CASAS: lines "date time sensor value [activity]"
for fn in find("casas-smart-home/*.txt"):
    home = os.path.basename(fn)[:-4]
    rows = []
    with open(fn, errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            sid, val = p[2], p[3]
            if sid[0] in "DM" and val in ("ON", "OFF", "OPEN", "CLOSE"):
                rows.append((p[0] + " " + p[1], sid, 1 if val in ("ON", "OPEN") else 0))
    if not rows:
        continue
    df = pd.DataFrame(rows, columns=["ts", "sid", "v"])
    df["t"] = pd.to_datetime(df["ts"], errors="coerce", format="mixed").astype("int64") / 1e9
    df = df.dropna().sort_values("t")
    for sid, g in df.groupby("sid"):
        if len(g) < 50:
            continue
        te, se = series_from_events(g["t"].values, g["v"].values)
        r = analyse_device("casas/%s/%s" % (home, sid), te, se, n_samples=6000)
        if r:
            r["kind"] = "door" if sid.startswith("D") else "motion"
            devices.append(r)
    print(home, "devices so far", len(devices), flush=True)

# ---- UCI occupancy (1-minute samples)
for fn in find("occupancy-detection-data-set-uci/*.txt"):
    d = pd.read_csv(fn)
    t = pd.to_datetime(d["date"]).astype("int64").values / 1e9
    te, se = series_from_events(t, d["Occupancy"].astype(int).values)
    r = analyse_device("uci/" + os.path.basename(fn), te, se, n_samples=6000)
    if r:
        r["kind"] = "occupancy"
        devices.append(r)

# ---- HAI: binary tags (values in {0,1}) in the 21.03 training files
hai_files = find("hai-21.03/train*.csv")
if hai_files:
    parts = [pd.read_csv(f, sep=None, engine="python") for f in hai_files[:2]]
    h = pd.concat(parts, ignore_index=True)
    tcol = [c for c in h.columns if c.lower() in ("time", "timestamp")][0]
    t = pd.to_datetime(h[tcol]).astype("int64").values / 1e9
    for c in h.columns:
        if c == tcol or c.lower().startswith("attack"):
            continue
        v = h[c].values
        u = np.unique(v[: 200000])
        if len(u) == 2 and set(np.round(u, 6)) <= {0.0, 1.0}:
            te, se = series_from_events(t, v.astype(int))
            if len(te) >= 8:
                r = analyse_device("hai/" + c, te, se, n_samples=6000)
                if r:
                    r["kind"] = "ics-binary"
                    devices.append(r)
print("devices", len(devices), flush=True)

# ============================================================== policy costs on the empirical curves
def fleet_eval(devs, r):
    """cost per request (units of L) of each policy, requests uniform over (device, read state, age)."""
    cells = []
    for d in devs:
        for x, c in d["curves"].items():
            q = d["q_day"][x]
            if not np.isfinite(q) or q <= 0:
                continue
            for a, pe, pm in zip(AGES, c["emp"], c["ctmc"]):
                cells.append((q, a, pe, pm))
    q, a, pe, pm = (np.array(z) for z in zip(*cells))
    out = {"n_cells": len(cells)}
    out["always_read"] = float(r)
    out["never_read"] = float(pe.mean())
    out["oracle_curve"] = float(np.minimum(pe, r).mean())                    # knows the empirical curve
    out["ctmc_gate"] = float(np.where(pm > r, r, pe).mean())                  # fitted-model optimum
    out["exit_rate_gate"] = float(np.where((q / 86400) * a > -math.log(1 - r), r, pe).mean())
    best = min((float(np.where(a > tau, r, pe).mean()), float(tau)) for tau in np.r_[0, AGES, 1e9])
    out["best_ttl"], out["best_ttl_tau_s"] = best
    return out


fleets = {}
for kind in ("door", "motion", "occupancy", "ics-binary", "all"):
    ds = [d for d in devices if kind == "all" or d["kind"] == kind]
    if not ds:
        continue
    fleets[kind] = {"n_devices": len(ds), "r": {str(r): fleet_eval(ds, r) for r in (0.02, 0.1, 0.3)}}
    print(kind, len(ds), json.dumps(fleets[kind]["r"]["0.1"]), flush=True)
json.dump({"ages_s": AGES.tolist(), "devices": devices, "fleets": fleets},
          open(os.path.join(OUT, "realdata_age.json"), "w"))

# ============================================================== ORDER: propagation latencies
lat = {}
if hai_files:
    h = h.set_index(pd.to_datetime(h[tcol]))
    pairs = []
    for c in h.columns:
        if c.endswith("D") and c[:-1] + "Z" in h.columns:
            pairs.append((c, c[:-1] + "Z"))
    res = []
    for dcol, zcol in pairs:
        dv = h[dcol].values.astype(float)
        zv = h[zcol].values.astype(float)
        span = np.nanpercentile(dv, 99) - np.nanpercentile(dv, 1)
        if not np.isfinite(span) or span <= 0:
            continue
        thr = 0.05 * span
        steps = np.where(np.abs(np.diff(dv)) > thr)[0] + 1
        lags = []
        last = -10 ** 9
        for s in steps:
            if s - last < 60 or s + 120 >= len(zv):
                last = s
                continue
            last = s
            base = zv[s - 1]
            direction = np.sign(dv[s] - dv[s - 1])
            moved = np.where(direction * (zv[s:s + 120] - base) > 0.5 * abs(dv[s] - dv[s - 1]))[0]
            if len(moved):
                lags.append(int(moved[0]))
        if len(lags) >= 5:
            res.append({"pair": "%s->%s" % (dcol, zcol), "n": len(lags), "lags_s": lags})
    lat["hai_demand_to_position"] = res
    all_l = np.concatenate([r["lags_s"] for r in res]) if res else np.array([])
    print("HAI pairs", len(res), "lags", len(all_l), "median", np.median(all_l) if len(all_l) else None, flush=True)

# CASAS: door contact opening -> first motion ON elsewhere within 120 s (household causal chains)
for fn in find("casas-smart-home/aruba.txt"):
    ev = []
    with open(fn, errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and ((p[2].startswith("D") and p[3] == "OPEN") or (p[2].startswith("M") and p[3] == "ON")):
                ev.append((p[0] + " " + p[1], p[2]))
    e = pd.DataFrame(ev, columns=["ts", "sid"])
    e["t"] = pd.to_datetime(e["ts"], errors="coerce", format="mixed").astype("int64") / 1e9
    e = e.dropna().sort_values("t")
    tt, ss = e["t"].values, e["sid"].values
    gl = []
    mot = np.where(np.char.startswith(ss.astype(str), "M"))[0]
    for i in np.where(np.char.startswith(ss.astype(str), "D"))[0]:
        j = np.searchsorted(tt[mot], tt[i], side="right")
        if j < len(mot) and tt[mot[j]] - tt[i] <= 120:
            gl.append(float(tt[mot[j]] - tt[i]))
    lat["casas_door_to_motion_s"] = gl
    print("CASAS door->motion", len(gl), "median", np.median(gl) if gl else None, flush=True)
json.dump(lat, open(os.path.join(OUT, "realdata_order.json"), "w"))
print("done")
