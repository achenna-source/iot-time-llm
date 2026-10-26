# -*- coding: utf-8 -*-
"""HAI propagation latencies, all releases (Kaggle CPU). For every controller demand tag (suffix D) with a
position feedback tag (suffix Z), detect demand steps and record the delay until the feedback has covered
half of the step. Writes /kaggle/working/hai_latency.json.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

res = {}
for fn in sorted(glob.glob("/kaggle/input/**/hai-*/*.csv", recursive=True)):
    if "summary" in fn:
        continue
    rel = fn.split("/")[-2] + "/" + os.path.basename(fn)
    try:
        h = pd.read_csv(fn, sep=None, engine="python", nrows=None)
    except Exception as e:
        print("skip", rel, e)
        continue
    cols = [c for c in h.columns if c.endswith("D") and (c[:-1] + "Z") in h.columns]
    att = [c for c in h.columns if c.lower().startswith("attack")]
    attack = h[att].max(axis=1).values if att else np.zeros(len(h))
    for dcol in cols:
        zcol = dcol[:-1] + "Z"
        dv = h[dcol].values.astype(float)
        zv = h[zcol].values.astype(float)
        span = np.nanpercentile(dv, 99) - np.nanpercentile(dv, 1)
        if not np.isfinite(span) or span <= 0:
            continue
        steps = np.where(np.abs(np.diff(dv)) > 0.02 * span)[0] + 1
        last = -10 ** 9
        lags = []
        for s in steps:
            if s - last < 30 or s + 300 >= len(zv) or s < 5:
                last = s
                continue
            last = s
            if attack[max(0, s - 5):s + 300].max() > 0:      # keep normal operation only
                continue
            step = dv[s] - dv[s - 1]
            base = zv[s - 1]
            moved = np.where(np.sign(step) * (zv[s:s + 300] - base) >= 0.5 * abs(step))[0]
            if len(moved):
                lags.append(int(moved[0]))
        if lags:
            res.setdefault(dcol + "->" + zcol, []).extend(lags)
    print(rel, "pairs so far", len(res), flush=True)
out = {k: v for k, v in res.items() if len(v) >= 5}
allv = np.concatenate([np.array(v) for v in out.values()]) if out else np.array([])
print("pairs", len(out), "events", len(allv), "quantiles", np.percentile(allv, [10, 25, 50, 75, 90]) if len(allv) else None)
json.dump(out, open("/kaggle/working/hai_latency.json", "w"))
