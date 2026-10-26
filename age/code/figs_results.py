# -*- coding: utf-8 -*-
"""AGE result figures from analysis_age.json: re-read curves per model (small multiples) and the
age/volatility/price weight ratios.
python figs_results.py <results_dir> <figures_dir>
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
from figstyle import INK2, MODEL_LABEL, MODELS, MUTED, SERIES, panel_label, plt, setup  # noqa: E402

RES, FIG = sys.argv[1:3]
setup()
A = json.load(open(os.path.join(RES, "analysis_age.json")))
models = [m for m in MODELS if m in A] or [m for m in A if not m.startswith("_")]
prices = ["0.02", "0.1", "0.3", "0.6"]

# ------------------------------------------------------------ re-read curves
n = len(models) + 1
cols = 3
rowsn = math.ceil(n / cols)
fig, axs = plt.subplots(rowsn, cols, figsize=(7.2, 2.05 * rowsn), sharex=True, sharey=True)
axs = np.array(axs).reshape(-1)
uu = np.exp(np.linspace(math.log(3e-4), math.log(150), 300))
for k, r in enumerate(prices):
    rv = float(r)
    opt = (0.5 * (1 - np.exp(-2 * uu)) > rv).astype(float)
    axs[0].plot(uu, opt, color=SERIES[k], label="r = %s" % r)
axs[0].set_title("Optimal policy", loc="left")
axs[0].legend(loc="center left", fontsize=6.2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
for i, m in enumerate(models, start=1):
    ax = axs[i]
    for k, r in enumerate(prices):
        c = A[m]["curve"].get(r)
        if not c:
            continue
        ax.plot([x[0] for x in c], [x[1] for x in c], color=SERIES[k], marker="o", ms=2.2, lw=1.1)
        rv = float(r)
        if rv < 0.5:
            ax.axvline(-0.5 * math.log(1 - 2 * rv), color=SERIES[k], lw=0.6, ls=":")
    ax.set_title(MODEL_LABEL.get(m, m), loc="left")
for j in range(n, len(axs)):
    axs[j].axis("off")
for ax in axs[:n]:
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.04)
for ax in axs[::cols]:
    ax.set_ylabel("re-read rate")
for ax in axs[n - cols:n]:
    ax.set_xlabel("normalised age u = q Δ")
fig.tight_layout(h_pad=0.6, w_pad=0.6)
fig.savefig(os.path.join(FIG, "fig_reread.png"))

# ------------------------------------------------------------ weight ratios
fig, ax = plt.subplots(figsize=(4.8, 2.4))
ys = np.arange(len(models))[::-1]
for k, (key, lab) in enumerate((("ratio_q_age", "volatility weight / age weight"), ("ratio_r_age", "price weight / age weight"))):
    vals = [A[m]["sym"].get(key) for m in models]
    xs = [v if v is not None else np.nan for v in vals]
    ax.scatter(xs, ys + (0.12 if k == 0 else -0.12), color=SERIES[k], marker="os"[k], s=22, label=lab, zorder=3)
ref = A.get("_reference", {}).get("sym", {})
for k, key in enumerate(("ratio_q_age", "ratio_r_age")):
    if ref.get(key) is not None:
        ax.axvline(ref[key], color=SERIES[k], lw=0.9, ls="--")
ax.axvline(0, color=MUTED, lw=0.7)
ax.set_yticks(ys)
ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in models])
ax.set_xlabel("ratio of fitted logistic weights (dashed: optimal policy)")
ax.legend(loc="lower right", fontsize=6.4, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_weights.png"))
print("saved")
