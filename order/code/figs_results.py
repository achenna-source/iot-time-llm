# -*- coding: utf-8 -*-
"""ORDER result figures from analysis_order_v2.json (prefilled scoring for every model): timestamp use on the
items where the cues disagree, cue weights, and certification.
python figs_results.py <results_dir> <figures_dir>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
from figstyle import INK2, MARKERS, MODEL_LABEL, MODELS, MUTED, SERIES, panel_label, plt, setup  # noqa: E402

RES, FIG = sys.argv[1:3]
setup()
A = json.load(open(os.path.join(RES, "analysis_order_v2.json")))
models = [m for m in MODELS if m in A] or [m for m in A if not m.startswith("_")]
lab = lambda m: MODEL_LABEL.get(m, m)

# ------------------------------------------------------------ cues
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axs[0]
gammas = ["0.5", "1.0", "2.0"]
bay = [A[models[0]]["main"]["follow_ts_dis_by_gamma_bayes"][g] for g in gammas]
ax.plot([0.5, 1, 2], bay, color=INK2, ls="--", marker="o", ms=3, label="Bayes attributor")
for k, m in enumerate(models):
    ax.plot([0.5, 1, 2], [A[m]["main"]["follow_ts_dis_by_gamma"][g] for g in gammas], color=SERIES[k], marker=MARKERS[k], label=lab(m))
ax.set_xscale("log", base=2)
ax.set_xticks([0.5, 1, 2])
ax.set_xticklabels(["0.5", "1", "2"])
ax.set_ylim(0, 1.0)
ax.set_xlabel("latency / clock-offset SD (γ)")
ax.set_ylabel("P(follows the earlier timestamp)")
ax.set_title("Timestamps against the history", loc="left")
ax.legend(loc="upper left", fontsize=6.0, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
panel_label(ax, "a")

ax = axs[1]
keys = [("history", "history" + chr(10) + "log-odds"), ("sign_D", "sign of D"), ("size_llr", "size of the" + chr(10) + "evidence"),
        ("position", "line" + chr(10) + "position")]
w = 0.8 / len(models)
for k, m in enumerate(models):
    cw = A[m]["main"]["cues"]
    vals = [cw[kk][0] for kk, _ in keys]
    errs = [cw[kk][1] for kk, _ in keys]
    xs = np.arange(len(keys)) + (k - (len(models) - 1) / 2) * w
    ax.bar(xs, vals, width=w * 0.9, color=SERIES[k], yerr=errs, error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": INK2}, label=lab(m))
for j, ref in enumerate((1, 0, 1, 0)):
    ax.plot([j - 0.45, j + 0.45], [ref, ref], color=INK2, ls="--", lw=0.9)
ax.axhline(0, color=MUTED, lw=0.7)
ax.set_xticks(np.arange(len(keys)))
ax.set_xticklabels([k[1] for k in keys])
ax.set_ylabel("fitted weight (Bayes: dashed)")
ax.set_title("Cue weights in the model's choice", loc="left")
panel_label(ax, "b")
fig.tight_layout(w_pad=1.6)
fig.savefig(os.path.join(FIG, "fig_cues.png"))

# ------------------------------------------------------------ certification
fig, ax = plt.subplots(figsize=(5.6, 2.5))
ratios = ["0.5", "1.0", "1.5", "2.5", "4.0"]
ax.step([0.25, 2, 2, 4.5], [1, 1, 0, 0], where="post", color=INK2, ls="--", lw=0.9, label="order certifier (Lemma 1)")
for k, m in enumerate(models):
    c = A[m].get("cert_px")
    if not c:
        continue
    ax.plot([float(x) for x in ratios], [c["by_ratio"][x]["dec"] for x in ratios], color=SERIES[k], marker=MARKERS[k], label=lab(m))
ax.axvline(2, color=MUTED, lw=0.7)
ax.set_xlabel("|timestamp difference| / offset bound δ")
ax.set_ylabel("P(\"not established\")")
ax.set_ylim(-0.03, 1.03)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.4, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_cert.png"))
print("saved")
