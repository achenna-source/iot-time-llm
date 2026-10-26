# -*- coding: utf-8 -*-
"""ORDER figures that do not depend on the model audit: the error floor and the fan-out effect (theory),
and the timestamp-only error on measured latencies as a function of the synchronisation interval.
python figs_theory.py <results_dir> <realdata_order.json> <figures_dir>
"""
import json
import math
import os
import sys

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
from figstyle import INK2, MUTED, SERIES, panel_label, plt, setup  # noqa: E402

RES, REAL, FIG = sys.argv[1:4]
setup()
th = json.load(open(os.path.join(RES, "theory_checks.json")))

# ------------------------------------------------------------ Figure: error floor and fan-out
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax = axs[0]
kap = np.linspace(0.01, 3, 300)
for k, pi in enumerate((0.5, 0.7, 0.9)):
    exact = []
    for kk in kap:
        g, s = kk, 1.0          # deterministic latency: kappa = g / sigma_theta
        dstar = (s ** 2 / g) * math.log((1 - pi) / pi)
        sd = math.sqrt(2) * s
        exact.append(pi * stats.norm.cdf((dstar - g) / sd) + (1 - pi) * (1 - stats.norm.cdf((dstar + g) / sd)))
    ax.plot(kap, exact, color=SERIES[k], label="Bayes error, prior %.1f" % pi)
    ax.plot(kap, 2 * (1 - pi) * stats.norm.cdf(-kap / math.sqrt(2)), color=SERIES[k], ls="--", lw=1.2)
ax.text(1.45, 0.19, "dashed: closed-form floor of\nCorollary 1, attained at a\nuniform prior (prior 0.5)", color=INK2, fontsize=6.4)
ax.set_xlabel("κ = RMS latency / clock-offset SD")
ax.set_ylabel("P(wrong first event)")
ax.set_title("Error floor (Theorem 2, Corollary 1)", loc="left")
ax.legend(loc="upper right", fontsize=6.4, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
ax.set_ylim(0, 0.52)
panel_label(ax, "a")

ax = axs[1]
fan = th["fanout"]
for k, gm in enumerate((0.5, 1, 2, 3)):
    rows = [r for r in fan if r["gamma"] == gm]
    ax.plot([r["k"] - 1 for r in rows], [r["P_k"] for r in rows], color=SERIES[k], marker="os^D"[k])
    ax.text(99 * 1.18, rows[-1]["P_k"], "%g" % gm, color=INK2, fontsize=7, va="center")
ax.text(99 * 1.18, 0.86, "latency /\noffset SD", color=INK2, fontsize=6.4, va="center")
ax.set_xscale("log")
ax.set_xlim(0.8, 260)
ax.set_xticks([1, 2, 4, 9, 29, 99])
ax.set_xticklabels(["1", "2", "4", "9", "29", "99"])
ax.set_ylim(0, 1.02)
ax.set_xlabel("number of downstream effects")
ax.set_ylabel("P(earliest timestamp is the root)")
ax.set_title("Fan-out (Proposition 1)", loc="left")
panel_label(ax, "b")
fig.tight_layout(w_pad=2)
fig.savefig(os.path.join(FIG, "fig_floor.png"))

# ------------------------------------------------------------ Figure: measured latencies vs synchronisation interval
summ = json.load(open(os.path.join(RES, "realdata_order_summary.json")))
fig, ax = plt.subplots(figsize=(4.6, 2.8))
names = {"homes (CASAS): door -> motion": "homes (CASAS)", "industrial testbed (HAI): valve demand -> position": "industrial (HAI)"}
for key, curve in summ["curves"].items():
    src, clock, sync = [x.strip() for x in key.split("|")]
    if not sync.startswith("NTP"):
        continue
    k = list(names).index(src)
    ls = "-" if clock.startswith("XO") else ":"
    ax.plot([c[0] / 86400 for c in curve], [c[1] for c in curve], color=SERIES[k], ls=ls,
            label="%s, %s" % (names[src], clock.replace("+-", "±")))
for d, lab in ((1, "daily"), (7, "weekly"), (30, "monthly")):
    ax.axvline(d, color=MUTED, lw=0.7)
    ax.text(d * 1.06, 0.505, lab, color=MUTED, fontsize=6.3)
ax.set_xscale("log")
ax.set_ylim(0, 0.55)
ax.set_xlabel("time between clock synchronisations (days), NTP residual 10 ms")
ax.set_ylabel("Bayes error from timestamps alone")
ax.text(0.0075, 0.06, "exact under Definition 1, prior 1/2", color=INK2, fontsize=6.2)
ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93), fontsize=6.0, frameon=True, facecolor="white",
          edgecolor="none", framealpha=0.92)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_sync.png"))
print("saved")
