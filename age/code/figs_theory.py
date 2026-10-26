# -*- coding: utf-8 -*-
"""AGE figures that do not depend on the model audit: staleness curves (theory and real traces) and the
cost of time-to-live rules (theory and real fleets).
python figs_theory.py <results_dir> <realdata_age.json> <figures_dir> <realdata_age_v2.json>
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
from figstyle import INK2, MUTED, SERIES, panel_label, plt, setup  # noqa: E402

RES, REAL, FIG = sys.argv[1:4]
REAL2 = json.load(open(sys.argv[4]))
setup()
th = json.load(open(os.path.join(RES, "theory_checks.json")))
real = json.load(open(REAL))
ages_real = np.array(real["ages_s"])

# ------------------------------------------------------------ Figure: staleness curves
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35))
ax = axs[0]
a = np.exp(np.linspace(math.log(60), math.log(4 * 86400), 300))
for k, q in enumerate([1, 4, 16, 64]):
    p = 0.5 * (1 - np.exp(-2 * q / 86400 * a))
    ax.plot(a / 3600, p, color=SERIES[k], label="%g change%s/day" % (q, "" if q == 1 else "s"))
    ustar = -0.5 * math.log(1 - 2 * 0.1)
    ax.plot(ustar / (q / 86400) / 3600, 0.1, marker="o", color=SERIES[k], ms=4)
ax.axhline(0.1, color=INK2, lw=0.9, ls="--")
ax.text(0.02, 0.115, "r = 0.1", color=INK2, fontsize=7, transform=ax.get_yaxis_transform())
ax.set_xscale("log")
ax.set_xlabel("age of the reading (h)")
ax.set_ylabel("P(reading is outdated)")
ax.set_title("Binary devices (Theorem 1)", loc="left")
ax.legend(loc="upper left", fontsize=6.5, handlelength=1.4, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
panel_label(ax, "a")

ax = axs[1]
t = np.array(th["dutycycle"]["t"])
p = np.array(th["dutycycle"]["p"])
ax.plot(t, p, color=SERIES[0])
ax.axhline(0.6, color=INK2, lw=0.9, ls="--")
ax.fill_between(t, 0, 1, where=p > 0.6, color=SERIES[0], alpha=0.10, lw=0)
ax.text(88, 0.05, "shaded: re-reading optimal at r = 0.6", color=INK2, fontsize=6.4, ha="right")
ax.set_xlabel("age of an 'on' reading (min)")
ax.set_ylim(0, 1.02)
ax.text(2, 0.62, "r = 0.6", color=INK2, fontsize=7, ha="left")
ax.set_title("Timer-driven pump (Example 1)", loc="left")
panel_label(ax, "b")

ax = axs[2]
occ = [d for d in real["devices"] if d["kind"] == "occupancy"]
d = max(occ, key=lambda x: x["span_days"])
for k, (x, lab) in enumerate((("1", "read 'occupied'"), ("0", "read 'vacant'"))):
    c = d["curves"].get(x)
    if not c:
        continue
    ax.plot(ages_real / 3600, c["emp"], color=SERIES[k], label=lab + ", measured")
    ax.plot(ages_real / 3600, c["ctmc"], color=SERIES[k], ls=":", lw=1.3, label=lab + ", Markov fit")
for hh in (24, 48):
    ax.axvline(hh, color=MUTED, lw=0.7, ls="-")
    ax.text(hh * (0.93 if hh == 24 else 1.07), 0.36, "%d h" % hh, color=MUTED, fontsize=6.2,
            ha="right" if hh == 24 else "left")
ax.set_xscale("log")
ax.set_xlabel("age of the reading (h)")
ax.set_title("Office occupancy (UCI)", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=6.2, handlelength=1.6, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
panel_label(ax, "c")
fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(FIG, "fig_staleness.png"))

# ------------------------------------------------------------ Figure: time-to-live rules
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1.5]})
ax = axs[0]
tr = th["ttl_regret"]
ax.plot([x["ratio"] for x in tr], [x["rel"] for x in tr], color=SERIES[0], marker="o")
for x in tr:
    ax.annotate("%.2f" % x["rel"], (x["ratio"], x["rel"]), textcoords="offset points", xytext=(4, -9), fontsize=6.5, color=INK2)
ax.set_xscale("log", base=4)
ax.set_xticks([1, 4, 16, 64, 256])
ax.set_xticklabels(["1", "4", "16", "64", "256"])
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel("ratio of change rates in a two-class fleet")
ax.set_ylabel("normalised regret of best TTL")
ax.set_title("Theorem 2, r = 0.1", loc="left")
panel_label(ax, "a")

ax = axs[1]
fleets = [("door", "door contacts"), ("motion", "motion sensors"), ("all", "all %d devices" % REAL2["n_devices"])]
pols = [("ttl_single", "single TTL"), ("ttl_state", "TTL per device state"), ("exit_rate_rule", "exit-rate rule (Prop. 4)"),
        ("two_param_gate", "two-parameter gate (Thm 1)")]
w = 0.2
for j, (key, lab) in enumerate(pols):
    vals = [REAL2["by"][fk]["share_r0.1"]["rho"][key] for fk, _ in fleets]
    xs = np.arange(len(fleets)) + (j - 1.5) * w
    ax.bar(xs, np.minimum(vals, 2.6), width=w - 0.03, color=SERIES[j], label=lab)
    for x0, v0 in zip(xs, vals):
        if v0 > 2.6:
            ax.text(x0, 2.42, ("%.1f" % v0) if v0 < 10 else ("%.0f" % v0), ha="center", fontsize=6.0, color="white", fontweight="bold")
        else:
            ax.text(x0, v0 + 0.03, "%.2f" % v0, ha="center", fontsize=6.0, color=INK2)
ax.axhline(1, color=INK2, lw=0.9, ls="--", label="better fixed rule")
ax.set_xticks(np.arange(len(fleets)))
ax.set_xticklabels([lab for _, lab in fleets])
ax.set_ylabel("normalised regret")
ax.set_ylim(0, 2.6)
ax.set_title("Real fleets (CASAS), held-out half, r = 0.1", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=6.4, ncol=2)
panel_label(ax, "b")
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIG, "fig_ttl.png"))
print("saved")
