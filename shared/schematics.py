# -*- coding: utf-8 -*-
"""Architecture schematics for both papers (matplotlib, 300 dpi).
python schematics.py <age_figures_dir> <order_figures_dir>
"""
import os
import sys

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
from figstyle import INK, INK2, SERIES, plt, setup  # noqa: E402

setup()
plt.rcParams["axes.grid"] = False
LIGHT = {"blue": "#e8f1fc", "orange": "#fdeee6", "aqua": "#e3f6ef", "gray": "#f3f2ee"}
EDGE_GRAY = "#c3c2b7"


def box(ax, x, y, w, h, title, lines, fill, edge, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015", fc=fill, ec=edge, lw=1.1))
    ax.text(x + 0.014, y + h - 0.03, title, fontsize=7.6, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.014, y + h - 0.10, "\n".join(lines), fontsize=fs, color=INK2, va="top", linespacing=1.45)


def arrow(ax, p, q, label=None, off=(0, 0.02), ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=INK2, shrinkA=1, shrinkB=1))
    if label:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, fontsize=6.3, color=INK2, ha=ha)


def age(out):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.005, 0.47, 0.25, 0.50, "Agent context",
        ["instruction: if the valve is idle,", "start a 20-minute cycle", "", "memory: read_state(valve)", "  = idle, as_of 08:12", "current time: 14:40"],
        LIGHT["gray"], EDGE_GRAY)
    box(ax, 0.30, 0.47, 0.23, 0.50, "Language model",
        ["interprets the instruction", "selects device and command", "", "proposes: start the cycle,", "acting on the remembered", "reading"],
        LIGHT["blue"], SERIES[0])
    box(ax, 0.58, 0.47, 0.405, 0.50, "Freshness gate (Section 4)",
        ["age  Δ = now − as_of", "normalised age  u = q·Δ", "staleness  p = ψ (1 − exp(−u/ψ))   (Theorem 1)", "",
         "p > r :  read first, then execute", "p ≤ r :  execute now", "deterministic, no extra model call"],
        LIGHT["aqua"], SERIES[2])
    box(ax, 0.30, 0.02, 0.34, 0.33, "Device registry and deployment",
        ["changes per day q and time share π", "of each state, from the device history", "read price r = c / L (battery, airtime)"],
        LIGHT["orange"], SERIES[1])
    box(ax, 0.70, 0.02, 0.285, 0.33, "Device", ["read_state, then the command", "or the command directly"], LIGHT["gray"], EDGE_GRAY)
    arrow(ax, (0.262, 0.72), (0.296, 0.72))
    arrow(ax, (0.538, 0.62), (0.576, 0.62), "command", off=(0, -0.05))
    arrow(ax, (0.60, 0.36), (0.66, 0.46), "q, π, r", off=(-0.045, 0.0))
    arrow(ax, (0.85, 0.46), (0.85, 0.36), "decision", off=(0.012, -0.01), ha="left")
    fig.savefig(os.path.join(out, "fig_arch.png"))
    plt.close(fig)


def order(out):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.005, 0.36, 0.30, 0.61, "Incident log (device clocks)",
        ["31.418 [bp2] pump BP2 stopped", "30.977 [fs2] no flow on FS2", "36.120 [scada] ALARM", "",
         "clock note: offset SD 1 s", "process note: delay 1 s", "history: pump first in 70%"],
        LIGHT["gray"], EDGE_GRAY, fs=6.3)
    box(ax, 0.35, 0.36, 0.24, 0.61, "Language model",
        ["acts as a reader:", "finds the candidate events", "copies their timestamps", "reads the declared offset", "spread, latency and history", "", "output: JSON fields"],
        LIGHT["blue"], SERIES[0])
    box(ax, 0.64, 0.71, 0.345, 0.26, "Certifier (Lemma 1)", ["order only if |D| > 2δ", "otherwise: not established"], LIGHT["aqua"], SERIES[2])
    box(ax, 0.64, 0.36, 0.345, 0.30, "Bayes attributor (Theorem 2)",
        ["log-odds = ln(π/(1 − π)) + g·D / σ²", "names the favoured event", "ignores the line order"], LIGHT["aqua"], SERIES[2])
    box(ax, 0.005, 0.02, 0.585, 0.26, "Clock metadata (Definition 1)",
        ["offset spread σ² = σs² + σρ²·T²/3 from the oscillator class", "and the time T since the last synchronisation, or a hard bound δ"],
        LIGHT["orange"], SERIES[1])
    arrow(ax, (0.312, 0.66), (0.346, 0.66))
    arrow(ax, (0.596, 0.75), (0.636, 0.80), "D, δ", off=(-0.012, 0.03))
    arrow(ax, (0.596, 0.50), (0.636, 0.47), "D, σ, g, π", off=(-0.02, -0.075))
    arrow(ax, (0.596, 0.15), (0.75, 0.352), "σ or δ", off=(0.045, -0.04))
    fig.savefig(os.path.join(out, "fig_arch.png"))
    plt.close(fig)


age(sys.argv[1])
order(sys.argv[2])
print("saved")
