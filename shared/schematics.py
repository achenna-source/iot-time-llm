# -*- coding: utf-8 -*-
"""Architecture schematics for both papers (matplotlib, 300 dpi).

Boxes are sized from their text: each box is drawn after its title and lines have been measured, so no text can
leave its box. A final check verifies that every text lies inside its box, that no two boxes overlap and that no
arrow label overlaps a box; the script stops with an error otherwise.
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
PAD = 0.018          # inner padding, in axes units
TITLE_FS, BODY_FS, LABEL_FS = 7.4, 6.5, 6.2


class Board:
    def __init__(self, w_in, h_in):
        self.fig, self.ax = plt.subplots(figsize=(w_in, h_in))
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.boxes, self.labels = [], []
        self.r = self.fig.canvas.get_renderer()

    def _size(self, t):
        bb = t.get_window_extent(self.r).transformed(self.ax.transAxes.inverted())
        return bb.width, bb.height

    def measure(self, title, lines):
        """natural (w, h) of a box with this content, in axes units."""
        tt = self.ax.text(0, 1, title, fontsize=TITLE_FS, fontweight="bold", va="top")
        bt = self.ax.text(0, 1, chr(10).join(lines), fontsize=BODY_FS, va="top", linespacing=1.4)
        (tw, th), (bw, bh) = self._size(tt), self._size(bt)
        tt.remove()
        bt.remove()
        return max(tw, bw) + 2 * PAD, th + bh + 0.03 + 2 * PAD

    def box(self, x, top, title, lines, fill, edge, min_w=0.0, min_h=0.0):
        """box whose top-left corner is (x, top); returns (x, y, w, h)."""
        tt = self.ax.text(x + PAD, top - PAD, title, fontsize=TITLE_FS, fontweight="bold", color=INK, va="top", ha="left")
        tw, th = self._size(tt)
        bt = self.ax.text(x + PAD, top - PAD - th - 0.03, "\n".join(lines), fontsize=BODY_FS, color=INK2, va="top", ha="left",
                          linespacing=1.4)
        bw, bh = self._size(bt)
        w = max(min_w, max(tw, bw) + 2 * PAD)
        h = max(min_h, th + bh + 0.03 + 2 * PAD)
        patch = FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fill, ec=edge, lw=1.1, zorder=0)
        self.ax.add_patch(patch)
        self.boxes.append({"name": title, "rect": (x, top - h, w, h), "texts": [tt, bt]})
        return (x, top - h, w, h)

    def arrow(self, p, q, label=None, at=None, ha="center", va="center"):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=INK2, shrinkA=0, shrinkB=0))
        if label:
            pos = at or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            t = self.ax.text(pos[0], pos[1], label, fontsize=LABEL_FS, color=INK2, ha=ha, va=va,
                             bbox=dict(fc="white", ec="none", pad=0.6))
            self.labels.append(t)

    def check_and_save(self, path):
        self.fig.canvas.draw()
        inv = self.ax.transAxes.inverted()
        errs = []
        for b in self.boxes:
            x, y, w, h = b["rect"]
            if x < 0 or y < 0 or x + w > 1 or y + h > 1:
                errs.append("box '%s' leaves the figure: %s" % (b["name"], b["rect"]))
            for t in b["texts"]:
                bb = t.get_window_extent(self.r).transformed(inv)
                if bb.x0 < x or bb.x1 > x + w or bb.y0 < y or bb.y1 > y + h:
                    errs.append("text overflows box '%s'" % b["name"])
        for i, a in enumerate(self.boxes):
            for c in self.boxes[i + 1:]:
                ax_, ay, aw, ah = a["rect"]
                cx, cy, cw, ch = c["rect"]
                if ax_ < cx + cw and cx < ax_ + aw and ay < cy + ch and cy < ay + ah:
                    errs.append("boxes overlap: '%s' and '%s'" % (a["name"], c["name"]))
        for t in self.labels:
            bb = t.get_window_extent(self.r).transformed(inv)
            for b in self.boxes:
                x, y, w, h = b["rect"]
                if bb.x0 < x + w and x < bb.x1 and bb.y0 < y + h and y < bb.y1:
                    errs.append("arrow label '%s' overlaps box '%s'" % (t.get_text(), b["name"]))
            if bb.x0 < 0 or bb.x1 > 1 or bb.y0 < 0 or bb.y1 > 1:
                errs.append("arrow label '%s' leaves the figure" % t.get_text())
        if errs:
            raise SystemExit("schematic check failed for %s:\n  " % path + "\n  ".join(errs))
        from matplotlib.transforms import Bbox
        x0 = min(b["rect"][0] for b in self.boxes)
        y0 = min(b["rect"][1] for b in self.boxes)
        x1 = max(b["rect"][0] + b["rect"][2] for b in self.boxes)
        y1 = max(b["rect"][1] + b["rect"][3] for b in self.boxes)
        (px0, py0), (px1, py1) = self.ax.transAxes.transform([(x0, y0), (x1, y1)])
        d = self.fig.dpi_scale_trans.inverted()
        (ix0, iy0), (ix1, iy1) = d.transform([(px0, py0), (px1, py1)])
        self.fig.savefig(path, bbox_inches=Bbox([[ix0 - 0.03, iy0 - 0.03], [ix1 + 0.03, iy1 + 0.03]]))
        plt.close(self.fig)


def age(out):
    B = Board(7.2, 3.1)
    G = 0.045                                     # horizontal gap between boxes
    GATE = ["age  Δ = now − as_of", "normalised age  u = q·Δ", "staleness  p = ψ(1 − exp(−u/ψ))", "(Theorem 1)", "",
               "p > r : read first, then execute", "p ≤ r : execute now", "deterministic, no model call"]
    H1 = B.measure("Freshness gate (Section 4)", GATE)[1]           # common height of the top row
    a = B.box(0.004, 0.99, "Agent context",
              ["instruction: if the valve is idle,", "start a 20-minute cycle", "", "memory: read_state(valve)",
               "= idle, as_of 08:12", "current time: 14:40"], LIGHT["gray"], EDGE_GRAY, min_h=H1)
    m = B.box(a[0] + a[2] + G, 0.99, "Language model",
              ["interprets the instruction", "selects device and command", "", "proposes: start the cycle,",
               "acting on the remembered", "reading"], LIGHT["blue"], SERIES[0], min_h=H1)
    G2 = 0.095                                    # wider gap for the "command" label
    g = B.box(m[0] + m[2] + G2, 0.99, "Freshness gate (Section 4)", GATE,
              LIGHT["aqua"], SERIES[2], min_h=H1, min_w=0.996 - (m[0] + m[2] + G2))
    top2 = min(a[1], m[1], g[1]) - 0.14
    r = B.box(m[0], top2, "Registry and deployment",
              ["exit rate q of each state,", "from the device history", "read price r = c / L", "(battery, airtime)"],
              LIGHT["orange"], SERIES[1])
    d = B.box(g[0] + 0.12, top2, "Device", ["read_state, then the command,", "or the command directly"], LIGHT["gray"], EDGE_GRAY,
              min_w=0.996 - (g[0] + 0.12), min_h=r[3])
    ymid = a[1] + a[3] * 0.55
    B.arrow((a[0] + a[2], ymid), (m[0], ymid))
    yc = m[1] + m[3] * 0.35
    B.arrow((m[0] + m[2], yc), (g[0], yc), "command", at=((m[0] + m[2] + g[0]) / 2, yc + 0.075))
    B.arrow((r[0] + r[2], r[1] + r[3] * 0.6), (g[0] + 0.06, g[1]), "q, r", at=(r[0] + r[2] + 0.035, (r[1] + r[3] + g[1]) / 2 - 0.02))
    xd = d[0] + d[2] * 0.4
    B.arrow((xd, g[1]), (xd, d[1] + d[3]), "decision", at=(xd + 0.012, (g[1] + d[1] + d[3]) / 2), ha="left")
    B.check_and_save(os.path.join(out, "fig_arch.png"))


def order(out):
    B = Board(7.2, 3.3)
    G = 0.05
    LOG = ["31.418 [bp2] pump BP2 stopped", "30.977 [fs2] no flow on FS2", "36.120 [scada] ALARM", "",
           "clock note: offset SD 1 s", "process note: delay 1 s", "history: pump first in 70%"]
    LM = ["acts as a reader:", "finds the candidate events,", "copies their timestamps,", "reads the declared offset",
          "spread, latency and history", "", "output: JSON fields"]
    H1 = max(B.measure("Incident log (device clocks)", LOG)[1], B.measure("Language model", LM)[1])
    lg = B.box(0.004, 0.99, "Incident log (device clocks)", LOG, LIGHT["gray"], EDGE_GRAY, min_h=H1)
    m = B.box(lg[0] + lg[2] + G, 0.99, "Language model", LM, LIGHT["blue"], SERIES[0], min_h=H1)
    xr = m[0] + m[2] + G + 0.02
    c = B.box(xr, 0.99, "Order certifier (Lemma 1)", ["order only if |D| > 2δ", "otherwise: not established"],
              LIGHT["aqua"], SERIES[2], min_w=0.996 - xr)
    b = B.box(xr, c[1] - 0.05, "Bayes attributor (Corollary 1)",
              ["log-odds = ln(π/(1 − π)) + g·D / σ²", "names the favoured event", "ignores the line order"],
              LIGHT["aqua"], SERIES[2], min_w=0.996 - xr, min_h=(c[1] - 0.05) - lg[1])
    k = B.box(0.004, min(lg[1], m[1], b[1]) - 0.14, "Clock metadata (Definition 1)",
              ["offset spread from the oscillator class and the time", "since the last synchronisation, or a hard bound δ"],
              LIGHT["orange"], SERIES[1], min_w=m[0] + m[2] - 0.004)
    ym = lg[1] + lg[3] * 0.55
    B.arrow((lg[0] + lg[2], ym), (m[0], ym))
    xm = m[0] + m[2]
    B.arrow((xm, c[1] + c[3] * 0.4), (xr, c[1] + c[3] * 0.4), "D, δ", at=((xm + xr) / 2, c[1] + c[3] * 0.4 + 0.06))
    B.arrow((xm, b[1] + b[3] * 0.5), (xr, b[1] + b[3] * 0.5), "D, σ, g, π", at=((xm + xr) / 2, b[1] + b[3] * 0.5 - 0.06))
    p0, p1 = (k[0] + k[2], k[1] + k[3] * 0.5), (xr + 0.08, b[1])
    B.arrow(p0, p1, "σ or δ", at=((p0[0] + p1[0]) / 2 + 0.02, (p0[1] + p1[1]) / 2 - 0.02), ha="left")
    B.check_and_save(os.path.join(out, "fig_arch.png"))


age(sys.argv[1])
order(sys.argv[2])
print("saved")
