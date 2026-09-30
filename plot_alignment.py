#!/usr/bin/env python3
"""
plot_alignment.py - overlay two sound-meter logs of the same sweep, aligned.

The G-code run is deterministic, so two recordings differ only by when the meter
was started: one constant offset.  Windowed cross-correlation recovers it, and
the windowing is the check - an offset that drifts across the run means the two
recordings are not of the same program.

    ./plot_alignment.py before.csv after.csv -o alignment.png
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

FS = 10.0                       # common grid, Hz

# dataviz reference palette, categorical slots 1 and 2 (light mode)
C_BEFORE = "#2a78d6"
C_AFTER = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e4e2dc"
SURF = "#fcfcfb"


def load(path):
    t, d = [], []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("time"):
            continue
        a, b = line.replace(";", ",").split(",")[:2]
        t.append(float(a))
        d.append(float(b))
    t, d = np.array(t), np.array(d)
    g = np.arange(t[0], t[-1], 1 / FS)
    return g, np.interp(g, t, d)


def lag_scan(va, vb, maxlag=60.0):
    na = (va - va.mean()) / va.std()
    nb = (vb - vb.mean()) / vb.std()
    c = np.correlate(na, nb, "full")
    L = (np.arange(len(c)) - (len(nb) - 1)) / FS
    m = np.abs(L) < maxlag
    return L[m], c[m] / len(nb)


def windowed(ga, va, gb, vb, win=80.0, step=60.0, maxlag=25.0):
    out = []
    lo, hi = max(ga[0], gb[0]), min(ga[-1], gb[-1])
    while lo + win <= hi:
        ma = (ga >= lo) & (ga < lo + win)
        mb = (gb >= lo - maxlag / 2) & (gb < lo + win + maxlag / 2)
        if ma.sum() > 200 and mb.sum() > 200:
            L, c = lag_scan(va[ma], vb[mb], maxlag)
            out.append((lo + win / 2, L[np.argmax(c)] + (ga[ma][0] - gb[mb][0])))
        lo += step
    return out


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3, color=GRID)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("-o", "--out", default="alignment.png")
    ap.add_argument("--label-before", default="before calibration")
    ap.add_argument("--label-after", default="after calibration")
    ap.add_argument("--schedule", default="")
    ap.add_argument("--zoom", type=float, default=25.0, help="width of each zoom window, s")
    args = ap.parse_args()

    ga, va = load(args.before)
    gb, vb = load(args.after)

    wins = windowed(ga, va, gb, vb)
    lags = np.array([w[1] for w in wins])
    shift = float(np.median(lags))
    spread = float(lags.max() - lags.min())

    # Put both on one clock, then re-zero so t=0 is the first common sample.
    base = np.arange(max(ga[0], gb[0] + shift), min(ga[-1], gb[-1] + shift), 1 / FS)
    A = np.interp(base, ga, va)
    B = np.interp(base, gb + shift, vb)
    t = base - base[0]
    r = float(np.corrcoef(A, B)[0, 1])
    delta = A - B

    segs = []
    if args.schedule and os.path.exists(args.schedule):
        rows = list(csv.DictReader(open(args.schedule)))
        T = float(rows[-1]["t_end_s"]) + 20
        g = np.arange(0, T, 1 / FS)
        exp = np.zeros_like(g)
        for row in rows:
            exp[(g >= float(row["t_start_s"])) & (g <= float(row["t_end_s"]))] = 1.0
        L, c = lag_scan(A, exp, 90.0)
        off = float(L[np.argmax(c)] + base[0])
        for row in rows:
            segs.append((float(row["t_start_s"]) + off - base[0],
                         float(row["t_end_s"]) + off - base[0],
                         row["axis"], row["kind"], row["speed_mms"]))

    fig = plt.figure(figsize=(14, 11.5), facecolor=SURF)
    gs = GridSpec(4, 3, height_ratios=[2.1, 1.05, 1.25, 1.0], hspace=0.52, wspace=0.26,
                  left=0.06, right=0.985, top=0.895, bottom=0.055)

    # --- full run -----------------------------------------------------------
    ax = fig.add_subplot(gs[0, :]); style(ax)
    for s0, s1, axis, kind, sp in segs:
        if s1 < 0 or s0 > t[-1]:
            continue
        ax.axvspan(s0, s1, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.plot(t, A, lw=1.1, color=C_BEFORE, label=args.label_before)
    ax.plot(t, B, lw=1.1, color=C_AFTER, label=args.label_after)
    ax.set_xlim(0, t[-1]); ax.set_ylabel("dB", color=INK2, fontsize=9)
    ax.set_title("Both recordings on one time base", loc="left", color=INK,
                 fontsize=12, fontweight="bold", pad=8)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, ncols=2)
    for txt in leg.get_texts():
        txt.set_color(INK2)
    if segs:
        ax.text(0.995, 0.03, "shaded = measurement windows from the G-code schedule",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=INK3)

    # --- difference ---------------------------------------------------------
    ax = fig.add_subplot(gs[1, :]); style(ax)
    ax.axhline(0, color=INK3, lw=0.9)
    ax.fill_between(t, delta, 0, color=C_BEFORE, alpha=0.18, lw=0)
    ax.plot(t, delta, lw=0.9, color=INK2)
    ax.set_xlim(0, t[-1]); ax.set_ylabel("difference, dB", color=INK2, fontsize=9)
    ax.set_title("Before minus after — misalignment would show as sharp spikes at every "
                 "segment edge, not a smooth offset", loc="left", color=INK, fontsize=10, pad=6)

    # --- zooms --------------------------------------------------------------
    marks = [0.16, 0.5, 0.84]
    for i, frac in enumerate(marks):
        ax = fig.add_subplot(gs[2, i]); style(ax)
        c = t[-1] * frac
        lo, hi = c - args.zoom / 2, c + args.zoom / 2
        m = (t >= lo) & (t <= hi)
        for s0, s1, axis, kind, sp in segs:
            if s1 > lo and s0 < hi:
                ax.axvspan(s0, s1, color=GRID, alpha=0.55, lw=0, zorder=0)
        ax.plot(t[m], A[m], lw=1.6, color=C_BEFORE)
        ax.plot(t[m], B[m], lw=1.6, color=C_AFTER)
        ax.set_xlim(lo, hi)
        ax.set_title(f"{lo:.0f}–{hi:.0f} s", loc="left", color=INK2, fontsize=9)
        if i == 0:
            ax.set_ylabel("dB", color=INK2, fontsize=9)
            zoom_top = ax.get_position().y1
    fig.text(0.06, zoom_top + 0.030, f"Zoomed to {args.zoom:.0f} s — the level differs but "
             f"the edges land together, which is what alignment looks like",
             color=INK, fontsize=10)

    # --- lag scan + windowed lags ------------------------------------------
    ax = fig.add_subplot(gs[3, 0]); style(ax)
    L, c = lag_scan(va, vb)
    L = L + (ga[0] - gb[0])
    ax.plot(L, c, lw=1.4, color=INK2)
    ax.axvline(shift, color=C_AFTER, lw=1.4)
    ax.plot([shift], [c.max()], "o", ms=6, color=C_AFTER, zorder=5)
    ax.annotate(f"{shift:+.2f} s", (shift, c.max()), textcoords="offset points",
                xytext=(8, -2), color=INK, fontsize=9, fontweight="bold")
    ax.set_xlim(-30, 30); ax.set_xlabel("trial offset, s", color=INK2, fontsize=9)
    ax.set_ylabel("correlation", color=INK2, fontsize=9)
    ax.set_title("One clear optimum", loc="left", color=INK, fontsize=10)

    ax = fig.add_subplot(gs[3, 1]); style(ax)
    ax.plot([w[0] for w in wins], lags, "o-", color=C_BEFORE, lw=1.6, ms=6)
    ax.axhline(shift, color=INK3, lw=0.9, ls="--")
    ax.set_ylim(shift - 1.2, shift + 1.2)
    ax.set_xlabel("position in run, s", color=INK2, fontsize=9)
    ax.set_ylabel("best offset, s", color=INK2, fontsize=9)
    ax.set_title("Offset holds across the run", loc="left", color=INK, fontsize=10)

    ax = fig.add_subplot(gs[3, 2]); ax.axis("off"); ax.set_facecolor(SURF)
    fa, fb = np.percentile(A, 5), np.percentile(B, 5)
    ma, mb = np.percentile(A, 90), np.percentile(B, 90)
    rows_ = [
        ("", "before", "after", ""),
        ("quietest 5%", f"{fa:.1f}", f"{fb:.1f}", f"{fb-fa:+.1f}"),
        ("while moving", f"{ma:.1f}", f"{mb:.1f}", f"{mb-ma:+.1f}"),
        ("median", f"{np.median(A):.1f}", f"{np.median(B):.1f}", f"{np.median(B)-np.median(A):+.1f}"),
        (None, None, None, None),
        ("offset applied", f"{shift:+.3f} s", "", ""),
        ("drift across run", f"{spread:.2f} s", "", ""),
        ("correlation r", f"{r:.3f}", "", ""),
    ]
    for j, row in enumerate(rows_):
        if row[0] is None:
            continue
        y = 0.94 - j * 0.118
        head = (j == 0)
        ax.text(0.0, y, row[0], fontsize=9, color=INK2, transform=ax.transAxes)
        for xx, val, col in ((0.60, row[1], INK), (0.80, row[2], INK), (1.0, row[3], INK2)):
            if not val:
                continue
            ax.text(xx, y, val, fontsize=8.5 if head else 10, ha="right", transform=ax.transAxes,
                    color=INK2 if head else col, fontweight="normal" if head else "bold")
    ax.text(0.0, 1.06, "dB summary", fontsize=10, color=INK, fontweight="bold",
            transform=ax.transAxes)

    fig.suptitle("MK4S noise sweep — before and after phase stepping calibration",
                 x=0.06, ha="left", y=0.978, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.06, 0.953, f"{os.path.basename(args.before)}  vs  {os.path.basename(args.after)}"
             f"   ·   time relative to the first common sample",
             fontsize=9, color=INK3)
    fig.savefig(args.out, dpi=150, facecolor=SURF)
    print(f"offset {shift:+.3f}s  drift {spread:.2f}s  r {r:.3f}  overlap {t[-1]:.0f}s")
    print(f"median delta {np.median(delta):+.1f} dB   wrote {args.out}")


if __name__ == "__main__":
    main()
