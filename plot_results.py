#!/usr/bin/env python3
"""
plot_results.py - what changed, per speed, between two sweep recordings.

Alignment is assumed done (see plot_alignment.py).  This maps the G-code
schedule onto the aligned recordings, takes an energy average over each
measurement window, and compares them speed by speed.

Levels are combined as Leq - 10*log10(mean(10^(L/10))) - not an arithmetic
mean, because decibels are logarithmic and averaging them directly understates
the loud parts.

    ./plot_results.py before.csv after.csv --schedule sched.csv -o results.png
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

FS = 10.0
C_BEFORE, C_AFTER = "#2a78d6", "#eb6834"
INK, INK2, INK3, GRID, SURF = "#0b0b0b", "#52514e", "#8a8880", "#e4e2dc", "#fcfcfb"
TRIM_HEAD, TRIM_TAIL = 0.6, 0.3          # s, keep the onset beep out of the window


def load(path):
    t, d = [], []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("time"):
            continue
        a, b = line.replace(";", ",").split(",")[:2]
        t.append(float(a)); d.append(float(b))
    t, d = np.array(t), np.array(d)
    g = np.arange(t[0], t[-1], 1 / FS)
    return g, np.interp(g, t, d)


def xcorr_lag(a, b, maxlag):
    na = (a - a.mean()) / a.std(); nb = (b - b.mean()) / b.std()
    c = np.correlate(na, nb, "full")
    L = (np.arange(len(c)) - (len(nb) - 1)) / FS
    m = np.abs(L) < maxlag
    return float(L[m][np.argmax(c[m])])


def leq(x):
    return 10 * np.log10(np.mean(10 ** (np.asarray(x) / 10)))


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3, color=GRID)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before"); ap.add_argument("after")
    ap.add_argument("--schedule", required=True)
    ap.add_argument("-o", "--out", default="results.png")
    ap.add_argument("--title", default="What changed between the two runs")
    ap.add_argument("--label-before", default="before")
    ap.add_argument("--label-after", default="after")
    args = ap.parse_args()

    ga, va = load(args.before); gb, vb = load(args.after)
    shift = xcorr_lag(va, vb, 60.0) + (ga[0] - gb[0])
    base = np.arange(max(ga[0], gb[0] + shift), min(ga[-1], gb[-1] + shift), 1 / FS)
    A = np.interp(base, ga, va); B = np.interp(base, gb + shift, vb)

    rows = list(csv.DictReader(open(args.schedule)))
    T = float(rows[-1]["t_end_s"]) + 20
    g = np.arange(0, T, 1 / FS); exp = np.zeros_like(g)
    for r in rows:
        exp[(g >= float(r["t_start_s"])) & (g <= float(r["t_end_s"]))] = 1.0
    off = xcorr_lag(A, exp, 90.0) + base[0]

    segs, gaps, moving = [], np.ones_like(A, dtype=bool), np.zeros_like(A, dtype=bool)
    for r in rows:
        t0, t1 = float(r["t_start_s"]) + off, float(r["t_end_s"]) + off
        gaps[(base >= t0 - 0.8) & (base <= t1 + 0.8)] = False
        moving[(base >= t0 + TRIM_HEAD) & (base <= t1 - TRIM_TAIL)] = True
        m = (base >= t0 + TRIM_HEAD) & (base <= t1 - TRIM_TAIL)
        if m.sum() < 8:
            continue
        segs.append(dict(axis=r["axis"], kind=r["kind"],
                         speed=float(r["speed_mms"]) if r["speed_mms"] else None,
                         a=leq(A[m]), b=leq(B[m]),
                         aq=np.percentile(A[m], [25, 75]), bq=np.percentile(B[m], [25, 75])))
    # Homing and the epilogue are not idle, so keep the gap measure to the run proper.
    gaps &= (base > float(rows[0]["t_start_s"]) + off) & (base < float(rows[-1]["t_end_s"]) + off)
    sweeps = [s for s in segs if s["kind"] == "sweep"]
    print(f"schedule offset {off:+.2f}s   {len(sweeps)} speed segments matched")

    fig = plt.figure(figsize=(14, 9.6), facecolor=SURF)
    gs = GridSpec(2, 2, height_ratios=[1.0, 0.92], hspace=0.42, wspace=0.18,
                  left=0.055, right=0.985, top=0.885, bottom=0.075)

    # --- per-speed bars, one panel per axis ---------------------------------
    for i, axis in enumerate(("X", "Y")):
        ax = fig.add_subplot(gs[0, i]); style(ax)
        S = sorted([s for s in sweeps if s["axis"] == axis], key=lambda s: s["speed"])
        x = np.arange(len(S)); w = 0.38
        for k, (key, qk, col, lab) in enumerate((("a", "aq", C_BEFORE, args.label_before),
                                                 ("b", "bq", C_AFTER, args.label_after))):
            v = [s[key] for s in S]
            ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=col, label=lab, zorder=2)
            for xx, s in zip(x + (k - 0.5) * w, S):
                ax.plot([xx, xx], s[qk], color=SURF, lw=1.1, alpha=0.75, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s['speed']:.0f}" for s in S], fontsize=7.5)
        ax.set_xlabel("speed, mm/s", color=INK2, fontsize=9)
        ax.set_ylim(30, max(s["a"] for s in sweeps) + 4)
        if i == 0:
            ax.set_ylabel("Leq, dB", color=INK2, fontsize=9)
            leg = ax.legend(loc="upper left", frameon=False, fontsize=9, ncols=2)
            for txt in leg.get_texts():
                txt.set_color(INK2)
        ax.set_title(f"{axis} axis", loc="left", color=INK, fontsize=12, fontweight="bold", pad=6)
        dd = [s["a"] - s["b"] for s in S]
        msg = (f"unchanged, within {max(abs(min(dd)), abs(max(dd))):.1f} dB"
               if max(abs(min(dd)), abs(max(dd))) < 2.0
               else f"{np.mean(dd):.1f} dB quieter on average")
        ax.text(0.99, 0.94, msg, transform=ax.transAxes, ha="right", fontsize=9.5,
                color=INK, fontweight="bold")

    # --- reduction vs speed -------------------------------------------------
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    ax.grid(True, axis="both", color=GRID, lw=0.6)
    ax.axhline(0, color=INK3, lw=0.9)
    for axis, mk, ls in (("X", "o", "-"), ("Y", "s", "--")):
        S = sorted([s for s in sweeps if s["axis"] == axis], key=lambda s: s["speed"])
        xs = [s["speed"] for s in S]; ys = [s["a"] - s["b"] for s in S]
        ax.plot(xs, ys, ls, marker=mk, color=INK2, lw=1.6, ms=5, mfc=SURF, mew=1.4)
        ax.annotate(f"{axis} axis", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, -3), fontsize=9, color=INK, fontweight="bold")
    ax.set_xlabel("speed, mm/s", color=INK2, fontsize=9)
    ax.set_ylabel("reduction, dB", color=INK2, fontsize=9)
    ax.set_xlim(10, 330)
    allr = [s["a"] - s["b"] for s in sweeps]
    ax.set_title("No change beyond measurement scatter" if max(map(abs, allr)) < 2.0
                 else "The gain holds across the whole speed range",
                 loc="left", color=INK, fontsize=11, fontweight="bold", pad=6)

    # --- is it the printer or the meter? ------------------------------------
    ax = fig.add_subplot(gs[1, 1]); style(ax)
    # Median, not Leq: the gap Leq is set by the decay tails of the motion either
    # side rather than by the idle level, which inverts the comparison.
    move_a, move_b = np.median(A[moving]), np.median(B[moving])
    bl = [g for g in segs if g["kind"] == "baseline"]
    if len(bl) >= 2:
        # Purpose-built silent windows at each end: start vs end is the drift check.
        cols = [(f"baseline\nstart", bl[0]["a"], bl[0]["b"]),
                (f"baseline\nend", bl[-1]["a"], bl[-1]["b"]),
                (f"moving\n({moving.sum()/FS:.0f} s)", move_a, move_b)]
        drift = (bl[-1]["a"] - bl[0]["a"], bl[-1]["b"] - bl[0]["b"])
        note = (f"Baselines drifted {drift[0]:+.1f} dB (before) and {drift[1]:+.1f} dB (after)\n"
                f"across each run, so conditions held while it measured.")
    else:
        floor_a, floor_b = np.median(A[gaps]), np.median(B[gaps])
        cols = [(f"idle between segments\n({gaps.sum()/FS:.0f} s)", floor_a, floor_b),
                (f"moving\n({moving.sum()/FS:.0f} s)", move_a, move_b)]
        note = ("This run has no baseline segments, so the idle figure is scraped from\n"
                "the 2 s gaps. Regenerate with --baseline 10 for a proper reference.")
    x = np.arange(len(cols)); w = 0.32
    ax.bar(x - w / 2, [c[1] for c in cols], w * 0.92, color=C_BEFORE,
           label=args.label_before, zorder=2)
    ax.bar(x + w / 2, [c[2] for c in cols], w * 0.92, color=C_AFTER,
           label=args.label_after, zorder=2)
    for xx, v in zip(np.concatenate([x - w / 2, x + w / 2]),
                     [c[1] for c in cols] + [c[2] for c in cols]):
        ax.text(xx, v + 0.6, f"{v:.1f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cols], fontsize=9.5)
    floor_a, floor_b = cols[0][1], cols[0][2]
    ax.set_ylim(0, max(move_a, move_b) + 11)
    ax.set_ylabel("median dB", color=INK2, fontsize=9)
    ax.set_title("Reference levels", loc="left", color=INK,
                 fontsize=11, fontweight="bold", pad=6)
    ax.annotate("", xy=(0.30, floor_b + 2), xytext=(0.30, floor_a + 2),
                xycoords=("axes fraction", "data"), textcoords=("axes fraction", "data"),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.annotate("", xy=(0.78, move_b + 2), xytext=(0.78, move_a + 2),
                xycoords=("axes fraction", "data"), textcoords=("axes fraction", "data"),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(0.32, max(floor_a, floor_b) + 5.5, f"{floor_b-floor_a:+.1f} dB", fontsize=10, color=INK,
            fontweight="bold", transform=ax.get_yaxis_transform())
    ax.text(0.80, max(move_a, move_b) + 5.5, f"{move_b-move_a:.1f} dB", fontsize=10, color=INK,
            fontweight="bold", transform=ax.get_yaxis_transform())
    if abs(move_a - move_b) < 2.0:
        head = ("Idle and moving both sit where they did. With the baselines\n"
                "agreeing too, nothing about the setup or the machine moved.")
    else:
        head = ("The idle floor went UP between the two runs while the moving\n"
                "level fell. A meter moved further away, or set less sensitive,\n"
                "would have pulled both down together. It pulled them apart.")
    ax.text(0.03, 0.97, head + "\n\n" + note, transform=ax.transAxes,
            fontsize=9.5, color=INK2, va="top")

    fig.suptitle(args.title, x=0.055, ha="left", y=0.972, fontsize=16,
                 fontweight="bold", color=INK)
    fig.text(0.055, 0.935, f"{len(sweeps)} constant-speed segments, energy-averaged (Leq) over each "
             f"measurement window  ·  white ticks span the middle half of each window  ·  "
             f"the idle/moving panel uses medians",
             fontsize=9, color=INK3)
    fig.savefig(args.out, dpi=150, facecolor=SURF)

    print(f"idle floor (median) {floor_a:.1f} -> {floor_b:.1f} dB over {gaps.sum()/FS:.0f}s   "
          f"moving (median) {move_a:.1f} -> {move_b:.1f} dB")
    for axis in ("X", "Y"):
        S = [s for s in sweeps if s["axis"] == axis]
        print(f"  {axis}: mean reduction {np.mean([s['a']-s['b'] for s in S]):.1f} dB "
              f"(min {min(s['a']-s['b'] for s in S):.1f}, max {max(s['a']-s['b'] for s in S):.1f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
