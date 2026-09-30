#!/usr/bin/env python3
"""
plot_states.py - compare three machine states across one speed sweep.

Takes three recordings of the same G-code and lines each one up to the schedule
independently, so a run that was left recording after the print still works.
Levels are Leq over each measurement window.

    ./plot_states.py off.csv uncal.csv cal.csv --schedule sched.csv -o states.png
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
# dataviz reference palette, categorical slots 1-3 (validated all-pairs)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, INK3, GRID, SURF = "#0b0b0b", "#52514e", "#8a8880", "#e4e2dc", "#fcfcfb"
TRIM_HEAD, TRIM_TAIL = 0.6, 0.3


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


def fit(v, exp, maxlag=120.0):
    na = (v - v.mean()) / v.std(); nb = (exp - exp.mean()) / exp.std()
    c = np.correlate(na, nb, "full")
    L = (np.arange(len(c)) - (len(nb) - 1)) / FS
    m = np.abs(L) < maxlag
    return float(L[m][np.argmax(c[m])]), float((c[m].max() - c[m].mean()) / c[m].std())


def leq(x):
    return 10 * np.log10(np.mean(10 ** (np.asarray(x) / 10)))


def robust_leq(x, k=3.0):
    """Leq after dropping samples more than k MADs from the median.

    Leq is an energy average, so one stray sample dominates it: a single 45 dB
    tick inside a 28 dB window lifts the result by nearly 2 dB.  That is what
    made the baselines look like they drifted 5 dB between the two ends of a
    run.  Rejecting on the median absolute deviation keeps the energy average
    where it belongs while ignoring door thumps, clicks and a stray beep.
    """
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    keep = np.ones(len(x), bool) if mad <= 1e-9 else np.abs(x - med) <= k * mad
    if keep.sum() < 3:
        keep = np.ones(len(x), bool)
    return leq(x[keep]), int((~keep).sum()), len(x)


def style(ax, ygrid_only=True):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3, color=GRID)
    ax.grid(True, axis="y" if ygrid_only else "both", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def profile(path, rows, exp):
    """Per-segment levels, plus the floor sampled from the gaps between them."""
    g, v = load(path)
    off, z = fit(v, exp)
    off += g[0]
    out, rejected, raw = {}, 0, {}
    for r in rows:
        t0, t1 = float(r["t_start_s"]) + off, float(r["t_end_s"]) + off
        m = (g >= t0 + TRIM_HEAD) & (g <= t1 - TRIM_TAIL)
        if m.sum() < 8:
            continue
        val, nrej, _ = robust_leq(v[m])
        rejected += nrej
        key = ((r["axis"], float(r["speed_mms"])) if r["kind"] == "sweep"
               else ("base", r["notes"]) if r["kind"] == "baseline" else None)
        if key:
            out[key] = val
            raw[key] = leq(v[m])

    # Floor from the 2 s gaps.  Skip any gap where the axis changes: the sweep
    # jogs to the new axis with no pause first, so that "gap" contains motion.
    sw = [r for r in rows if r["kind"] == "sweep"]
    floor = []
    for a, b in zip(sw[:-1], sw[1:]):
        if a["axis"] != b["axis"]:
            continue
        gs = float(a["t_end_s"]) + off + 1.0
        ge = float(b["t_start_s"]) + off - 0.15
        m = (g >= gs) & (g <= ge)
        if ge - gs >= 0.5 and m.sum() >= 5:
            floor.append(((gs + ge) / 2 - off, float(np.median(v[m]))))
    out[("floor",)] = floor
    out[("meta",)] = {"rejected": rejected, "raw": raw}
    out[("trace",)] = (g - off, v)          # x in schedule time
    return out, z


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs=3)
    ap.add_argument("--labels", default="phase stepping off|on, no correction|on, calibrated",
                    help="three labels separated by | (they may contain commas)")
    ap.add_argument("--schedule", required=True)
    ap.add_argument("-o", "--out", default="states.png")
    ap.add_argument("--title",
                    default="Phase stepping calibration takes 5 dB off the MK4S Y axis")
    ap.add_argument("--subtitle",
                    default="Prusa MK4S, firmware 6.5.7. Phase stepping ships disabled and "
                            "PrusaSlicer never enables it \u2014 M970 X1 Y1 in the start G-code "
                            "does. Sound meter in a fixed spot, 36 constant-speed segments from "
                            "30 to 300 mm/s, run three times a power cycle apart. The gain is "
                            "entirely the calibrated correction table, only on Y and only below "
                            "100 mm/s.")
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split("|")]
    if len(labels) != 3:
        raise SystemExit(f"need exactly 3 labels separated by |, got {len(labels)}")
    rows = list(csv.DictReader(open(args.schedule)))
    T = float(rows[-1]["t_end_s"]) + 20
    g = np.arange(0, T, 1 / FS); exp = np.zeros_like(g)
    for r in rows:
        exp[(g >= float(r["t_start_s"])) & (g <= float(r["t_end_s"]))] = 1.0

    P, Z = [], []
    for f in args.runs:
        p, z = profile(f, rows, exp)
        P.append(p); Z.append(z)
        print(f"{os.path.basename(f)[:44]:46s} schedule fit z={z:.1f}")

    speeds = sorted({k[1] for k in P[0] if k[0] == "X"})
    lo = [s for s in speeds if s <= 100]
    hi = [s for s in speeds if s > 100]

    fig = plt.figure(figsize=(14, 15.2), facecolor=SURF)
    gs = GridSpec(4, 6, height_ratios=[0.62, 0.55, 1.0, 0.72], hspace=0.50, wspace=0.55,
                  left=0.055, right=0.985, top=0.916, bottom=0.045)

    for i, axis in enumerate(("X", "Y")):
        ax = fig.add_subplot(gs[2, 0:3] if i == 0 else gs[2, 3:6]); style(ax)
        x = np.arange(len(speeds)); w = 0.27
        for k, (p, lab, col) in enumerate(zip(P, labels, COLORS)):
            ax.bar(x + (k - 1) * w, [p[(axis, s)] for s in speeds], w * 0.9,
                   color=col, label=lab, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels([f"{s:.0f}" for s in speeds], fontsize=7.5)
        ax.set_xlabel("speed, mm/s", color=INK2, fontsize=9)
        ax.set_ylim(30, 54)
        gain = np.mean([P[0][(axis, s)] - P[2][(axis, s)] for s in lo])
        if i == 0:
            ax.set_ylabel("Leq, dB", color=INK2, fontsize=9)
            leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncols=3,
                            columnspacing=1.1, handlelength=1.2)
            for t in leg.get_texts():
                t.set_color(INK2)
        ax.set_title(f"{axis} axis", loc="left", color=INK, fontsize=12, fontweight="bold", pad=6)
        verb = "quieter" if gain > 0 else "louder"
        ax.text(1.0, 1.015, f"{abs(gain):.1f} dB {verb} at 30-100 mm/s", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10.5, color=INK, fontweight="bold")

    # --- where the gain lives ----------------------------------------------
    ax = fig.add_subplot(gs[3, 0:3]); style(ax, ygrid_only=False)
    ax.axhline(0, color=INK3, lw=0.9)
    for axis, mk, ls, dy in (("X", "o", "-", 12), ("Y", "s", "--", -16)):
        ys = [P[0][(axis, s)] - P[2][(axis, s)] for s in speeds]
        ax.plot(speeds, ys, ls, marker=mk, color=INK2, lw=1.6, ms=5, mfc=SURF, mew=1.4)
        j = int(np.argmax(np.abs(ys)))          # label where the curves are furthest apart
        ax.annotate(f"{axis} axis", (speeds[j], ys[j]), textcoords="offset points",
                    xytext=(6, dy), fontsize=9.5, color=INK, fontweight="bold")
    ax.axvspan(25, 105, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(65, ax.get_ylim()[1] * 0.93, "singing band", ha="center", fontsize=8.5, color=INK3)
    ax.set_xlabel("speed, mm/s", color=INK2, fontsize=9)
    ax.set_ylabel("quieter by, dB", color=INK2, fontsize=9)
    ax.set_xlim(10, 330)
    ax.set_title("All of the gain is on Y, below 100 mm/s", loc="left",
                 color=INK, fontsize=11, fontweight="bold", pad=6)

    # --- decomposition ------------------------------------------------------
    ax = fig.add_subplot(gs[3, 3:6]); ax.axis("off"); ax.set_facecolor(SURF)
    ax.text(0, 1.0, "Splitting the two effects", fontsize=11, color=INK,
            fontweight="bold", transform=ax.transAxes)
    hdr = ["", "drive mode", "correction", "net"]
    colw = [0.40, 0.63, 0.85, 1.0]
    for j, h in enumerate(hdr):
        ax.text(colw[j], 0.88, h, fontsize=8.5, color=INK2, ha="right" if j else "left",
                transform=ax.transAxes)
    row = 0
    for axis in ("X", "Y"):
        for nm, ss in (("30-100 mm/s", lo), ("120-300 mm/s", hi)):
            o = np.mean([P[0][(axis, s)] for s in ss])
            u = np.mean([P[1][(axis, s)] for s in ss])
            c = np.mean([P[2][(axis, s)] for s in ss])
            y = 0.76 - row * 0.115
            ax.text(0, y, f"{axis}  {nm}", fontsize=9, color=INK2, transform=ax.transAxes)
            for j, val in enumerate((u - o, c - u, c - o)):
                big = abs(val) >= 2
                ax.text(colw[j + 1], y, f"{val:+.1f}", fontsize=10.5 if big else 9.5,
                        ha="right", transform=ax.transAxes,
                        color=INK if big else INK3, fontweight="bold" if big else "normal")
            row += 1
    ax.text(0, 0.35, "Enabling phase stepping with no correction\ntable costs about 1.5 dB. "
            "Every gain comes\nfrom the calibrated table, and only on Y.",
            fontsize=9, color=INK2, transform=ax.transAxes, va="top")
    bits = []
    for lbl, p in zip(["off", "uncal", "calib"], P):
        fl = np.array([f[1] for f in p[("floor",)]])
        bits.append(f"{lbl} {np.median(fl):.1f}")
    ax.text(0, 0.17, "floor between segments, median dB\n   " + "    ".join(bits)
            + "\noutlier samples rejected\n   "
            + "   ".join(f"{l} {p[('meta',)]['rejected']}" for l, p in zip(["off", "uncal", "calib"], P)),
            fontsize=8, color=INK3, transform=ax.transAxes, va="top")

    # --- raw traces, whole run --------------------------------------------
    sweeps = [r for r in rows if r["kind"] == "sweep"]
    t0all = float(rows[0]["t_start_s"]) - 6
    t1all = float(rows[-1]["t_end_s"]) + 6

    def draw_raw(ax, lo, hi, label_segments=False):
        style(ax, ygrid_only=False)
        for r in rows:
            a, b = float(r["t_start_s"]), float(r["t_end_s"])
            if b < lo or a > hi:
                continue
            ax.axvspan(a, b, color=GRID, alpha=0.55, lw=0, zorder=0)
            if label_segments and r["kind"] == "sweep" and lo + 1 < (a + b) / 2 < hi - 1:
                ax.text((a + b) / 2, 0.955, f"{r['axis']}{float(r['speed_mms']):.0f}",
                        transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=7.5, color=INK2)
        for p, col in zip(P, COLORS):
            tt, vv = p[("trace",)]
            m = (tt >= lo) & (tt <= hi)
            ax.plot(tt[m], vv[m], lw=1.5 if label_segments else 0.8, color=col, alpha=0.95)
        ax.set_xlim(lo, hi)

    ax = fig.add_subplot(gs[0, 0:6])
    draw_raw(ax, t0all, t1all)
    ax.set_ylabel("dB", color=INK2, fontsize=9)
    ax.set_xlabel("seconds into the sweep", color=INK2, fontsize=9)
    ax.set_title("Raw traces, whole run \u2014 shaded = measurement windows",
                 loc="left", color=INK, fontsize=11, fontweight="bold", pad=6)
    for r, tag in ((rows[0], "baseline\nstart"), (rows[-1], "baseline\nend")):
        ax.annotate(tag, ((float(r["t_start_s"]) + float(r["t_end_s"])) / 2, 0.02),
                    xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=8, color=INK, fontweight="bold")

    # --- zooms: where it helps, where it does not --------------------------
    def window(axis, speeds_wanted):
        sel = [r for r in sweeps if r["axis"] == axis
               and float(r["speed_mms"]) in speeds_wanted]
        return float(sel[0]["t_start_s"]) - 3, float(sel[-1]["t_end_s"]) + 3

    zooms = [("Y 30-50 mm/s \u2014 the whole effect", window("Y", {30, 40, 50})),
             ("X 30-50 mm/s \u2014 nothing to gain", window("X", {30, 40, 50})),
             ("Y 260-300 mm/s \u2014 bearings, not singing", window("Y", {260, 280, 300}))]
    for i, (title, (lo, hi)) in enumerate(zooms):
        ax = fig.add_subplot(gs[1, i * 2:(i + 1) * 2])
        draw_raw(ax, lo, hi, label_segments=True)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5, fontweight="bold", pad=5)
        ax.set_xlabel("s", color=INK2, fontsize=8.5)
        if i == 0:
            ax.set_ylabel("dB", color=INK2, fontsize=9)

    fig.suptitle(args.title, x=0.055, ha="left", y=0.992, va="top", fontsize=17,
                 fontweight="bold", color=INK)
    fig.text(0.055, 0.9685, args.subtitle, fontsize=9.2, color=INK2, va="top", wrap=True)
    fig.savefig(args.out, dpi=150, facecolor=SURF)

    csv_path = os.path.splitext(args.out)[0] + "_levels.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "speed_mms"] + [f"{l} (dB)" for l in labels]
                   + ["drive_mode_dB", "correction_dB", "net_dB"])
        for axis in ("X", "Y"):
            for s_ in speeds:
                o, u, c = (p[(axis, s_)] for p in P)
                w.writerow([axis, f"{s_:.0f}", f"{o:.2f}", f"{u:.2f}", f"{c:.2f}",
                            f"{u-o:+.2f}", f"{c-u:+.2f}", f"{c-o:+.2f}"])
    print(f"wrote {args.out}")
    print(f"wrote {csv_path}")
    for lbl, p in zip(labels, P):
        fl = np.array([f[1] for f in p[("floor",)]])
        d = [abs(p[k] - p[("meta",)]["raw"][k]) for k in p if k[0] in ("X", "Y")]
        print(f"  {lbl:22s} floor {np.median(fl):.1f} dB (p10-p90 "
              f"{np.percentile(fl,10):.1f}-{np.percentile(fl,90):.1f}), "
              f"baselines {p[('base','start')]:.1f}/{p[('base','end')]:.1f}, "
              f"robust-vs-plain max {max(d):.2f} dB, {p[('meta',)]['rejected']} samples dropped")


if __name__ == "__main__":
    main()
