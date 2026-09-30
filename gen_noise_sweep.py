#!/usr/bin/env python3
"""
gen_noise_sweep.py - acoustic speed-sweep G-code generator for the Prusa MK4S.

Emits a cold G-code file (no heaters, no extrusion, fan off) that runs one axis
at a time across its full travel at a series of constant speeds, holding each
speed long enough for a sound level meter to settle, with a beep marker and a
short silence between segments.  A companion CSV lists every segment with its
start/end time and an empty dB column.

The speed is shown on the printer's screen while it moves: every move is
commanded at a fixed 100 mm/s base feedrate and the speed factor (M220) carries
the real number, so the screen's speed field reads mm/s directly - 240% == 240
mm/s.  M117 is not used because the MK4 does not display it.

  ./gen_noise_sweep.py                          # X+Y, 30-400 mm/s
  ./gen_noise_sweep.py --axes Y --speeds 30:120:5
  ./gen_noise_sweep.py --accel 1250 --out stock_accel.gcode
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import sys

SQRT_HALF = math.sqrt(0.5)

# Prusa MK4S usable volume (mm): bed 250 x 210, Z 220.
AXIS_LIMITS = {"X": (0.0, 250.0), "Y": (0.0, 210.0), "Z": (0.0, 220.0)}

# Buddy firmware defaults, Configuration_MK4.h:
#   DEFAULT_MAX_FEEDRATE     { 400, 400, 40, 50 }
#   DEFAULT_MAX_ACCELERATION { 1250, 1250, 200, 1500 }
#   DEFAULT_AXIS_STEPS_PER_UNIT  X/Y 100, Z 400   -> 32 mm/rev, 8 mm/rev
FW_MAX_FEEDRATE = {"X": 400.0, "Y": 400.0, "Z": 40.0}

# Configuration_MK4.h also carries a second, harder set of limits that clamp
# whatever M203/M201 ask for.  DEFAULT_MAX_FEEDRATE {400,400,..} is only the
# value the planner starts at - HWLIMIT is the ceiling it is held to, and for
# X/Y in normal mode that is 300 mm/s, which is why M203 X400 changes nothing:
#   #define HWLIMIT_NORMAL_MAX_FEEDRATE      { 300, 300, 40, 100 }
#   #define HWLIMIT_STEALTH_MAX_FEEDRATE     { 160, 160, 40, 100 }
#   #define HWLIMIT_NORMAL_MAX_ACCELERATION  { 7000, 7000, 750, 6000 }
#   #define HWLIMIT_STEALTH_MAX_ACCELERATION { 2500, 2500, 200, 2500 }
#   #define HWLIMIT_NORMAL_JERK  {10, 10, 2, 10}
#   #define HWLIMIT_STEALTH_JERK {8, 8, 2, 10}
# The limit is per axis, so a 45 deg diagonal carrying 1/sqrt(2) per motor runs
# a vector 424 mm/s while either axis alone stops at 300.
HWLIMIT = {
    "normal":  {"feedrate": {"X": 300.0, "Y": 300.0, "Z": 40.0},
                "accel": 7000.0, "accel_z": 750.0, "jerk": 10.0},
    "stealth": {"feedrate": {"X": 160.0, "Y": 160.0, "Z": 40.0},
                "accel": 2500.0, "accel_z": 200.0, "jerk": 8.0},
}
FW_MAX_ACCEL = {"X": 1250.0, "Y": 1250.0, "Z": 200.0}
MM_PER_REV = {"X": 32.0, "Y": 32.0, "Z": 8.0}

# Configuration_MK4_adv.h carries two motor variants for X/Y, both landing on
# 100 steps/mm, and X/Y run spreadCycle either way (only STEALTHCHOP_Z is on):
#   X_400_STEP_CURRENT 550   Y_400_STEP_CURRENT 700   microsteps 8
#   X_200_STEP_CURRENT 300   Y_200_STEP_CURRENT 370   microsteps 16
#   INTERPOLATE true, HOLD_MULTIPLIER {1,1,1,1}, CHOPPER_PRUSAMK3_24V
# Which one you have doubles or halves the full-step rate, so it decides what
# frequency the singing should be at: 12.5*v Hz for 400-step, 6.25*v for 200.
MOTOR_VARIANTS = {
    400: {"microsteps": 8,  "current": {"X": 550, "Y": 700}},
    200: {"microsteps": 16, "current": {"X": 300, "Y": 370}},
}
FULL_STEPS_PER_REV = 200.0

# Every measured move is commanded at this feedrate; M220 supplies the rest.
BASE_SPEED = 100.0

DIRECTIONS = {
    "X":  (1.0, 0.0, 0.0),
    "Y":  (0.0, 1.0, 0.0),
    "Z":  (0.0, 0.0, 1.0),
    "XY": (SQRT_HALF,  SQRT_HALF, 0.0),   # 45 deg diagonal, both motors turning
    "YX": (SQRT_HALF, -SQRT_HALF, 0.0),   # the other diagonal
}

DEFAULT_SPEEDS = "30:100:10,120:300:20"
DEFAULT_Z_SPEEDS = "5:40:5"
DEFAULT_PROBE_SPEEDS = "200:400:20"
DEFAULT_CHOPPER_SPEEDS = "30,60,100,150"

CSV_FIELDS = [
    "index", "kind", "axis", "clock", "t_start_s", "t_end_s", "duration_s",
    "speed_mms", "axis_speed_mms", "elec_hz", "full_step_hz", "motor_rev_s",
    "accel_mms2", "stroke_mm", "strokes", "cruise_pct", "db_a", "notes",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def parse_speeds(spec: str) -> list[float]:
    """Parse "30,60,120" / "30:100:10" / a mix of both into a sorted list."""
    out: list[float] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            parts = tok.split(":")
            if len(parts) != 3:
                raise argparse.ArgumentTypeError(f"bad range {tok!r}, want start:stop:step")
            start, stop, step = (float(p) for p in parts)
            if step <= 0:
                raise argparse.ArgumentTypeError(f"bad step in {tok!r}")
            v = start
            while v <= stop + 1e-9:
                out.append(round(v, 4))
                v += step
        else:
            out.append(float(tok))
    if not out:
        raise argparse.ArgumentTypeError("empty speed list")
    if min(out) <= 0:
        raise argparse.ArgumentTypeError("speeds must be > 0")
    return sorted(set(out))


def move_time(dist: float, v: float, accel: float) -> float:
    """Trapezoid/triangle time for one move.  Junction deviation is ignored, so
    this runs a few percent long - good enough for scheduling."""
    if dist <= 0 or v <= 0:
        return 0.0
    if v * v / accel <= dist:            # reaches cruise speed
        return dist / v + v / accel
    return 2.0 * math.sqrt(dist / accel)  # triangular, never gets there


def plan_segment(v: float, accel: float, length: float, measure_s: float,
                 runway_margin: float = 1.25):
    """How many full-length strokes are needed to hold `v` for measure_s.

    The stroke is always the full axis travel, so consecutive segments simply
    carry on from whichever end the previous one finished at - no repositioning
    jogs in between.  Returns ((strokes, cruise_seconds), runway_needed).
    """
    accel_dist = v * v / (2.0 * accel)
    needed = 2.0 * accel_dist * runway_margin
    if needed > length:
        return None, needed
    cruise = (length - 2.0 * accel_dist) / v      # steady-speed time per stroke
    strokes = max(1, math.ceil(measure_s / cruise))
    return (strokes, cruise * strokes), needed


def geometry(axis: str, z_park: float, margin: float) -> tuple[list[float], float]:
    """Bed-centred midpoint and the full stroke length available along `axis`."""
    d = DIRECTIONS[axis]
    center = [
        (AXIS_LIMITS["X"][0] + AXIS_LIMITS["X"][1]) / 2.0,
        (AXIS_LIMITS["Y"][0] + AXIS_LIMITS["Y"][1]) / 2.0,
        z_park,
    ]
    half = None
    for i, name in enumerate("XYZ"):
        comp = d[i]
        if abs(comp) < 1e-9:
            continue
        lo, hi = AXIS_LIMITS[name]
        lo, hi = lo + margin, hi - margin
        if name == "Z":
            lo = max(lo, 10.0)
        room = min(center[i] - lo, hi - center[i])
        if room <= 0:
            raise SystemExit(f"{axis}: no room around {name}={center[i]:.1f} (usable {lo:.1f}-{hi:.1f})")
        h = room / abs(comp)
        half = h if half is None else min(half, h)
    return center, 2.0 * half


def point(center, d, s: float) -> list[float]:
    """Position `s` mm from the midpoint along the axis direction."""
    return [c + dd * s for c, dd in zip(center, d)]


def axis_accel(args, axis: str) -> float:
    """Firmware clamps acceleration per axis; Z gets far less than X/Y."""
    hw = HWLIMIT["stealth" if args.stealth else "normal"]
    return min(args.accel, hw["accel_z"] if axis == "Z" else hw["accel"])


def axis_max_feedrate(axis: str, per_axis: dict) -> float:
    """Highest vector speed for which no single motor exceeds its own ceiling.

    Divide, do not just take the minimum: on a 45 deg diagonal each axis only
    carries 1/sqrt(2) of the vector speed, so the move can go sqrt(2) faster
    than either axis is allowed to run on its own.
    """
    return min(per_axis[a] / abs(c) for a, c in zip("XYZ", DIRECTIONS[axis]) if abs(c) > 1e-9)


def clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60:d}:{s % 60:02d}"


# --------------------------------------------------------------------------
# program builder
# --------------------------------------------------------------------------

class Program:
    def __init__(self, accel: float):
        self.lines: list[str] = []
        self.rows: list[dict] = []
        self.t = 0.0
        self.pos = [0.0, 0.0, 0.0]
        self.accel = accel
        self.factor = 100.0          # current M220 speed factor

    def raw(self, line: str = "") -> None:
        self.lines.append(line)

    def comment(self, text: str) -> None:
        self.lines.append("; " + text)

    def cmd(self, line: str, dt: float = 0.0) -> None:
        self.lines.append(line)
        self.t += dt

    def dwell(self, seconds: float, why: str = "") -> None:
        if seconds <= 0:
            return
        ms = int(round(seconds * 1000.0))
        self.cmd(f"G4 P{ms}" + (f"  ; {why}" if why else ""), ms / 1000.0)

    def beep(self, freq: int, ms: int) -> None:
        self.cmd(f"M300 S{freq} P{ms}", ms / 1000.0)

    def set_factor(self, percent: float, why: str = "") -> None:
        """M220 doubles as the on-screen speed readout: 240% == 240 mm/s."""
        if abs(percent - self.factor) < 1e-6:
            return
        self.cmd(f"M220 S{percent:.0f}" + (f"  ; {why}" if why else ""))
        self.factor = percent

    def move(self, target, speed: float, why: str = "", commanded: float | None = None,
             blend: bool = False) -> float:
        """`speed` is what the axis actually does; `commanded` is what goes in
        the F word when M220 is carrying part of the scaling.  `blend` means
        this move is chained to the next one at speed, so it costs no
        accel/decel time - used for the glide's micro-moves."""
        delta = [t - p for t, p in zip(target, self.pos)]
        dist = math.sqrt(sum(x * x for x in delta))
        words = [f"{ax}{tgt:.3f}" for ax, tgt, dl in zip("XYZ", target, delta) if abs(dl) > 1e-6]
        if not words:
            return 0.0
        dt = dist / speed if blend else move_time(dist, speed, self.accel)
        f = (commanded if commanded is not None else speed) * 60.0
        self.cmd("G1 " + " ".join(words) + f" F{f:.0f}" + (f"  ; {why}" if why else ""), dt)
        self.pos = list(target)
        return dt

    def record(self, **row) -> None:
        row["index"] = len(self.rows) + 1
        row["clock"] = clock(row["t_start_s"])
        dur = row["t_end_s"] - row["t_start_s"]
        row["duration_s"] = round(dur, 2)
        cruise = row.pop("cruise_s", None)
        row["cruise_pct"] = round(100.0 * cruise / dur) if cruise and dur > 0 else ""
        row["t_start_s"] = round(row["t_start_s"], 2)
        row["t_end_s"] = round(row["t_end_s"], 2)
        fs = row.get("full_step_hz", "")
        row["elec_hz"] = round(fs / 4.0, 2) if isinstance(fs, (int, float)) else ""
        row.setdefault("db_a", "")
        row.setdefault("notes", "")
        self.rows.append(row)


# --------------------------------------------------------------------------
# emitters
# --------------------------------------------------------------------------

def emit_baseline(prog: Program, args, tag: str) -> None:
    """Motors energised, nothing moving - the level every segment is measured against.

    One at each end of the run, so the pair is a check as well as a reference: if
    the two disagree, the room or the meter moved during the run and the whole
    comparison is on sand.  Motors stay energised because that is the state the
    2 s gaps between segments are in, which makes all three directly comparable.
    """
    prog.raw()
    prog.comment(f"--- BASELINE ({tag}): motors energised, nothing moving ---")
    prog.cmd("M400")
    prog.dwell(args.pause, "silence")
    prog.cmd(f"M117 baseline {tag}")
    prog.beep(args.beep_hz, args.beep_ms)
    t0 = prog.t
    prog.dwell(args.baseline, "hold still")
    prog.record(kind="baseline", axis="-", speed_mms="", axis_speed_mms="", full_step_hz="",
                motor_rev_s="", accel_mms2="", stroke_mm="", strokes=0,
                t_start_s=t0, t_end_s=prog.t, notes=tag)


def emit_stroke(prog: Program, args, target, v: float, label: str, commanded, length: float) -> None:
    """One end-to-end stroke, with the M117 label refreshed while it runs.

    The MK4 shows an M117 message only briefly before the status line takes the
    screen back, so the stroke is split into collinear chunks and the label is
    re-sent between them.  Collinear chunks at one feedrate are blended by the
    planner, so the motion is identical to a single G1 - no extra stops, no
    change in sound.
    """
    if args.label_refresh <= 0:
        prog.move(target, v, commanded=commanded)
        return
    chunks = max(1, math.ceil(length / max(v * args.label_refresh, 5.0)))
    if chunks == 1:
        prog.move(target, v, commanded=commanded)
        return
    start = list(prog.pos)
    for i in range(1, chunks + 1):
        if i > 1:
            prog.cmd(f"M117 {label}")
        prog.move([a + (b - a) * i / chunks for a, b in zip(start, target)], v,
                  commanded=commanded, blend=True)
    prog.t += v / prog.accel          # the one stop-start the blended chunks skipped


def emit_glide(prog: Program, args, axis: str, v_lo: float, v_hi: float) -> None:
    """Continuous speed ramp v_lo -> v_hi -> v_lo, geometric in time.

    Geometric means the full-step frequency - the note you actually hear - rises
    at a constant number of octaves per second, so the pitch glides linearly and
    a resonance shows up as a bump at a fixed point in the sweep instead of a
    smeared band.  The ramp is chopped into micro-moves; the planner blends
    them, so the machine runs a smooth ramp rather than a staircase.

    Speeds go in the F word here, not M220.  That makes the glide an independent
    check: if the glide obviously sweeps but the stepped segments all sound
    alike, the firmware is ignoring M220 on travel moves.
    """
    ratio = v_hi / v_lo
    if ratio <= 1.0 + 1e-9:
        prog.comment(f"glide skipped on {axis}: needs a speed range")
        return

    center, length = geometry(axis, args.z_park, args.margin)
    d = DIRECTIONS[axis]
    lead = next(name for i, name in enumerate("XYZ") if abs(d[i]) > 1e-9)
    comp = max(abs(c) for c in d)
    mm_rev = MM_PER_REV[lead]
    prog.accel = axis_accel(args, axis)
    hz = lambda v: v * comp / mm_rev * args.motor_steps

    half = length / 2.0
    up = args.glide / 2.0
    octaves = math.log2(ratio)

    prog.raw()
    prog.comment(f"-- {axis} glide: {v_lo:g} -> {v_hi:g} -> {v_lo:g} mm/s over {args.glide:g} s, "
                 f"{octaves:.2f} octaves each way ({octaves / up:.2f} oct/s) --")
    prog.comment(f"   full-step {hz(v_lo):.0f} -> {hz(v_hi):.0f} Hz, "
                 f"micro-move every {args.glide_step * 1000:.0f} ms "
                 f"({100 * (ratio ** (args.glide_step / up) - 1):.1f}% speed per step)")
    prog.set_factor(100.0, "100% while repositioning")
    prog.move(point(center, d, -half), args.travel_speed, f"to {axis} end")
    prog.cmd("M400")
    prog.dwell(args.pause, "silence before the glide")
    prog.cmd(f"M117 {axis} glide {v_lo:.0f} mm/s")
    prog.beep(args.beep_hz, args.beep_ms)

    t0 = prog.t
    s, direction, t, reversals, note_at = -half, 1, 0.0, 0, 1.0
    while t < args.glide - 1e-9:
        frac = t / up if t <= up else (args.glide - t) / up
        v = v_lo * ratio ** max(0.0, min(1.0, frac))
        remaining = (half - s) if direction > 0 else (s + half)
        if remaining <= 1e-6:                    # bounce off the end of the axis
            direction = -direction
            remaining = length
            reversals += 1
            prog.t += v / prog.accel             # decel to 0 and back up again
        step = min(v * args.glide_step, remaining, v * (args.glide - t))
        if step <= 1e-9:
            break
        s += direction * step
        prog.move(point(center, d, s), v, blend=True)
        t += step / v
        if t >= note_at:
            prog.comment(f"   t={t:.1f}s  {v:.0f} mm/s  {hz(v):.0f} Hz")
            prog.cmd(f"M117 {axis} glide {v:.0f} mm/s")
            note_at += 1.0

    prog.record(kind="glide", axis=axis, speed_mms="", axis_speed_mms="",
                full_step_hz=f"{hz(v_lo):.0f}-{hz(v_hi):.0f}", motor_rev_s="",
                accel_mms2=round(prog.accel), stroke_mm=round(length, 1), strokes=reversals,
                t_start_s=t0, t_end_s=prog.t,
                notes=f"log glide {v_lo:g}-{v_hi:g}-{v_lo:g} mm/s, {reversals} reversals")


def emit_sweep(prog: Program, args, axis: str, speeds: list[float], pass_no: int) -> None:
    center, length = geometry(axis, args.z_park, args.margin)
    d = DIRECTIONS[axis]
    lead = next(name for i, name in enumerate("XYZ") if abs(d[i]) > 1e-9)
    comp = max(abs(c) for c in d)
    mm_rev = MM_PER_REV[lead]
    prog.accel = axis_accel(args, axis)
    ends = [[c + dd * s * length / 2.0 for c, dd in zip(center, d)] for s in (-1.0, 1.0)]

    prog.raw()
    prog.comment("=" * 68)
    prog.comment(f"{axis} sweep, pass {pass_no}: {len(speeds)} speeds, "
                 f"full stroke {length:.1f} mm, accel {prog.accel:.0f} mm/s^2")
    prog.comment("=" * 68)

    # One jog to an end of the axis; from here on every stroke is end-to-end and
    # the next speed starts from wherever the last one stopped.
    #
    # Known quirk, left as-is so the schedule keeps matching the 30 Sep
    # recordings: there is no pause before this jog, so the previous axis's last
    # segment runs straight into it with no audible gap.  The analysis drops the
    # axis-transition gap from its floor statistics for that reason.  Adding
    # M400 + a dwell here would fix it, at the cost of shifting every later
    # segment by 2 s and invalidating existing recordings.
    prog.set_factor(100.0, "100% while repositioning")
    prog.move(ends[0], args.travel_speed, f"to {axis} end")
    at = 0

    for v in speeds:
        plan, needed = plan_segment(v, prog.accel, length, args.measure)
        if plan is None:
            msg = (f"skip {axis} @ {v:g} mm/s: needs {needed:.0f} mm of runway at "
                   f"{prog.accel:.0f} mm/s^2, stroke is {length:.0f} mm")
            prog.comment(msg)
            print("  ! " + msg, file=sys.stderr)
            continue
        strokes, cruise_s = plan
        axis_speed = v * comp

        prog.raw()
        prog.comment(f"-- {axis} @ {v:g} mm/s | {strokes} x {length:.1f} mm "
                     f"| full-step {axis_speed / mm_rev * args.motor_steps:.0f} Hz --")
        # M220/M117 go AFTER the pause on purpose.  The parser runs ahead of the
        # motion queue, so anything placed before the dwell is executed - and
        # shown on screen - while the previous segment is still moving.  M400
        # drains the queue, the dwell blocks the gcode loop, and only then does
        # the display change, so the screen matches what is actually running.
        prog.cmd("M400")
        prog.dwell(args.pause, "silence between segments")
        if args.speed_display == "m220":
            prog.set_factor(v, f"screen reads {v:g}% = {v:g} mm/s")
            commanded = BASE_SPEED
        else:
            prog.set_factor(100.0)
            commanded = None
        label = f"{axis} {v:g} mm/s"
        prog.cmd(f"M117 {label}")
        prog.beep(args.beep_hz, args.beep_ms)
        t0 = prog.t
        for _ in range(strokes):
            at ^= 1
            emit_stroke(prog, args, ends[at], v, label, commanded, length)
        prog.record(kind="sweep", axis=axis, speed_mms=round(v, 3),
                    axis_speed_mms=round(axis_speed, 3),
                    full_step_hz=round(axis_speed / mm_rev * args.motor_steps, 1),
                    motor_rev_s=round(axis_speed / mm_rev, 3),
                    accel_mms2=round(prog.accel), stroke_mm=round(length, 1), strokes=strokes,
                    cruise_s=cruise_s, t_start_s=t0, t_end_s=prog.t, notes=f"pass {pass_no}")


def emit_probe(prog: Program, args, axis: str, speeds: list[float]) -> None:
    """Fixed-stroke timing probe: beep, run exactly N full strokes, beep.

    Every speed covers the same distance, so the beep-to-beep interval off your
    recording is a direct measurement of what the axis really did.  If the
    intervals stop shrinking above some commanded speed, that is a real clamp;
    if they keep shrinking, the machine is reaching the speed and what changed
    was only the character of the sound.  Compare the intervals to each other,
    not to the absolute predictions - a fixed offset cancels out.
    """
    center, length = geometry(axis, args.z_park, args.margin)
    d = DIRECTIONS[axis]
    lead = next(name for i, name in enumerate("XYZ") if abs(d[i]) > 1e-9)
    comp = max(abs(c) for c in d)
    prog.accel = axis_accel(args, axis)
    ends = [point(center, d, sgn * length / 2.0) for sgn in (-1.0, 1.0)]
    n = args.probe_strokes

    prog.raw()
    prog.comment("=" * 68)
    prog.comment(f"{axis} timing probe: {n} x {length:.1f} mm = {n * length:.0f} mm per speed")
    prog.comment("high beep starts the run, low beep ends it; time between them is the answer")
    prog.comment("=" * 68)
    prog.set_factor(100.0, "100% while repositioning")
    prog.move(ends[0], args.travel_speed, f"to {axis} end")
    at = 0

    for v in speeds:
        predicted = n * move_time(length, v, prog.accel)
        prog.raw()
        prog.comment(f"-- {axis} probe @ {v:g} mm/s | predicted beep-to-beep {predicted:.2f} s --")
        prog.cmd("M400")
        prog.dwell(args.pause, "silence between segments")
        if args.speed_display == "m220":
            prog.set_factor(v, f"screen reads {v:g}% = {v:g} mm/s")
            commanded = BASE_SPEED
        else:
            prog.set_factor(100.0)
            commanded = None
        label = f"{axis} {v:g} mm/s"
        prog.cmd(f"M117 {label}")
        t0 = prog.t
        prog.beep(args.beep_hz, args.beep_ms)
        for _ in range(n):
            at ^= 1
            emit_stroke(prog, args, ends[at], v, label, commanded, length)
        t1 = prog.t
        prog.beep(max(200, args.beep_hz // 2), args.beep_ms)
        prog.record(kind="probe", axis=axis, speed_mms=round(v, 3),
                    axis_speed_mms=round(v * comp, 3),
                    full_step_hz=round(v * comp / MM_PER_REV[lead] * args.motor_steps, 1),
                    motor_rev_s=round(v * comp / MM_PER_REV[lead], 3),
                    accel_mms2=round(prog.accel), stroke_mm=round(length, 1), strokes=n,
                    t_start_s=t0, t_end_s=t1,
                    notes=f"{n * length:.0f} mm, predicted {predicted:.2f} s")


def emit_mode_ab(prog: Program, args, axis: str, speeds: list[float],
                 modes: list[tuple[str, str]], title: str, restore: str) -> None:
    """Run each speed once per mode, back to back, so the ear can compare.

    Configuration_MK4_adv.h defines STEALTHCHOP_Z but no STEALTHCHOP_XY, so X/Y
    boot in spreadCycle - the loud, tonal mode - and Prusa's Stealth power mode
    does not change that, it only lowers feedrate/accel/current.  M569 sets the
    mode per axis at runtime, which is worth a try even though the axis was not
    compiled with stealthChop as its default.

    `modes` is a list of (gcode, short name); the first is the reference.  The
    marker beep is low for the first mode and high for the rest, so the pairs
    are findable in a recording without counting segments.
    """
    center, length = geometry(axis, args.z_park, args.margin)
    d = DIRECTIONS[axis]
    prog.accel = axis_accel(args, axis)
    ends = [point(center, d, sgn * length / 2.0) for sgn in (-1.0, 1.0)]

    prog.raw()
    prog.comment("=" * 68)
    prog.comment(f"{axis} {title}")
    prog.comment("=" * 68)
    prog.set_factor(100.0, "100% while repositioning")
    prog.move(ends[0], args.travel_speed, f"to {axis} end")
    at = 0

    for v in speeds:
        plan, needed = plan_segment(v, prog.accel, length, args.measure)
        if plan is None:
            prog.comment(f"skip {axis} @ {v:g} mm/s: needs {needed:.0f} mm")
            continue
        strokes, cruise_s = plan
        for mode, (setup, name) in enumerate(modes):
            prog.raw()
            prog.comment(f"-- {axis} @ {v:g} mm/s, {name} --")
            prog.cmd("M400")
            prog.cmd(setup)
            prog.dwell(args.pause, "let the driver settle")
            if args.speed_display == "m220":
                prog.set_factor(v)
                commanded = BASE_SPEED
            else:
                prog.set_factor(100.0)
                commanded = None
            label = f"{axis} {v:g} {name}"
            prog.cmd(f"M117 {label}")
            prog.beep(args.beep_hz if mode else args.beep_hz // 2, args.beep_ms)
            t0 = prog.t
            for _ in range(strokes):
                at ^= 1
                emit_stroke(prog, args, ends[at], v, label, commanded, length)
            prog.record(kind="modeab", axis=axis, speed_mms=round(v, 3),
                        axis_speed_mms=round(v, 3),
                        full_step_hz=round(v / MM_PER_REV[axis] * args.motor_steps, 1),
                        motor_rev_s=round(v / MM_PER_REV[axis], 3), accel_mms2=round(prog.accel),
                        stroke_mm=round(length, 1), strokes=strokes, cruise_s=cruise_s,
                        t_start_s=t0, t_end_s=prog.t, notes=name)
    prog.cmd(restore, 0.0)
    prog.comment(f"{axis} restored to its default mode")


def currents_for(args, axis: str) -> tuple[list[int], int]:
    """Current ladder for one axis, plus that axis's stock value.

    Stock differs per axis and per motor variant (200-step: X 300 / Y 370 mA;
    400-step: X 550 / Y 700), so a single hard-coded list is wrong for one of
    them - and badly wrong across variants, where the same number can be double
    the rated current.
    """
    stock = MOTOR_VARIANTS[args.motor_steps]["current"][axis]
    if args.current_sweep.strip().lower() == "auto":
        return [int(round(stock * f)) for f in (0.6, 0.75, 0.9, 1.0)], stock
    return [int(c) for c in args.current_sweep.split(",") if c.strip()], stock


def emit_current_sweep(prog: Program, args, axis: str, currents: list[int]) -> None:
    """Optional: hold one speed and step the motor RMS current instead.
    Nothing is saved to EEPROM - a power cycle restores the stock values."""
    center, length = geometry(axis, args.z_park, args.margin)
    d = DIRECTIONS[axis]
    prog.accel = axis_accel(args, axis)
    v = args.current_speed
    plan, needed = plan_segment(v, prog.accel, length, args.measure)
    if plan is None:
        print(f"  ! current sweep on {axis} skipped: {v:g} mm/s needs {needed:.0f} mm", file=sys.stderr)
        return
    strokes, cruise_s = plan
    ends = [[c + dd * s * length / 2.0 for c, dd in zip(center, d)] for s in (-1.0, 1.0)]

    prog.raw()
    prog.comment("=" * 68)
    prog.comment(f"{axis} motor-current sweep @ {v:g} mm/s (not saved; reboot restores defaults)")
    prog.comment("=" * 68)
    prog.set_factor(100.0)
    prog.move(ends[0], args.travel_speed, f"to {axis} end")
    at = 0
    for ma in currents:
        prog.raw()
        prog.comment(f"-- {axis} @ {ma} mA --")
        prog.cmd(f"M906 {axis}{ma}")
        if args.speed_display == "m220":
            prog.set_factor(v)
            commanded = BASE_SPEED
        else:
            commanded = None
        prog.beep(args.beep_hz, args.beep_ms)
        prog.dwell(args.pause, "settle")
        t0 = prog.t
        for _ in range(strokes):
            at ^= 1
            prog.move(ends[at], v, commanded=commanded)
        prog.record(kind="current", axis=axis, speed_mms=round(v, 3), axis_speed_mms=round(v, 3),
                    full_step_hz=round(v / MM_PER_REV[axis] * args.motor_steps, 1),
                    motor_rev_s=round(v / MM_PER_REV[axis], 3), accel_mms2=round(prog.accel),
                    stroke_mm=round(length, 1), strokes=strokes, cruise_s=cruise_s,
                    t_start_s=t0, t_end_s=prog.t, notes=f"{ma} mA")
    prog.comment("motor current left at the last swept value - power cycle to restore")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(args) -> Program:
    prog = Program(args.accel)
    axes = [a.strip().upper() for a in args.axes.split(",") if a.strip()]
    for a in axes:
        if a not in DIRECTIONS:
            raise SystemExit(f"unknown axis {a!r}; pick from {', '.join(DIRECTIONS)}")

    prog.comment("cold noise sweep - no heaters, no extrusion, nothing gets printed")
    if not args.no_is_check:
        prog.comment("the same print checks PrusaSlicer 2.9.6 emits for this printer. Without the")
        prog.comment("feature check the printer warns the file is not sliced for input shaping and")
        prog.comment("runs it down a legacy compatibility path - different limits, different noise.")
        prog.cmd('M862.3 P "MK4S"')
        prog.cmd("M862.5 P2")
        prog.cmd('M862.6 P"Input shaper"')
    prog.cmd("M104 S0")
    prog.cmd("M140 S0")
    prog.cmd("M107")
    prog.cmd("G21")
    prog.cmd("G90")
    prog.cmd("M220 S100")

    prog.raw()
    prog.comment("--- home and park (noisy, not measured) ---")
    prog.cmd("M17", 0.5)
    if args.home == "all":
        prog.cmd("G28", 45.0)
        prog.pos = [0.0, 0.0, 0.0]
        prog.move([125.0, 105.0, args.z_park], 30.0, "park")
    else:
        prog.cmd("G28 X Y", 25.0)
        prog.pos = [0.0, 0.0, args.z_park]
        prog.comment("Z NOT homed: raise Z by hand before starting this file")
        prog.move([125.0, 105.0, args.z_park], args.travel_speed, "park")

    if args.phase_stepping != "leave":
        on = 1 if args.phase_stepping == "on" else 0
        prog.raw()
        prog.comment(f"--- phase stepping {'ON' if on else 'OFF'} ---")
        prog.comment("set explicitly, after homing. On an MK4 the enable flag defaults to FALSE")
        prog.comment("(defaults.hpp: true only for XL/CoreOne/iX) and is declared ram_only, so it")
        prog.comment("reverts to OFF at every power cycle. PrusaSlicer never sends M970, so a print")
        prog.comment("runs without phase stepping unless something enables it after boot.")
        prog.comment("M972 saves the correction table but does NOT enable - separate states.")
        prog.cmd(f"M970 X{on} Y{on}", 0.5)
        prog.dwell(1.0, "let the drivers switch over")

    prog.raw()
    prog.comment("--- motion limits, held constant for the whole run ---")
    hw = HWLIMIT["stealth" if args.stealth else "normal"]
    prog.comment(f"planner starts at M201 X1250 Y1250; raising to {args.accel:.0f} "
                 f"(HWLIMIT_{'STEALTH' if args.stealth else 'NORMAL'}_MAX_ACCELERATION "
                 f"allows {hw['accel']:.0f})")
    prog.comment("PrusaSlicer emits M203 X300 Y300 for this printer too - the 400 in the")
    prog.comment("firmware header is a startup value the HWLIMIT never lets you reach.")
    prog.comment("NOTE: a PrusaSlicer print ENDS with M593 X T2 F0 / M593 Y T2 F0, which turns")
    prog.comment("input shaping off and leaves it off until reboot. Power-cycle the printer")
    prog.comment("before measuring, or this file may run with a different shaper state.")
    prog.comment(f"M203 is set to the hard limit, not the {FW_MAX_FEEDRATE['X']:.0f} startup value: "
                 f"HWLIMIT clamps X/Y to {hw['feedrate']['X']:.0f} mm/s and asking for more is a no-op")
    # Line for line what PrusaSlicer 2.9.6 emits for the MK4S, so the planner is in
    # the same state a real print puts it in.  The one deliberate difference is that
    # M204 stays put: a print varies P from 500 to 4000 and T between 250 and 4000
    # per feature, which would confound a measurement whose only variable is speed.
    z_accel = min(args.accel, hw["accel_z"]) if "Z" in axes else 200
    prog.cmd(f"M201 X{args.accel:.0f} Y{args.accel:.0f} Z{z_accel:.0f} E2500"
             "  ; maximum accelerations, mm/s^2")
    prog.cmd(f"M203 X{hw['feedrate']['X']:.0f} Y{hw['feedrate']['Y']:.0f} Z40 E100"
             "  ; maximum feedrates, mm/s")
    prog.cmd(f"M204 P{args.accel:.0f} R2500 T{args.accel:.0f}  ; acceleration, mm/s^2")
    prog.cmd("M205 X8.00 Y8.00 Z2.00 E10.00  ; jerk limits, mm/s")
    prog.cmd(f"M205 J{args.junction_deviation:.3f}  ; junction deviation, mm")
    prog.cmd("M205 S0 T0  ; minimum extruding and travel feed rate, mm/s")
    if args.disable_is:
        prog.comment("input shaper off for this run (delete these if the firmware rejects them)")
        prog.cmd("M593 X F0")
        prog.cmd("M593 Y F0")
    if args.fan > 0:
        prog.cmd(f"M106 S{round(args.fan * 255 / 100):d}", 3.0)
        prog.dwell(3.0, "let the fan spin up")

    if args.baseline > 0:
        emit_baseline(prog, args, "start")

    default = (DEFAULT_PROBE_SPEEDS if args.probe
               else DEFAULT_CHOPPER_SPEEDS if (args.chopper_ab or args.phase_ab)
               else DEFAULT_SPEEDS)
    speeds = parse_speeds(args.speeds if args.speeds is not None else default)
    z_speeds = parse_speeds(args.z_speeds)
    hw = HWLIMIT["stealth" if args.stealth else "normal"]
    per_axis = dict(hw["feedrate"])
    if args.axis_max_speed > 0:
        per_axis["X"] = per_axis["Y"] = min(args.axis_max_speed, hw["feedrate"]["X"])
    plan: dict[str, list[float]] = {}
    for axis in axes:
        todo = z_speeds if axis == "Z" else speeds
        # The probe exists to find the ceiling, so it is allowed to aim past it.
        cap = args.max_speed if args.probe else min(axis_max_feedrate(axis, per_axis), args.max_speed)
        keep = [v for v in todo if v <= cap + 1e-9]
        for v in todo:
            if v > cap + 1e-9:
                print(f"  ! dropped {axis} @ {v:g} mm/s: cap is {cap:.0f} mm/s", file=sys.stderr)
        if args.speed_display == "m220":
            rounded = [float(round(v)) for v in keep]
            if rounded != keep:
                print("  ! speeds rounded to whole mm/s (M220 takes an integer percent)",
                      file=sys.stderr)
            keep = sorted(set(rounded))
        plan[axis] = keep

    # All the continuous glides first, then the stepped runs: listening to the
    # axes back to back is what tells you whether they resonate at the same note.
    if args.glide > 0 and not args.probe and not args.chopper_ab and not args.phase_ab:
        for axis in axes:
            if plan[axis]:
                emit_glide(prog, args, axis, min(plan[axis]), max(plan[axis]))

    for p in range(1, args.repeat + 1):
        for axis in axes:
            if args.probe:
                emit_probe(prog, args, axis, plan[axis])
            elif args.chopper_ab or args.phase_ab:
                if axis not in ("X", "Y"):
                    continue
                if args.chopper_ab:
                    emit_mode_ab(prog, args, axis, plan[axis],
                                 [(f"M569 S0 {axis}", "spreadCycle"),
                                  (f"M569 S1 {axis}", "stealthChop")],
                                 "chopper A/B: spreadCycle vs stealthChop",
                                 f"M569 S0 {axis}")
                else:
                    emit_mode_ab(prog, args, axis, plan[axis],
                                 [(f"M970 {axis}0", "phase stepping off"),
                                  (f"M970 {axis}1", "phase stepping on")],
                                 "phase stepping A/B",
                                 f"M970 {axis}1")
            else:
                emit_sweep(prog, args, axis, plan[axis], p)

    if args.current_sweep:
        for axis in axes:
            if axis not in ("X", "Y"):
                continue
            currents, stock = currents_for(args, axis)
            for ma in currents:
                if ma > stock * 1.1:
                    raise SystemExit(
                        f"{axis} current {ma} mA is above the {args.motor_steps}-step stock value "
                        f"of {stock} mA - overcurrent risks overheating the motor, and raising "
                        f"current makes the singing worse, not better. Sweep downwards.")
                if ma < 100:
                    raise SystemExit(f"{axis} current {ma} mA is too low to hold position")
            emit_current_sweep(prog, args, axis, currents)

    if args.baseline > 0:
        emit_baseline(prog, args, "end")

    prog.raw()
    prog.comment("--- done: put the motion settings back ---")
    prog.comment("leaving these raised would apply to the next G28, and sensorless homing")
    prog.comment("detects a stall from back-EMF - a hard acceleration ramp can mimic one")
    prog.set_factor(100.0)
    prog.cmd(f"M201 X{FW_MAX_ACCEL['X']:.0f} Y{FW_MAX_ACCEL['Y']:.0f}")
    prog.cmd(f"M204 P{FW_MAX_ACCEL['X']:.0f} T{FW_MAX_ACCEL['X']:.0f} R{FW_MAX_ACCEL['X']:.0f}")
    prog.cmd(f"M203 X{FW_MAX_FEEDRATE['X']:.0f} Y{FW_MAX_FEEDRATE['Y']:.0f}")
    if args.current_sweep:
        stock = MOTOR_VARIANTS[args.motor_steps]["current"]
        prog.cmd(f"M906 X{stock['X']} Y{stock['Y']}", 0.0)
    prog.cmd("M107")
    prog.accel = args.accel
    prog.move([125.0, 105.0, min(args.z_park + 50.0, 200.0)], 30.0, "lift out of the way")
    prog.cmd("M84", 0.5)
    for _ in range(3):
        prog.beep(args.beep_hz, 120)
        prog.dwell(0.15)
    return prog


def header(args, prog: Program) -> list[str]:
    measured = sum(r["duration_s"] for r in prog.rows)
    lines = [
        "; Prusa MK4S acoustic speed sweep",
        f"; generated by: {shlex.join(sys.argv)}",
        ";",
        f"; segments        : {len(prog.rows)}",
        f"; est. runtime    : {clock(prog.t)} (mm:ss; junction deviation ignored, runs a touch short)",
        f"; measured time   : {clock(measured)} across all windows",
        f"; acceleration    : {args.accel:.0f} mm/s^2 (M201+M204; 1250 is only the startup "
        f"value, HWLIMIT allows {HWLIMIT['stealth' if args.stealth else 'normal']['accel']:.0f})",
        f"; junction dev.   : {args.junction_deviation:g} mm",
        f"; measure / pause : {args.measure:g} / {args.pause:g} s",
        f"; glide           : {args.glide:g} s per axis" if args.glide > 0 else "; glide: none",
        f"; feature check   : {'M862.6 Input shaper' if not args.no_is_check else 'OMITTED'}",
        f"; phase stepping  : {args.phase_stepping.upper()}"
        f"{' (M970 not sent - state inherited)' if args.phase_stepping == 'leave' else ' (M970)'}",
        f"; per-axis ceiling: {HWLIMIT['stealth' if args.stealth else 'normal']['feedrate']['X']:g}"
        f" mm/s, firmware HWLIMIT_{'STEALTH' if args.stealth else 'NORMAL'}_MAX_FEEDRATE",
        f"; Z parked at     : {args.z_park:g} mm",
        f"; X/Y motors      : {args.motor_steps}-step, "
        f"{MOTOR_VARIANTS[args.motor_steps]['microsteps']} microsteps, "
        f"stock current X{MOTOR_VARIANTS[args.motor_steps]['current']['X']} "
        f"Y{MOTOR_VARIANTS[args.motor_steps]['current']['Y']} mA",
        f"; full-step rate  : {args.motor_steps / 32.0:g} Hz per mm/s "
        f"(electrical {args.motor_steps / 128.0:g} Hz per mm/s)",
        ";",
        "; WHICH HARMONIC IS LOUDEST TELLS YOU WHAT IS WRONG.  Per segment the CSV",
        "; gives elec_hz; one electrical cycle is four full steps, so:",
        ";   4x elec_hz  = the full-step rate: cogging.  Always present, always normal.",
        ";   2x elec_hz  = phase amplitude imbalance - one coil pulling weaker than",
        ";                 the other.  Bad crimp, partly seated connector, damaged",
        ";                 winding, or a driver channel down.",
        ";   1x elec_hz  = DC offset in one phase.",
        "; Normal singing peaks at 4x.  A peak at 2x louder than the one at 4x is a",
        "; fault signature, and no amount of current or damping will fix it.",
        ";",
        "; Cold run: heaters stay off so the hotend fan never starts and the meter",
        "; only hears the motion system.  Each axis runs end to end at every speed,",
        "; so nothing repositions between segments.  One beep = the next window",
        "; starts in {:g} s; three beeps = end of file.".format(args.pause),
        ";",
    ]
    if args.speed_display == "m220":
        lines += [
            "; ON-SCREEN SPEED: every move is commanded F6000 (100 mm/s) and M220 carries",
            "; the real speed, so the printer's speed field reads mm/s directly - 240%",
            "; means 240 mm/s.  Sanity check on the first run: the segments must audibly",
            "; differ.  If they all sound the same the firmware is ignoring M220 on travel",
            "; moves - regenerate with --speed-display off and read the CSV instead.",
            ";",
        ]
    return lines


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a constant-speed sweep G-code for measuring Prusa MK4S axis noise.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", default="mk4s_noise_sweep.gcode", help="output .gcode path")
    p.add_argument("--axes", default="X,Y", help="comma list from X,Y,Z,XY,YX")
    p.add_argument("--speeds", default=None,
                   help=f"mm/s: list and/or start:stop:step (default {DEFAULT_SPEEDS}, "
                        f"or {DEFAULT_PROBE_SPEEDS} with --probe)")
    p.add_argument("--max-speed", type=float, default=400.0,
                   help="ceiling on the vector speed of a move")
    p.add_argument("--axis-max-speed", type=float, default=0.0,
                   help="per-motor ceiling; 0 = the firmware HWLIMIT (300 normal, 160 stealth)")
    p.add_argument("--phase-stepping", choices=("on", "off", "leave"), default="on",
                   help="set phase stepping explicitly for the run (M970). 'leave' inherits "
                        "whatever the printer was left in, which makes the run unreproducible")
    p.add_argument("--stealth", action="store_true",
                   help="model the printer's Stealth mode limits (160 mm/s, 2500 mm/s^2, jerk 8) "
                        "- you must also switch the printer into Stealth mode yourself")
    p.add_argument("--chopper-ab", action="store_true",
                   help="A/B each speed in spreadCycle then stealthChop via M569 (X/Y only); "
                        "not saved to EEPROM, and it switches back at the end")
    p.add_argument("--phase-ab", action="store_true",
                   help="A/B each speed with phase stepping off then on via M970 (X/Y only). "
                        "Works uncalibrated - M970 has no check for a correction table - so it "
                        "tests the drive mechanism itself, not the per-motor correction")
    p.add_argument("--probe", action="store_true",
                   help="timing-probe mode: equal distance at every speed, beep-bracketed")
    p.add_argument("--probe-strokes", type=int, default=8,
                   help="full strokes per probe speed")
    p.add_argument("--z-speeds", default=DEFAULT_Z_SPEEDS, help="mm/s used when sweeping Z")
    p.add_argument("--measure", type=float, default=5.0, help="target seconds at speed per segment")
    p.add_argument("--pause", type=float, default=2.0, help="silent seconds between segments")
    p.add_argument("--baseline", type=float, default=10.0,
                   help="seconds of no-movement baseline at each end of the run; 0 = none. "
                        "The two should agree - if they do not, conditions drifted")
    p.add_argument("--glide", type=float, default=30.0,
                   help="seconds for the continuous log speed glide at the head of each axis; 0 = none")
    p.add_argument("--glide-step", type=float, default=0.04,
                   help="seconds per micro-move in the glide (smaller = smoother, more blocks)")
    p.add_argument("--accel", type=float, default=4000.0,
                   help="mm/s^2 held constant for the run. 4000 is what PrusaSlicer profiles use; "
                        "the firmware hard limit is 7000, but hundreds of reversals at the limit "
                        "is harder use than any print")
    p.add_argument("--junction-deviation", type=float, default=0.01, help="mm (M205 J)")
    p.add_argument("--travel-speed", type=float, default=100.0, help="mm/s for un-measured jogs")
    p.add_argument("--motor-steps", type=int, choices=(400, 200), default=200,
                   help="full steps/rev of the X/Y motors: 400 (0.9 deg, 8 microsteps) or "
                        "200 (1.8 deg, 16 microsteps). Only affects the reported frequencies")
    p.add_argument("--z-park", type=float, default=50.0, help="Z height for the whole run")
    p.add_argument("--margin", type=float, default=5.0, help="mm kept clear of each axis limit")
    p.add_argument("--fan", type=float, default=0.0, help="part fan %% during the run (0 = off)")
    p.add_argument("--repeat", type=int, default=1, help="repeat the whole sweep N times")
    p.add_argument("--home", choices=("all", "xy"), default="all", help="G28 vs G28 X Y")
    p.add_argument("--speed-display", choices=("m220", "off"), default="m220",
                   help="m220: show mm/s on screen; off: put the real speed in F")
    p.add_argument("--label-refresh", type=float, default=1.0,
                   help="seconds between M117 refreshes during a move; 0 = label once per segment")
    p.add_argument("--no-is-check", action="store_true",
                   help="omit M862.6 P\"Input shaper\" (the firmware feature declaration)")
    p.add_argument("--beep-hz", type=int, default=1200, help="marker beep frequency")
    p.add_argument("--beep-ms", type=int, default=60, help="marker beep length")
    p.add_argument("--disable-is", action="store_true", help="emit M593 F0 to turn input shaping off")
    p.add_argument("--current-sweep", default="",
                   help="also sweep motor RMS current: 'auto' for 60/75/90/100%% of this axis's "
                        "stock value, or an explicit list. Never above stock")
    p.add_argument("--current-speed", type=float, default=60.0, help="mm/s used for the current sweep")
    args = p.parse_args()

    if args.measure <= 0 or args.repeat < 1:
        raise SystemExit("--measure must be > 0 and --repeat >= 1")
    hw_max = HWLIMIT["stealth" if args.stealth else "normal"]["accel"]
    args.accel = min(args.accel, hw_max) if args.stealth else args.accel
    hw_accel = HWLIMIT["stealth" if args.stealth else "normal"]["accel"]
    if args.accel > hw_accel:
        print(f"  ! --accel {args.accel:g} exceeds the firmware hard limit of {hw_accel:g} mm/s^2; "
              f"the printer will clamp it and the timings here will be optimistic", file=sys.stderr)

    prog = build(args)

    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(header(args, prog) + prog.lines) + "\n")

    thin = [r for r in prog.rows if r["cruise_pct"] != "" and r["cruise_pct"] < 50]
    if thin:
        worst = min(r["cruise_pct"] for r in thin)
        print(f"  ! {len(thin)} segment(s) spend under half the window at speed "
              f"(worst {worst}%) - they are mostly accel/decel and measure reversals, "
              f"not steady running.  Raise --accel or drop the top speeds.", file=sys.stderr)

    csv_path = os.path.splitext(out)[0] + "_schedule.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in prog.rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})

    print(f"gcode    : {out}")
    print(f"schedule : {csv_path}")
    print(f"segments : {len(prog.rows)}   est. runtime {clock(prog.t)}")
    print()
    print(f"{'#':>3}  {'start':>6}  {'what':<18} {'strokes':>8}  {'window':>7}  "
          f"{'at speed':>9}  {'full-step':>10}")
    for r in prog.rows:
        if r["kind"] == "ambient":
            what = "ambient"
        elif r["kind"] == "baseline":
            what = f"baseline {r['notes']}"
        elif r["kind"] == "probe":
            what = f"{r['axis']} {r['speed_mms']:g} mm/s"
        elif r["kind"] == "glide":
            what = f"{r['axis']} glide"
        else:
            what = f"{r['axis']} {r['speed_mms']:g} mm/s"
            if r["kind"] == "current":
                what += f" {r['notes']}"
        strokes = f"{r['strokes']}x{r['stroke_mm']:g}" if r["strokes"] else "-"
        cruise = f"{r['cruise_pct']}%" if r["cruise_pct"] != "" else "-"
        hz = r["full_step_hz"]
        fs = "-" if hz == "" else (f"{hz} Hz" if isinstance(hz, str) else f"{hz:g} Hz")
        print(f"{r['index']:>3}  {r['clock']:>6}  {what:<18} {strokes:>8}  "
              f"{r['duration_s']:>6.1f}s  {cruise:>9}  {fs:>10}")


if __name__ == "__main__":
    main()
