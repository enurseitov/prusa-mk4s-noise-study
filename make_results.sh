#!/bin/sh
# Rebuild everything in results/ from the recordings.
set -e
cd "$(dirname "$0")"
mkdir -p results
SCHED=gcode/mk4s_sweep_phase_on_schedule.csv

./plot_states.py \
    recordings/2026-09-30_1_phase_off.csv \
    recordings/2026-09-30_2_phase_on_uncalibrated.csv \
    recordings/2026-09-30_3_phase_on_calibrated.csv \
    --schedule $SCHED -o results/phase_stepping.png

# sanity check that two runs of the same file really line up
./plot_alignment.py \
    recordings/2026-09-30_1_phase_off.csv \
    recordings/2026-09-30_3_phase_on_calibrated.csv \
    --schedule $SCHED --label-before "phase stepping off" --label-after "on, calibrated" \
    -o results/alignment_check.png
