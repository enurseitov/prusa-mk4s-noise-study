#!/bin/sh
# Rebuild everything in gcode/.
set -e
cd "$(dirname "$0")"
mkdir -p gcode

# the sweep, phase stepping on and off - the pair that gives the main result
./gen_noise_sweep.py --phase-stepping on  --out gcode/mk4s_sweep_phase_on.gcode
./gen_noise_sweep.py --phase-stepping off --out gcode/mk4s_sweep_phase_off.gcode

# short A/B: every speed twice, phase stepping off then on, in one file
./gen_noise_sweep.py --phase-ab           --out gcode/mk4s_ab_phase.gcode

echo "done"
