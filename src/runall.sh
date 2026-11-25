#!/bin/sh
# Rebuild the figures, the tables and the write-up from the per-run statistics kept in results/runs.
# No simulation is run. Needs Python 3 with NumPy and Matplotlib, and pdflatex.
set -e
cd "$(dirname "$0")/.."
python3 src/7_parameter_sweep/sweep_maps.py results/runs/sweep/done results 0.005 0.02 0.005
python3 src/7_parameter_sweep/writeup_sweep.py results/runs/sweep/done results/runs/Aprime results/runs/port results doc/sweep_writeup.pdf
