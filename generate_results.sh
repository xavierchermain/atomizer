#!/usr/bin/env bash
set -euo pipefail

# check that Conda is available
command -v conda >/dev/null || echo "conda not found in PATH, try init.sh" || exit 1

# trap ERR to report the failing command
trap 'echo "Command failed: $BASH_COMMAND" >&2' ERR

conda activate atomizer
# python3="conda run python3"

conda run "python tools/atomize.py data/param/triangle_24.json"
conda run "python tools/atomize.py data/param/calibration_cube.json"
conda run "python tools/atomize.py data/param/tubes.json"
conda run "python tools/atomize.py data/param/opposite_curvature_final_90.json"
conda run "python tools/atomize.py data/param/pisa_tower.json"
conda run "python tools/atomize.py data/param/dome.json"
conda run "python tools/atomize.py data/param/cubic.json"
conda run "python tools/atomize.py data/param/city_final.json"
conda run "python tools/atomize.py data/param/car.json"
conda run "python tools/atomize.py data/param/ankle.json"
