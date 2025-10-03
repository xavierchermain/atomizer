#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null; then
    echo "conda not found in PATH, try init.sh"
    exit 1
fi

trap 'echo "Command failed: $BASH_COMMAND" >&2' ERR

param_files=(
    "triangle_24"
    "calibration_cube"
    "tubes"
    "opposite_curvature_final_90"
    "pisa_tower"
    "dome"
    "cubic"
    "city_final"
    "car"
    "ankle"
)

for file in "${param_files[@]}"; do
  echo ">>> Running atomize on $file"
  conda run --live-stream -n atomizer python tools/atomize.py "data/param/$file"
done