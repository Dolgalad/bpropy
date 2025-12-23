#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-configs/dataset_gen}"
TEMPLATE="${2:-slurm/templates/dataset_gen.slurm}"
OUT_ROOT="${3:-}"   # optional, e.g. /scratch/$USER/datasets

if [[ ! -f "$TEMPLATE" ]]; then
  echo "Template not found: $TEMPLATE" >&2
  exit 1
fi

shopt -s nullglob
configs=("$CONFIG_DIR"/*.json)
if (( ${#configs[@]} == 0 )); then
  echo "No .json configs found in: $CONFIG_DIR" >&2
  exit 1
fi

for cfg in "${configs[@]}"; do
  name="$(basename "$cfg" .json)"

  if [[ -n "$OUT_ROOT" ]]; then
    out="${OUT_ROOT}/${name}"
    echo "[sbatch] $name -> $out"
    sbatch --job-name="$name" "$TEMPLATE" "$cfg" "$out"
  else
    echo "[sbatch] $name"
    sbatch --job-name="$name" "$TEMPLATE" "$cfg"
  fi
done

