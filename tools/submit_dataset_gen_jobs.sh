#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-configs/datasets_gen}"
TEMPLATE="${2:-slurm/templates/dataset_gen.slurm}"
OUT_ROOT="${3:-}"   # optional: /scratch/$USER/bpropy_datasets

if [[ ! -f "$TEMPLATE" ]]; then
  echo "Template not found: $TEMPLATE" >&2
  exit 1
fi

# Find JSON files recursively
mapfile -d '' configs < <(find "$CONFIG_DIR" -type f -name '*.json' -print0 | sort -z)

if (( ${#configs[@]} == 0 )); then
  echo "No .json configs found under: $CONFIG_DIR" >&2
  exit 1
fi

for cfg in "${configs[@]}"; do
  # Create a nice job name like: mis_n100_p0.15  or scp_n50_m50_k5
  name="$(basename "$cfg" .json)"

  # Optionally mirror the subdir structure in OUT_ROOT (mis/... and scp/...)
  if [[ -n "$OUT_ROOT" ]]; then
    rel="${cfg#${CONFIG_DIR}/}"              # path relative to CONFIG_DIR
    rel_noext="${rel%.json}"                 # drop .json
    out="${OUT_ROOT}/${rel_noext}"           # e.g. /scratch/.../mis/n100_p0.15
    mkdir -p "$(dirname "$out")"
    echo "[sbatch] $name cfg=$cfg out=$out"
    sbatch --job-name="$name" "$TEMPLATE" "$cfg" "$out"
  else
    echo "[sbatch] $name cfg=$cfg"
    sbatch --job-name="$name" "$TEMPLATE" "$cfg"
  fi
done

