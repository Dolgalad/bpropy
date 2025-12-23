#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-99
#SBATCH --output=slurm_outputs/dataset_gen/%x/%A_%a.out
#SBATCH --error=slurm_outputs/dataset_gen/%x/%A_%a.err

set -euo pipefail

CONFIG="${1:?Usage: sbatch dataset_gen.slurm path/to/config.json}"
OUT_OVERRIDE="${2:-}"

mkdir -p slurm_outputs/dataset_gen/${SLURM_JOB_NAME}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Use your activated env python (or set PYTHON=... when submitting)
PYTHON="${PYTHON:-python}"

if [[ -n "$OUT_OVERRIDE" ]]; then
  bpropy-datasets --config "$CONFIG" --out-override "$OUT_OVERRIDE"
else
  bpropy-datasets --config "$CONFIG"
fi

