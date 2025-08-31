#!/bin/bash
#SBATCH --job-name=mis_n200:300_p0.15_maxdeg15_gen
##SBATCH --partition=<your-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-99     # 100 shards
#SBATCH --output=slurm_outputs/data/mis/mis_n200:300_p0.15_maxdeg15_gen-%a.out
#SBATCH --error=slurm_outputs/data/mis/mis_n200:300_p0.15_maxdeg15_gen-%a.err
#SBATCH --exclusive

# Hardening: keep single-threaded libs single-threaded
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -euo pipefail

# Parameters
OUTDIR="datasets/problems/mis/mis_n200:300_p0.15_maxdeg15"
NV="200:300"
MAXDEG=15
P=0.15

# Reproducibility: same base seed for everyone
BASE_SEED=2025

# Per-array-shard workload
SHARD_COUNT=100
TOTAL=5000           # total instances desired
PER_SHARD=$(( TOTAL / SHARD_COUNT ))

# Compute offset from SLURM_ARRAY_TASK_ID
OFFSET=$(( SLURM_ARRAY_TASK_ID * PER_SHARD ))

# Create output dir once (ok if exists)
mkdir -p "${OUTDIR}"

# Activate your environment if needed
# source ~/.bashrc
# conda activate bpropy

cd scripts/data
python -u mis_dataset_generator.py \
  "${OUTDIR}" \
  "${NV}" "${MAXDEG}" \
  "${PER_SHARD}" \
  --p "${P}" \
  --seed "${BASE_SEED}" \
  --index-offset "${OFFSET}" \
  --solve kamis \
  --splits 80 10 10 \
  --threshold-bytes 8000 \
  --prefix "shard${SLURM_ARRAY_TASK_ID}_"

# Optional: verify count
echo "Shard ${SLURM_ARRAY_TASK_ID} done."

