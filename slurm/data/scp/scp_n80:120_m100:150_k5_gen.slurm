#!/bin/bash
#SBATCH --job-name=scp_n80:120_m100:150_k5_gen
#SBATCH -A bnq@v100
#SBATCH --gpus 1
##SBATCH --partition=<your-partition>
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-99     # 100 shards
#SBATCH --output=slurm_outputs/data/scp/scp_n80:120_m100:150_k5_gen-%a.out
#SBATCH --error=slurm_outputs/data/scp/scp_n80:120_m100:150_k5_gen-%a.err


set -euo pipefail

# Parameters
OUTDIR="datasets/problems/scp/scp_n80:120_m100:150_k5"
N_ITEMS="80:120"
N_SETS="100:150"
K=5

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
python -u scp_dataset_generator.py \
  "${OUTDIR}" \
  "${N_ITEMS}" "${N_SETS}" "${K}" \
  "${PER_SHARD}" \
  --seed "${BASE_SEED}" \
  --index-offset "${OFFSET}" \
  --balanced \
  --solve cbc \
  --splits 80 10 10 \
  --threshold-bytes 8000 \
  --prefix "shard${SLURM_ARRAY_TASK_ID}_"

# Optional: verify count
echo "Shard ${SLURM_ARRAY_TASK_ID} done."

