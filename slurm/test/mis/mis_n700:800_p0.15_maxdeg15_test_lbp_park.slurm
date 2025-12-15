#!/bin/bash
#SBATCH --job-name=mis_n700:800_p0.15_maxdeg15_test_lbp_park
##SBATCH --partition=<your-partition>
#SBATCH --gpus 1
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/test/mis/mis_n700:800_p0.15_maxdeg15_test_lbp_park.out
#SBATCH --error=slurm_outputs/test/mis/mis_n700:800_p0.15_maxdeg15_test_lbp_park.err
##SBATCH --exclusive
#SBATCH -A bnq@v100

set -euo pipefail
export PYTHONUNBUFFERED=TRUE

# Parameters
DATASET="$WORK/datasets/mis/mis_n700:800_p0.15_maxdeg15"
PROCESSED_DIR="$WORK/datasets/mis/mis_n700:800_p0.15_maxdeg15/processed_lbp"

# Reproducibility: same base seed for everyone
SEED=2025

cd scripts/test
python -u test_lbp_park.py \
  "${DATASET}" \
  "${PROCESSED_DIR}"
