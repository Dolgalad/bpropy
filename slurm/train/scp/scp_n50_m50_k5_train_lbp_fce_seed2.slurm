#!/bin/bash
#SBATCH --job-name=scp_n50_m50_k5_train_lbp_fce_seed2
##SBATCH --partition=<your-partition>
#SBATCH --gpus 1
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/train/scp/scp_n50_m50_k5_train_lbp_fce_seed2.out
#SBATCH --error=slurm_outputs/train/scp/scp_n50_m50_k5_train_lbp_fce_seed2.err
##SBATCH --exclusive
#SBATCH -A bnq@v100

set -euo pipefail
export PYTHONUNBUFFERED=TRUE

# Parameters
DATASET="$WORK/datasets/scp/scp_n50_m50_k5"
LOGDIR="$WORK/logs/scp/scp_n50_m50_k5/lbp_fce"
LR="1e-6"
SOLVER="cbc"
MODEL_CLASS="Hetero_GCN_MRF_v4"
MODEL_ARGS='{"in_variable_dim": 10, "in_constraint_dim": 4, "factor_sizes": [1,2,3,4,5], "normalize_theta":true}'
PROCESSED_DIR="$WORK/datasets/scp/scp_n50_m50_k5/processed_lbp_yss"
EPOCHS=10000
PATIENCE=100

# Reproducibility: same base seed for everyone
SEED=2025

cd scripts/train
python -u train_lbp_factor_ce.py \
  "${DATASET}" \
  --log_dir "${LOGDIR}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --hetero \
  --solver "${SOLVER}" \
  --model_class "${MODEL_CLASS}" \
  --model_args "${MODEL_ARGS}" \
  --processed_dir "${PROCESSED_DIR}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --with_lbp
