#!/bin/bash
#SBATCH --job-name=scp_n50_m50_k5_train_lbp_bce
##SBATCH --partition=<your-partition>
#SBATCH --gpus 1
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/train/scp/scp_n50_m50_k5_train_lbp_bce.out
#SBATCH --error=slurm_outputs/train/scp/scp_n50_m50_k5_train_lbp_bce.err
#SBATCH --exclusive

set -euo pipefail

# Parameters
DATASET="/data1/schulz/datasets/scp/scp_n50_m50_k5"
LOGDIR="/data1/schulz/logs/scp/lbp_bce"
LR="1e-6"
SOLVER="cbc"
MODEL_CLASS="Hetero_GCN_MRF_v4"
MODEL_ARGS='{"in_variable_dim": 10, "in_constraint_dim": 4, "factor_sizes": [1,2,3,4,5]}'
PROCESSED_DIR="/data1/schulz/datasets/scp/scp_n50_m50_k5/processed_lbp_bce"
EPOCHS=10000
PATIENCE=100

# Reproducibility: same base seed for everyone
SEED=2025

cd scripts/train
python -u train_lbp.py \
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
