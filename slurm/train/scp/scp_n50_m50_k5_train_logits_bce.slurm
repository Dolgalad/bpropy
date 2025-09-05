#!/bin/bash
#SBATCH --job-name=scp_n50_m50_k5_train_logits_bce
##SBATCH --partition=<your-partition>
#SBATCH --gpus 1
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/train/scp/scp_n50_m50_k5_train_logits_bce.out
#SBATCH --error=slurm_outputs/train/scp/scp_n50_m50_k5_train_logits_bce.err
##SBATCH --exclusive
#SBATCH -A bnq@v100

set -euo pipefail
export PYTHONUNBUFFERED=TRUE

# Parameters
DATASET="$WORK/datasets/scp/scp_n50_m50_k5"
LOGDIR="$WORK/logs/scp/scp_n50_m50_k5/var_logits_bce"
LR="1e-6"
SOLVER="cbc"
MODEL_CLASS="HeteroGCNVarLogits"
MODEL_ARGS='{"in_variable_dim": 10, "in_constraint_dim": 4}'
PROCESSED_DIR="$WORK/datasets/scp/scp_n50_m50_k5/processed_logits"
EPOCHS=10000
PATIENCE=100

# Reproducibility: same base seed for everyone
SEED=2025

cd scripts/train
python -u train_var_logits.py \
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
  --patience "${PATIENCE}"
