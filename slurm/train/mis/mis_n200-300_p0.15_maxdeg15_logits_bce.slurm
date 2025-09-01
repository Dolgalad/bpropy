#!/bin/bash
#SBATCH --job-name=mis_n200-300_p0.15_maxdeg15_train_logits_bce
##SBATCH --partition=<your-partition>
#SBATCH --gpus 1
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/train/mis/mis_n200-300_p0.15_maxdeg15_train_logits_bce.out
#SBATCH --error=slurm_outputs/train/mis/mis_n200-300_p0.15_maxdeg15_train_logits_bce.err
#SBATCH --exclusive

set -euo pipefail

# Parameters
DATASET="/data1/schulz/datasets/mis/mis_n200-300_p0.15_maxdeg15"
LOGDIR="/data1/schulz/logs/mis/mis_n200-300_p0.15_maxdeg15/logits_bce"
LR="1e-6"
SOLVER="kamis"
MODEL_CLASS="HeteroGCNVarLogits"
MODEL_ARGS='{"in_variable_dim": 10, "in_constraint_dim": 4}'
PROCESSED_DIR="/data1/schulz/datasets/mis/mis_n200-300_p0.15_maxdeg15/processed_logits_bce"
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
  --patience "${PATIENCE}" \
  --maximize
