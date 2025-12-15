#!/bin/bash
#SBATCH --job-name=mis_n700-800_p0.15_maxdeg15_train_lbp_fce_seed0
##SBATCH --partition=<your-partition>
#SBATCH --gpus 1
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/train/mis/mis_n700-800_p0.15_maxdeg15_train_lbp_fce_seed0.out
#SBATCH --error=slurm_outputs/train/mis/mis_n700-800_p0.15_maxdeg15_train_lbp_fce_seed0.err
##SBATCH --exclusive
#SBATCH -A bnq@v100

set -euo pipefail
export PYTHONUNBUFFERED=TRUE

# Parameters
DATASET="$WORK/datasets/mis/mis_n700:800_p0.15_maxdeg15"
LOGDIR="$WORK/logs/mis/mis_n700:800_p0.15_maxdeg15/lbp_fce"
LR="1e-6"
SOLVER="kamis"
MODEL_CLASS="Hetero_GCN_MRF_v4"
MODEL_ARGS='{"in_variable_dim": 10, "in_constraint_dim": 4, "factor_sizes": [1,2,3,4,5]}'
PROCESSED_DIR="$WORK/datasets/mis/mis_n700:800_p0.15_maxdeg15/processed_lbp_yss"
EPOCHS=10000
PATIENCE=100
BATCHSIZE=10
RESUME="$WORK/logs/mis/mis_n700:800_p0.15_maxdeg15/lbp_fce/Hetero_GCN_MRF_v4_hetero_1757562540/checkpoints/last.pt"

# Reproducibility: same base seed for everyone
SEED=2023

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
  --batch_size "${BATCHSIZE}" \
  --with_lbp \
  --maximize \
  --resume "${RESUME}"
