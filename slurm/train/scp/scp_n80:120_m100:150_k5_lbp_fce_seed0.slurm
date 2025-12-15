#!/bin/bash
#SBATCH --job-name=scp_n80_120_m100_150_k5_train_lbp_fce_seed0
##SBATCH --partition=<your-partition>
#SBATCH --gpus 1
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/train/scp/scp_n80_120_m100_150_k5_train_lbp_fce_seed0.out
#SBATCH --error=slurm_outputs/train/scp/scp_n80_120_m100_150_k5_train_lbp_fce_seed0.err
##SBATCH --exclusive
#SBATCH -A bnq@v100

set -euo pipefail
export PYTHONUNBUFFERED=TRUE

# Parameters
DATASET="$WORK/datasets/scp/scp_n80:120_m100:150_k5"
LOGDIR="$WORK/logs/scp/scp_n80:120_m100:150_k5/lbp_fce"
LR="1e-6"
SOLVER="cbc"
MODEL_CLASS="Hetero_GCN_MRF_v4"
MODEL_ARGS='{"in_variable_dim": 10, "in_constraint_dim": 4, "factor_sizes": [1,2,3,4,5], "normalize_theta":true}'
PROCESSED_DIR="$WORK/datasets/scp/scp_n80:120_m100:150_k5/processed_lbp_yss"
EPOCHS=10000
PATIENCE=100
RESUME="/lustre/fswork/projects/rech/bnq/udj74tc/logs/scp/scp_n80:120_m100:150_k5/lbp_fce/Hetero_GCN_MRF_v4_hetero_1761309491_2023"

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
  --resume "${RESUME}" \
  --with_lbp
