#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?Usage: submit_test_nair.sh path/to/config.json FIX_PROP DECODING [extra args...]}"
FIX_PROP="${2:?Missing FIX_PROP}"
DECODING="${3:?Missing DECODING}"
shift 3 || true

EXTRA_ARGS=("$@")
EXTRA_ARGS_ESCAPED="$(printf '%q ' "${EXTRA_ARGS[@]}")"

ROOT="${BPROPY_ROOT:-${WORK:-$(pwd)}}"

jget() {
  local key="$1"
  local def="${2:-}"
  local val
  val="$(jq -r --arg k "$key" '.slurm[$k] // empty' "$CONFIG")"
  if [[ -n "$val" && "$val" != "null" ]]; then
    echo "$val"
  else
    echo "$def"
  fi
}

PARTITION="$(jget partition "gpu")"
TIME="$(jget time "20:00:00")"
CPUS="$(jget cpus_per_task "8")"
MEM="$(jget mem "32G")"
GPUS="$(jget gpus "1")"
BASENAME="$(jget job_name "bpropy_train")"

CONFIG_BASE="$(basename "$CONFIG" .json)"

# -------------------------
# Dataset name
# -------------------------
case "$CONFIG_BASE" in
  scp_n50_m50_k5_*)
    DATASET_NAME="scp_n50_m50_k5"
    ;;
  scp_n120-180_m150-250_k5_*)
    DATASET_NAME="scp_n120-180_m150-250_k5"
    ;;
  scp_n150-300_m150-300_k15_*)
    DATASET_NAME="scp_n150-300_m150-300_k15"
    ;;
  *)
    echo "Could not infer dataset name from config: $CONFIG_BASE" >&2
    exit 1
    ;;
esac

# -------------------------
# Model name
# -------------------------
case "$CONFIG_BASE" in
  *_var_logits_bce)
    MODEL_NAME="gnn"
    ;;
  *_lbp_vce)
    MODEL_NAME="gnn_lbp_full"
    ;;
  *_lbp_vce_tree_approx_max_transformer)
    MODEL_NAME="gnn_lbp_tree"
    ;;
  *_lbp_vce_pairwise_approx_max_transformer)
    MODEL_NAME="gnn_lbp_pair"
    ;;
  *)
    echo "Could not infer model name from config: $CONFIG_BASE" >&2
    exit 1
    ;;
esac

JOBNAME_BASE="test_nair_${DATASET_NAME}_${MODEL_NAME}_${DECODING}_p${FIX_PROP//./p}"

OUT_BASE="${ROOT}/slurm_outputs/${JOBNAME_BASE}/%x_%A_%a.out"
ERR_BASE="${ROOT}/slurm_outputs/${JOBNAME_BASE}/%x_%A_%a.err"
mkdir -p "$(dirname "${OUT_BASE}")"

# -------------------------
# Hardcoded seeds
# -------------------------
SEEDS=(0 1 2)
ARRAY_LAST=2

CACHE_DIR="${ROOT}/cache/nair_full_ilp_opt"
mkdir -p "${CACHE_DIR}"

RESULTS_DIR="${ROOT}/test_fix_and_solve/${DATASET_NAME}/${MODEL_NAME}"
mkdir -p "${RESULTS_DIR}"

TMP_SCRIPT="$(mktemp /tmp/bpropy-test-nair-XXXXXX.slurm)"

cat > "${TMP_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_p13
##SBATCH --qos=qos_gpu-dev
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --array=0-${ARRAY_LAST}
#SBATCH --job-name=${JOBNAME_BASE}
#SBATCH -A bnq@v100

set -euo pipefail

export ROOT="${ROOT}"
export CONFIG="${CONFIG}"
export FIX_PROP="${FIX_PROP}"
export DECODING="${DECODING}"
export RESULTS_DIR="${RESULTS_DIR}"
export BASENAME="${BASENAME}"
export PYTHONUNBUFFERED=1
export BPROPY_NAIR_OPT_CACHE="${CACHE_DIR}"

cd "\${ROOT}"
export PYTHONPATH="\${PYTHONPATH:-}:\${ROOT}"

eval "set -- ${EXTRA_ARGS_ESCAPED}"

SEEDS=(0 1 2)
SEED=\${SEEDS[\${SLURM_ARRAY_TASK_ID}]}

CKPT_PATH="runs/${BASENAME}_seed\${SEED}/checkpoints/best.pt"
OUTPUT_JSON="\${RESULTS_DIR}/${DECODING}_p${FIX_PROP}_seed_\${SEED}.json"

srun --job-name="${JOBNAME_BASE}_seed\${SEED}" \\
     --output="${OUT_BASE}" \\
     --error="${ERR_BASE}" \\
     bpropy-test-nair \\
       --config "\${CONFIG}" \\
       --checkpoint "\${CKPT_PATH}" \\
       --seed "\${SEED}" \\
       --fix-prop "\${FIX_PROP}" \\
       --decoding "\${DECODING}" \\
       --output-json "\${OUTPUT_JSON}" \\
#       "\$@"
EOF

cat "${TMP_SCRIPT}"
sbatch "${TMP_SCRIPT}"
