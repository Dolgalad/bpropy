#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?Usage: submit_generate_gd_entropy_analysis.sh path/to/config.json [extra args...]}"
shift || true

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

PARTITION="$(jget partition "gpu_p13")"
TIME="$(jget time "20:00:00")"
CPUS="$(jget cpus_per_task "8")"
MEM="$(jget mem "32G")"
GPUS="$(jget gpus "1")"
BASENAME="$(jget job_name "bpropy_train")"

CONFIG_BASE="$(basename "$CONFIG" .json)"

case "$CONFIG_BASE" in
  scp_n50_m50_k5_*)
    DATASET_ID="scp_n50_m50_k5"
    ;;
  scp_n120-180_m150-250_k5_*)
    DATASET_ID="scp_n120-180_m150-250_k5"
    ;;
  scp_n150-300_m150-300_k15_*)
    DATASET_ID="scp_n150-300_m150-300_k15"
    ;;
  *)
    echo "Could not infer dataset id from config: $CONFIG_BASE" >&2
    exit 1
    ;;
esac

case "$CONFIG_BASE" in
  *_var_logits_bce)
    MODEL_ID="gnn"
    DECODINGS=(EGG)
    ;;
  *_lbp_vce)
    MODEL_ID="gnn_lbp_full"
    DECODINGS=(EGG GD)
    ;;
  *_lbp_vce_tree_approx_max_transformer)
    MODEL_ID="gnn_lbp_tree"
    DECODINGS=(EGG GD)
    ;;
  *_lbp_vce_pairwise_approx_max_transformer)
    MODEL_ID="gnn_lbp_pair"
    DECODINGS=(EGG GD)
    ;;
  *)
    echo "Could not infer model id from config: $CONFIG_BASE" >&2
    exit 1
    ;;
esac

JOBNAME_BASE="gd_entropy_${DATASET_ID}_${MODEL_ID}"

OUT_BASE="${ROOT}/slurm_outputs/${JOBNAME_BASE}/%x_%A_%a.out"
ERR_BASE="${ROOT}/slurm_outputs/${JOBNAME_BASE}/%x_%A_%a.err"
mkdir -p "$(dirname "${OUT_BASE}")"

RESULTS_DIR="${ROOT}/test_gd_entropy_per_iteration/${DATASET_ID}/${MODEL_ID}"
mkdir -p "${RESULTS_DIR}"

TMP_SCRIPT="$(mktemp /tmp/gd-entropy-XXXXXX.slurm)"

cat > "${TMP_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --array=0-2
#SBATCH --job-name=${JOBNAME_BASE}
#SBATCH -A bnq@v100

set -euo pipefail

export ROOT="${ROOT}"
export CONFIG="${CONFIG}"
export RESULTS_DIR="${RESULTS_DIR}"
export BASENAME="${BASENAME}"
export PYTHONUNBUFFERED=1

cd "\${ROOT}"
export PYTHONPATH="\${PYTHONPATH:-}:\${ROOT}"

eval "set -- ${EXTRA_ARGS_ESCAPED}"

SEEDS=(0 1 2)
SEED=\${SEEDS[\${SLURM_ARRAY_TASK_ID}]}

CKPT_PATH="runs/${BASENAME}_seed\${SEED}/checkpoints/best.pt"
OUTPUT_JSON="\${RESULTS_DIR}/\${SEED}.json"

srun --job-name="${JOBNAME_BASE}_seed\${SEED}" \\
     --output="${OUT_BASE}" \\
     --error="${ERR_BASE}" \\
     python $HOME/bpropy/tools/generate_gd_entropy_analysis.py \\
       --config "\${CONFIG}" \\
       --checkpoint "\${CKPT_PATH}" \\
       --seed "\${SEED}" \\
       --output-json "\${OUTPUT_JSON}" \\
       --decodings ${DECODINGS[*]} \\
EOF

cat "${TMP_SCRIPT}"
sbatch "${TMP_SCRIPT}"
