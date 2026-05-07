#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?Usage: submit_test_nair.sh path/to/config.json [extra args...]}"
shift || true

EXTRA_ARGS=("$@")
EXTRA_ARGS_ESCAPED="$(printf '%q ' "${EXTRA_ARGS[@]}")"

ROOT="${BPROPY_ROOT:-${WORK:-$(pwd)}}"

abspath() {
  local p="${1:-}"
  [[ -z "$p" ]] && { echo ""; return; }
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    p="${p#./}"
    echo "${ROOT%/}/$p"
  fi
}

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
TIME="$(jget time "24:00:00")"
CPUS="$(jget cpus_per_task "8")"
MEM="$(jget mem "32G")"
GPUS="$(jget gpus "1")"
BASENAME="$(jget job_name "bpropy_train")"

# Distinct Slurm naming/output tree for Nair tests.
JOBNAME_BASE="test_nair_${BASENAME}"

QOS="$(jget qos "")"
ACCOUNT="$(jget account "")"
CONSTRAINT="$(jget constraint "")"

OUT_BASE="${ROOT}/slurm_outputs/${JOBNAME_BASE}/%x_%A_%a.out"
ERR_BASE="${ROOT}/slurm_outputs/${JOBNAME_BASE}/%x_%A_%a.err"
mkdir -p "$(dirname "${OUT_BASE}")"

SEEDS_JSON="$(jq -c '.slurm.seeds // [0,1,2]' "$CONFIG")"
NUM_SEEDS="$(python - <<PY
import json
print(len(json.loads('''${SEEDS_JSON}''')))
PY
)"
ARRAY_LAST=$((NUM_SEEDS - 1))

TEST_SCRIPT="${ROOT}/test_nair.py"

# Shared cache, exported for the Python script.
CACHE_DIR="${ROOT}/cache/nair_full_ilp_opt"
mkdir -p "${CACHE_DIR}"

SBATCH_OPT_LINES=()
[[ -n "$QOS" ]]        && SBATCH_OPT_LINES+=("#SBATCH --qos=${QOS}")
[[ -n "$ACCOUNT" ]]    && SBATCH_OPT_LINES+=("#SBATCH --account=${ACCOUNT}")
[[ -n "$CONSTRAINT" ]] && SBATCH_OPT_LINES+=("#SBATCH --constraint=${CONSTRAINT}")

TMP_SCRIPT="$(mktemp /tmp/bpropy-test-nair-XXXXXX.slurm)"

cat > "${TMP_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_p13
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --array=0-${ARRAY_LAST}
#SBATCH --job-name=${JOBNAME_BASE}
$(printf "%s\n" "${SBATCH_OPT_LINES[@]}")

set -euo pipefail

export ROOT="${ROOT}"
export CONFIG="${CONFIG}"
export SEEDS_JSON='${SEEDS_JSON}'
export EXTRA_ARGS_ESCAPED='${EXTRA_ARGS_ESCAPED}'
export PYTHONUNBUFFERED=1
export BPROPY_NAIR_OPT_CACHE="${CACHE_DIR}"

cd "\${ROOT}"
export PYTHONPATH="\${PYTHONPATH:-}:\${ROOT}"

eval "set -- \${EXTRA_ARGS_ESCAPED}"

SEED=\$(python - <<'PY'
import json, os
seeds = json.loads(os.environ["SEEDS_JSON"])
tid = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(seeds[tid])
PY
)

RUN_NAME="${BASENAME}_nair_test_seed\${SEED}"
CKPT_PATH="runs/${BASENAME}_seed\${SEED}/checkpoints/best.pt"

srun --job-name="${JOBNAME_BASE}_seed\${SEED}" \\
     --output="${OUT_BASE}" \\
     --error="${ERR_BASE}" \\
     bpropy-test-nair \\
       --config "\${CONFIG}" \\
       --checkpoint "\${CKPT_PATH}" \\
       --seed "\${SEED}"
EOF

cat "${TMP_SCRIPT}"
sbatch "${TMP_SCRIPT}"
