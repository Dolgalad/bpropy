#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?Usage: submit_train.sh path/to/config.json [extra args...]}"
shift || true

EXTRA_ARGS=("$@")
EXTRA_ARGS_ESCAPED="$(printf '%q ' "${EXTRA_ARGS[@]}")"

ROOT="${BPROPY_ROOT:-${WORK:-$(pwd)}}"
CACHE_ROOT="${BPROPY_CACHE_ROOT:-${SCRATCH:-${ROOT}}/bpropy/cache}"

# helper: make relative path absolute wrt ROOT; keep absolute paths untouched
abspath() {
  local p="${1:-}"
  [[ -z "$p" ]] && { echo ""; return; }
  if [[ "$p" = /* ]]; then echo "$p"; else p="${p#./}"; echo "${ROOT%/}/$p"; fi
}

# helper: read .slurm.KEY if it exists and is not null/empty, else use default
jget() {
  local key="$1"
  local def="${2:-}"
  local val
  val="$(jq -r --arg k "$key" '.slurm[$k] // empty' "$CONFIG")"
  if [[ -n "$val" && "$val" != "null" ]]; then echo "$val"; else echo "$def"; fi
}

# Slurm resources from config (with sensible fallbacks)
PARTITION="$(jget partition "gpu")"
TIME="$(jget time "24:00:00")"
CPUS="$(jget cpus_per_task "8")"
MEM="$(jget mem "32G")"
GPUS="$(jget gpus "1")"
BASENAME="$(jget job_name "bpropy_train")"
JOBNAME_BASE="test_$(jget job_name "bpropy_train")"


# We'll suffix these per-seed inside the job
OUT_BASE="$(jget output "slurm/out/%x_%A_%a.out")"
ERR_BASE="$(jget error  "slurm/err/%x_%A_%a.err")"

# Optional extras
QOS="$(jget qos "")"
ACCOUNT="$(jget account "")"
CONSTRAINT="$(jget constraint "")"

OUT_BASE="${ROOT}/slurm_outputs/${JOBNAME_BASE}/%x_%A_%a.out"
ERR_BASE="${ROOT}/slurm_outputs/${JOBNAME_BASE}/%x_%A_%a.err"
mkdir -p $(dirname ${OUT_BASE})

echo $PARTITION
echo $QOS
echo $ACCOUNT

echo ${OUT_BASE}
echo ${ERR_BASE}

# Seeds: either from config .slurm.seeds (array), or default [0,1,2]
# Example in config:
# "slurm": { "seeds": [0, 1, 2], ... }
SEEDS_JSON="$(jq -c '.slurm.seeds // [0,1,2]' "$CONFIG")"

mkdir -p "$(abspath slurm/out)" "$(abspath slurm/err)"
TMP_SCRIPT="$(mktemp /tmp/bpropy-train-XXXXXX.slurm)"

SBATCH_OPT_LINES=()
[[ -n "$QOS" ]]        && SBATCH_OPT_LINES+=("#SBATCH --qos=${QOS}")
[[ -n "$ACCOUNT" ]]    && SBATCH_OPT_LINES+=("#SBATCH --account=${ACCOUNT}")
[[ -n "$CONSTRAINT" ]] && SBATCH_OPT_LINES+=("#SBATCH --constraint=${CONSTRAINT}")

cat > "${TMP_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --array=0-2
#SBATCH --job-name=${JOBNAME_BASE}
#SBATCH -A bnq@v100
#SBATCH --exclusive
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
$(printf "%s\n" "${SBATCH_OPT_LINES[@]}")

set -euo pipefail

ROOT="${ROOT}"
CONFIG="${CONFIG}"
SEEDS_JSON='${SEEDS_JSON}'
EXTRA_ARGS_ESCAPED='${EXTRA_ARGS_ESCAPED}'

eval "set -- ${EXTRA_ARGS_ESCAPED}"

# Pick seed for this array task
SEED=\$(python - <<'PY'
import json, os
seeds = json.loads(os.environ["SEEDS_JSON"])
tid = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(seeds[tid])
PY
)

# Suffix for job name and log dirs
SUFFIX="_seed\${SEED}"

# Make job run as if launched from ROOT
cd "\${ROOT}"
export PYTHONPATH="\${PYTHONPATH:-}:\${ROOT}"

export PYTHONUNBUFFERED=1

CACHE_ROOT="${CACHE_ROOT}"
mkdir -p "${CACHE_ROOT}"

# Ensure slurm log dirs exist under ROOT (if you keep them relative)
mkdir -p "\${ROOT}/slurm/out" "\${ROOT}/slurm/err"

# IMPORTANT:
# - %x expands to job-name; we set job-name dynamically with srun --job-name
# - %A and %a are job id and array task id
OUT_FILE="${OUT_BASE}"
ERR_FILE="${ERR_BASE}"

# If OUT/ERR are relative, interpret wrt ROOT
case "\$OUT_FILE" in
  /*) : ;;
  *) OUT_FILE="\${ROOT}/\${OUT_FILE#./}" ;;
esac
case "\$ERR_FILE" in
  /*) : ;;
  *) ERR_FILE="\${ROOT}/\${ERR_FILE#./}" ;;
esac

# Launch with per-seed job name and per-seed logging directory
# 1) job name suffix: use srun --job-name (works per array task)
# 2) log directory suffix: pass through env var or CLI arg (choose one)

srun --job-name="${JOBNAME_BASE}\${SUFFIX}" \\
     --output="\${OUT_FILE}" --error="\${ERR_FILE}" \\
     bpropy-test --config "\${CONFIG}" --checkpoint "runs/${BASENAME}\${SUFFIX}/checkpoints/best.pt" --seed "\${SEED}" --cache-dir "\${CACHE_ROOT}"
EOF

sbatch "${TMP_SCRIPT}"

