#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?Usage: submit_transfer.sh path/to/config.json [extra args...]}"
shift || true

ROOT="${BPROPY_ROOT:-${WORK:-$(pwd)}}"
CACHE_ROOT="${BPROPY_CACHE_ROOT:-${SCRATCH:-${ROOT}}/bpropy/cache}"

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
TIME="$(jget time "04:00:00")"
CPUS="$(jget cpus_per_task "8")"
MEM="$(jget mem "32G")"
GPUS="$(jget gpus "1")"
JOBNAME_BASE="$(jget job_name "bpropy_transfer")"

OUT_BASE="$(jget output "slurm/out/%x_%j.out")"
ERR_BASE="$(jget error  "slurm/err/%x_%j.err")"

QOS="$(jget qos "")"
ACCOUNT="$(jget account "")"
CONSTRAINT="$(jget constraint "")"

OUT_BASE="${ROOT}/slurm_outputs/${JOBNAME_BASE}/%x_%A_%a.out"
ERR_BASE="${ROOT}/slurm_outputs/${JOBNAME_BASE}/%x_%A_%a.err"
mkdir -p $(dirname ${OUT_BASE})

# single seed only
SEED="$(jq -r '.seed // 0' "$CONFIG")"

mkdir -p "$(abspath slurm/out)" "$(abspath slurm/err)"
TMP_SCRIPT="$(mktemp /tmp/bpropy-transfer-XXXXXX.slurm)"

SBATCH_OPT_LINES=()
[[ -n "$QOS" ]]        && SBATCH_OPT_LINES+=("#SBATCH --qos=${QOS}")
[[ -n "$ACCOUNT" ]]    && SBATCH_OPT_LINES+=("#SBATCH --account=${ACCOUNT}")
[[ -n "$CONSTRAINT" ]] && SBATCH_OPT_LINES+=("#SBATCH --constraint=${CONSTRAINT}")

cat > "${TMP_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=${TIME}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --job-name=${JOBNAME_BASE}
#SBATCH -A bnq@v100
#SBATCH --exclusive
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#
$(printf "%s\n" "${SBATCH_OPT_LINES[@]}")

set -euo pipefail

ROOT="${ROOT}"
CONFIG="${CONFIG}"
SEED="${SEED}"

cd "\${ROOT}"
export PYTHONPATH="\${PYTHONPATH:-}:\${ROOT}"

CACHE_ROOT="${CACHE_ROOT}"
mkdir -p "${CACHE_ROOT}"

OUT_FILE="${OUT_BASE}"
ERR_FILE="${ERR_BASE}"

case "\$OUT_FILE" in
  /*) : ;;
  *) OUT_FILE="\${ROOT}/\${OUT_FILE#./}" ;;
esac

case "\$ERR_FILE" in
  /*) : ;;
  *) ERR_FILE="\${ROOT}/\${ERR_FILE#./}" ;;
esac

srun --job-name="${JOBNAME_BASE}_seed\${SEED}" \\
     --output="\${OUT_FILE}" --error="\${ERR_FILE}" \\
     bpropy-train --config "\${CONFIG}" --seed "\${SEED}" --cache-dir "\${CACHE_ROOT}"
EOF

sbatch "${TMP_SCRIPT}"
