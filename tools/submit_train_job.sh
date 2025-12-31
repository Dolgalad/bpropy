#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?Usage: submit_train.sh path/to/config.json [extra args...]}"
shift || true

# 1) Choose a "root" directory:
# - If BPROPY_ROOT is set, use it
# - else if WORK is set, use it (common on HPC clusters)
# - else use current directory (lab cluster / local)
ROOT="${BPROPY_ROOT:-${WORK:-$(pwd)}}"

# Optional: if you submit from anywhere, but want repo root automatically,
# you can replace $(pwd) with: git rev-parse --show-toplevel 2>/dev/null || pwd

# load global defaults (optional)
DEFAULTS_FILE="${BPROPY_SLURM_DEFAULTS:-./slurm_defaults.sh}"
if [[ -f "${DEFAULTS_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${DEFAULTS_FILE}"
fi

# helper: make relative path absolute wrt ROOT; keep absolute paths untouched
abspath() {
  local p="${1:-}"
  [[ -z "$p" ]] && { echo ""; return; }
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    # strip leading ./ if present
    p="${p#./}"
    echo "${ROOT%/}/$p"
  fi
}

# helper: read .slurm.KEY if it exists and is not null/empty, else use default
jget() {
  local key="$1"
  local def="${2:-}"
  local val
  val="$(jq -r --arg k "$key" '.slurm[$k] // empty' "$CONFIG")"
  if [[ -n "$val" && "$val" != "null" ]]; then echo "$val"; else echo "$def"; fi
}

PARTITION="$(jget partition      "${SLURM_DEF_PARTITION:-gpu}")"
QOS="$(jget qos                 "${SLURM_DEF_QOS:-}")"
ACCOUNT="$(jget account         "${SLURM_DEF_ACCOUNT:-}")"
TIME="$(jget time               "${SLURM_DEF_TIME:-24:00:00}")"
CPUS="$(jget cpus_per_task      "${SLURM_DEF_CPUS_PER_TASK:-8}")"
MEM="$(jget mem                 "${SLURM_DEF_MEM:-32G}")"
GPUS="$(jget gpus               "${SLURM_DEF_GPUS:-1}")"
CONSTRAINT="$(jget constraint   "${SLURM_DEF_CONSTRAINT:-}")"
JOBNAME="$(jget job_name        "${SLURM_DEF_JOB_NAME:-bpropy_train}")"

# Paths from config: if relative, interpret them wrt ROOT
OUT="$(abspath "$(jget output "${SLURM_DEF_OUTPUT:-slurm/out/%x_%j.out}")")"
ERR="$(abspath "$(jget error  "${SLURM_DEF_ERROR:-slurm/err/%x_%j.err}")")"

# Optionally also allow a default log root under ROOT:
mkdir -p "$(dirname "$OUT")" "$(dirname "$ERR")"

TMP_SCRIPT="$(mktemp /tmp/bpropy-train-XXXXXX.slurm)"

SBATCH_OPT_LINES=()
[[ -n "$QOS" ]]        && SBATCH_OPT_LINES+=("#SBATCH --qos=${QOS}")
[[ -n "$ACCOUNT" ]]    && SBATCH_OPT_LINES+=("#SBATCH --account=${ACCOUNT}")
[[ -n "$CONSTRAINT" ]] && SBATCH_OPT_LINES+=("#SBATCH --constraint=${CONSTRAINT}")

cat > "${TMP_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --job-name=${JOBNAME}
#SBATCH --output=${OUT}
#SBATCH --error=${ERR}
$(printf "%s\n" "${SBATCH_OPT_LINES[@]}")

set -euo pipefail

# Make job run "as if launched from ROOT"
cd "${ROOT}"

# If you want logs created under ROOT even when OUT/ERR contain relative pieces:
mkdir -p "$(dirname "${OUT}")" "$(dirname "${ERR}")"

# Optional: put ROOT on PYTHONPATH so running from WORK still finds your repo package
export PYTHONPATH="\${PYTHONPATH:-}:${ROOT}"

# ---- Training launch ----
# If user passes --data-root relative, interpret wrt ROOT by telling them:
#   --data-root datasets/...   (becomes ${ROOT}/datasets/...)
srun bpropy-train --config "${CONFIG}" "\$@"
EOF

sbatch "${TMP_SCRIPT}"

