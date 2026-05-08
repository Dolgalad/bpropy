#!/bin/bash
set -euo pipefail

SUBMIT_SCRIPT="./tools/submit_flow_train.sh"
BPROPY_CONFIG_ROOT="${BPROPY_CONFIG_ROOT:-$HOME/bpropy-github-clean/configs/train}"

if [[ ! -x "$SUBMIT_SCRIPT" ]]; then
  echo "Error: submit script not found or not executable: $SUBMIT_SCRIPT" >&2
  exit 1
fi

# Flow configs are stored as:
#   configs/train/{mis,scp}/flow/<dataset_id>/<run_name>.json
# Iterate deterministically over every config file in those directories.
mapfile -t CONFIG_FILES < <(
  find \
    "$BPROPY_CONFIG_ROOT/mis/flow" \
    "$BPROPY_CONFIG_ROOT/scp/flow" \
    -mindepth 2 -maxdepth 2 -type f -name '*.json' 2>/dev/null \
  | sort
)

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
  echo "Error: no flow config files found under:" >&2
  echo "  $BPROPY_CONFIG_ROOT/mis/flow" >&2
  echo "  $BPROPY_CONFIG_ROOT/scp/flow" >&2
  exit 1
fi

echo "Submitting ${#CONFIG_FILES[@]} flow training jobs..."

for config in "${CONFIG_FILES[@]}"; do
  echo "Submitting: $config"
  "$SUBMIT_SCRIPT" "$config"
done
