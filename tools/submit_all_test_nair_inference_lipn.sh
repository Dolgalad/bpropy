#!/bin/bash
set -euo pipefail

SUBMIT_SCRIPT="./tools/submit_test_nair_lipn.sh"

PROPS=(0.5 0.6 0.7 0.8 0.9)
DECODINGS=(GD EGG)

MAX_INSTANCES=2

CONFIGS=(
  "$HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n50_m50_k5/scp_n50_m50_k5_var_logits_bce.json"
  "$HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n50_m50_k5/scp_n50_m50_k5_lbp_fce.json"
  "$HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n50_m50_k5/scp_n50_m50_k5_nmf_fce.json"

  "$HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_var_logits_bce.json"
  "$HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_lbp_fce.json"
  "$HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_nmf_fce.json"

  "$HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n100_p0.15/mis_n100_p0.15_var_logits_bce.json"
  "$HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n100_p0.15/mis_n100_p0.15_lbp_fce.json"
  "$HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n100_p0.15/mis_n100_p0.15_nmf_fce.json"

  "$HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n200-300_p0.15/mis_n200-300_p0.15_var_logits_bce.json"
  "$HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n200-300_p0.15/mis_n200-300_p0.15_lbp_fce.json"
  "$HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n200-300_p0.15/mis_n200-300_p0.15_nmf_fce.json"

)

for cfg in "${CONFIGS[@]}"; do
  for p in "${PROPS[@]}"; do
    for dec in "${DECODINGS[@]}"; do
      "${SUBMIT_SCRIPT}" "$cfg" "$p" "$dec"
    done
  done
done
