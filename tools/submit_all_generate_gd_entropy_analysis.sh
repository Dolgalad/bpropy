#!/bin/bash
set -euo pipefail

SUBMIT_SCRIPT="./tools/submit_generate_gd_entropy_analysis.sh"

CONFIGS=(
  "$HOME/bpropy/configs/train/scp/scp_n50_m50_k5_var_logits_bce.json"
  "$HOME/bpropy/configs/train/scp/scp_n50_m50_k5_lbp_vce.json"
  "$HOME/bpropy/configs/train/scp/scp_n50_m50_k5_lbp_vce_tree_approx_max_transformer.json"
  "$HOME/bpropy/configs/train/scp/scp_n50_m50_k5_lbp_vce_pairwise_approx_max_transformer.json"

  "$HOME/bpropy/configs/train/scp/scp_n120-180_m150-250_k5_var_logits_bce.json"
  "$HOME/bpropy/configs/train/scp/scp_n120-180_m150-250_k5_lbp_vce.json"
  "$HOME/bpropy/configs/train/scp/scp_n120-180_m150-250_k5_lbp_vce_tree_approx_max_transformer.json"
  "$HOME/bpropy/configs/train/scp/scp_n120-180_m150-250_k5_lbp_vce_pairwise_approx_max_transformer.json"

  "$HOME/bpropy/configs/train/scp/scp_n150-300_m150-300_k15_var_logits_bce.json"
  "$HOME/bpropy/configs/train/scp/scp_n150-300_m150-300_k15_lbp_vce_tree_approx_max_transformer.json"
  "$HOME/bpropy/configs/train/scp/scp_n150-300_m150-300_k15_lbp_vce_pairwise_approx_max_transformer.json"
)

for cfg in "${CONFIGS[@]}"; do
  "${SUBMIT_SCRIPT}" "$cfg"
done
