#!/bin/bash
set -euo pipefail

SUBMIT_SCRIPT="bpropy-test-nair"

PROPS=(0.5 0.6 0.7 0.8 0.9)
DECODINGS=(GD EGG)

MAX_INSTANCES=50

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
  config_basename="$(basename "$cfg")"
  config_stem="${config_basename%.*}"
  checkpoint_dirname="${config_stem}_seed0"
  checkpoint_path="$HOME/bpropy-jz/runs/${checkpoint_dirname}/checkpoints/best.pt"
  for p in "${PROPS[@]}"; do
    for dec in "${DECODINGS[@]}"; do
      output_path="$HOME/bpropy-jz/test_inference_fix_and_solve/${checkpoint_dirname}_${dec}_p$p.json"
      "${SUBMIT_SCRIPT}" \
	      --config "$cfg" \
	      --checkpoint "${checkpoint_path}" \
	      --seed 0 \
	      --fix-prop "$p" \
	      --decoding "$dec" \
	      --output-json "${output_path}" \
	      --max-test-instances $MAX_INSTANCES
    done
  done
done
