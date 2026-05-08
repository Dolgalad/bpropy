#!/bin/bash

SUBMIT_SCRIPT="./tools/submit_flow_train.sh"
BPROPY_CONFIG_ROOT="$HOME/bpropy-github-clean/configs/train"

$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n50_m50_k5/scp_n50_m50_k5_lbp_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n50_m50_k5/scp_n50_m50_k5_lbp_pairwise_approx_transformer_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n50_m50_k5/scp_n50_m50_k5_lbp_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n50_m50_k5/scp_n50_m50_k5_var_logits_flow_q1_bce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n50_m50_k5/scp_n50_m50_k5_lbp_pairwise_approx_transformer_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n50_m50_k5/scp_n50_m50_k5_var_logits_flow_velocity_mse.json

$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_lbp_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_lbp_pairwise_approx_transformer_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_lbp_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_var_logits_flow_q1_bce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_lbp_pairwise_approx_transformer_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_var_logits_flow_velocity_mse.json

$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n150-300_m150-300_k15/scp_n150-300_m150-300_k15_lbp_pairwise_approx_transformer_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n150-300_m150-300_k15/scp_n150-300_m150-300_k15_var_logits_flow_q1_bce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n150-300_m150-300_k15/scp_n150-300_m150-300_k15_lbp_pairwise_approx_transformer_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/flow/scp_n150-300_m150-300_k15/scp_n150-300_m150-300_k15_var_logits_flow_velocity_mse.json

$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n100_p0.15/mis_n100_p0.15_cc_lbp_flow_pairwise_approx_transformer_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n100_p0.15/mis_n100_p0.15_eb_lbp_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n100_p0.15/mis_n100_p0.15_cc_lbp_flow_pairwise_approx_transformer_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n100_p0.15/mis_n100_p0.15_eb_lbp_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n100_p0.15/mis_n100_p0.15_cc_lbp_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n100_p0.15/mis_n100_p0.15_eb_lbp_pairwise_approx_transformer_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n100_p0.15/mis_n100_p0.15_cc_lbp_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n100_p0.15/mis_n100_p0.15_eb_lbp_pairwise_approx_transformer_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n100_p0.15/mis_n100_p0.15_cc_var_logits_flow_q1_bce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n100_p0.15/mis_n100_p0.15_eb_var_logits_flow_q1_bce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n100_p0.15/mis_n100_p0.15_cc_var_logits_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n100_p0.15/mis_n100_p0.15_eb_var_logits_flow_velocity_mse.json

$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n200-300_p0.15/mis_n200-300_p0.15_cc_lbp_flow_pairwise_approx_transformer_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n200-300_p0.15/mis_n200-300_p0.15_eb_lbp_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n200-300_p0.15/mis_n200-300_p0.15_cc_lbp_flow_pairwise_approx_transformer_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n200-300_p0.15/mis_n200-300_p0.15_eb_lbp_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n200-300_p0.15/mis_n200-300_p0.15_cc_lbp_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n200-300_p0.15/mis_n200-300_p0.15_eb_lbp_pairwise_approx_transformer_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n200-300_p0.15/mis_n200-300_p0.15_cc_lbp_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n200-300_p0.15/mis_n200-300_p0.15_eb_lbp_pairwise_approx_transformer_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n200-300_p0.15/mis_n200-300_p0.15_cc_var_logits_flow_q1_bce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n200-300_p0.15/mis_n200-300_p0.15_eb_var_logits_flow_q1_bce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n200-300_p0.15/mis_n200-300_p0.15_cc_var_logits_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n200-300_p0.15/mis_n200-300_p0.15_eb_var_logits_flow_velocity_mse.json

$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n700-800_p0.15/mis_n700-800_p0.15_cc_lbp_flow_pairwise_approx_transformer_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n700-800_p0.15/mis_n700-800_p0.15_cc_lbp_flow_pairwise_approx_transformer_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n700-800_p0.15/mis_n700-800_p0.15_eb_lbp_pairwise_approx_transformer_flow_q1_vce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n700-800_p0.15/mis_n700-800_p0.15_eb_lbp_pairwise_approx_transformer_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n700-800_p0.15/mis_n700-800_p0.15_cc_var_logits_flow_q1_bce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n700-800_p0.15/mis_n700-800_p0.15_eb_var_logits_flow_q1_bce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n700-800_p0.15/mis_n700-800_p0.15_cc_var_logits_flow_velocity_mse.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/flow/mis_n700-800_p0.15/mis_n700-800_p0.15_eb_var_logits_flow_velocity_mse.json
