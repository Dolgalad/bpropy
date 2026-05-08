#!/bin/bash

SUBMIT_SCRIPT="./tools/submit_train.sh"
BPROPY_CONFIG_ROOT="$HOME/bpropy-github-clean/configs/train"

$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/inference/scp_n50_m50_k5/scp_n50_m50_k5_lbp_fce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/inference/scp_n50_m50_k5/scp_n50_m50_k5_nmf_fce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/inference/scp_n50_m50_k5/scp_n50_m50_k5_var_logits_bce.json

$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/inference/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_lbp_fce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/inference/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_nmf_fce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/inference/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_var_logits_bce.json

#$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/inference/scp_n150-300_m150-300_k15/scp_n150-300_m150-300_k15_lbp_fce.json
#$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/inference/scp_n150-300_m150-300_k15/scp_n150-300_m150-300_k15_nmf_fce.json
#$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/scp/inference/scp_n150-300_m150-300_k15/scp_n150-300_m150-300_k15_var_logits_bce.json

$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/inference/mis_n100_p0.15/mis_n100_p0.15_lbp_fce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/inference/mis_n100_p0.15/mis_n100_p0.15_nmf_fce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/inference/mis_n100_p0.15/mis_n100_p0.15_var_logits_bce.json

$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/inference/mis_n200-300_p0.15/mis_n200-300_p0.15_lbp_fce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/inference/mis_n200-300_p0.15/mis_n200-300_p0.15_nmf_fce.json
$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/inference/mis_n200-300_p0.15/mis_n200-300_p0.15_var_logits_bce.json

#$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/inference/mis_n700-800_p0.15/mis_n700-800_p0.15_lbp_fce.json
#$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/inference/mis_n700-800_p0.15/mis_n700-800_p0.15_nmf_fce.json
#$SUBMIT_SCRIPT ${BPROPY_CONFIG_ROOT}/mis/inference/mis_n700-800_p0.15/mis_n700-800_p0.15_var_logits_bce.json
