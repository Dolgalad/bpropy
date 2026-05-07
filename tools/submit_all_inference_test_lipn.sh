#!/bin/bash

SUBMIT_SCRIPT="./tools/submit_test.sh"

# SCP(n=50,m=50,k=5)
$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n50_m50_k5/scp_n50_m50_k5_lbp_fce.json
$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n50_m50_k5/scp_n50_m50_k5_nmf_fce.json
$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n50_m50_k5/scp_n50_m50_k5_var_logits_bce.json

# SCP(n=120-180,m=150-250,k=5)
$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_lbp_fce.json
$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_nmf_fce.json
$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_var_logits_bce.json

# SCP(n=150-300,m=150-300,k=15)
#$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n150-300_m150-300_k15/scp_n150-300_m150-300_k15_lbp_fce.json
#$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n150-300_m150-300_k15/scp_n150-300_m150-300_k15_nmf_fce.json
#$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n150-300_m150-300_k15/scp_n150-300_m150-300_k15_var_logits_bce.json

# MIS(n=100,p=0.15)
$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n100_p0.15/mis_n100_p0.15_lbp_fce.json
$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n100_p0.15/mis_n100_p0.15_nmf_fce.json
$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n100_p0.15/mis_n100_p0.15_var_logits_bce.json

# MIS(n=200-300,p=0.15)
$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n200-300_p0.15/mis_n200-300_p0.15_lbp_fce.json
$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n200-300_p0.15/mis_n200-300_p0.15_nmf_fce.json
$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n200-300_p0.15/mis_n200-300_p0.15_var_logits_bce.json

# MIS(n=700-800,p=0.15)
#$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n700-800_p0.15/mis_n700-800_p0.15_lbp_fce.json
#$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n700-800_p0.15/mis_n700-800_p0.15_nmf_fce.json
#$SUBMIT_SCRIPT $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n700-800_p0.15/mis_n700-800_p0.15_var_logits_bce.json
