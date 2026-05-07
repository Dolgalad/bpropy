#!/bin/bash

## SCP(n=50,m=50,k=5)
#python bpropy/tools/generate_inference_theta_distribution.py \
#        --config $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n50_m50_k5/scp_n50_m50_k5_lbp_fce.json \
#        --checkpoint runs/scp_n50_m50_k5_lbp_fce_seed0/checkpoints/best.pt \
#        --output runs/inference_theta_distribution/scp_n50_m50_k5_lbp_fce.json
#
#python bpropy/tools/generate_inference_theta_distribution.py \
#        --config $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n50_m50_k5/scp_n50_m50_k5_nmf_fce.json \
#        --checkpoint runs/scp_n50_m50_k5_nmf_fce_seed0/checkpoints/best.pt \
#        --output runs/inference_theta_distribution/scp_n50_m50_k5_nmf_fce.json
# SCP(n=120-180,m=150-250,k=5)
python bpropy/tools/generate_inference_theta_distribution.py \
        --config $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_lbp_fce.json \
        --checkpoint runs/scp_n120-180_m150-250_k5_lbp_fce_seed0/checkpoints/best.pt \
        --output runs/inference_theta_distribution/scp_n120-180_m150-250_k5_lbp_fce.json

python bpropy/tools/generate_inference_theta_distribution.py \
        --config $HOME/bpropy-jz/bpropy/configs/train/scp/inference/scp_n120-180_m150-250_k5/scp_n120-180_m150-250_k5_nmf_fce.json \
        --checkpoint runs/scp_n120-180_m150-250_k5_nmf_fce_seed0/checkpoints/best.pt \
        --output runs/inference_theta_distribution/scp_n120-180_m150-250_k5_nmf_fce.json

## MIS(n=100,p=0.15)
#python bpropy/tools/generate_inference_theta_distribution.py \
#        --config $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n100_p0.15/mis_n100_p0.15_lbp_fce.json \
#        --checkpoint runs/mis_n100_p0.15_lbp_fce_seed0/checkpoints/best.pt \
#        --output runs/inference_theta_distribution/mis_n100_p0.15_lbp_fce.json
#
#python bpropy/tools/generate_inference_theta_distribution.py \
#        --config $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n100_p0.15/mis_n100_p0.15_nmf_fce.json \
#        --checkpoint runs/mis_n100_p0.15_nmf_fce_seed0/checkpoints/best.pt \
#        --output runs/inference_theta_distribution/mis_n100_p0.15_nmf_fce.json
# MIS(n=200-300,p=0.15)
python bpropy/tools/generate_inference_theta_distribution.py \
        --config $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n200-300_p0.15/mis_n200-300_p0.15_lbp_fce.json \
        --checkpoint runs/mis_n200-300_p0.15_lbp_fce_seed0/checkpoints/best.pt \
        --output runs/inference_theta_distribution/mis_n200-300_p0.15_lbp_fce.json

python bpropy/tools/generate_inference_theta_distribution.py \
        --config $HOME/bpropy-jz/bpropy/configs/train/mis/inference/mis_n200-300_p0.15/mis_n200-300_p0.15_nmf_fce.json \
        --checkpoint runs/mis_n200-300_p0.15_nmf_fce_seed0/checkpoints/best.pt \
        --output runs/inference_theta_distribution/mis_n200-300_p0.15_nmf_fce.json
