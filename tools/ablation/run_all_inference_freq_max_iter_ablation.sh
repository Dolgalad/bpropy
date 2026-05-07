#!/bin/bash

# MIS (n=100, p=0.15)
for i in {0..2}; do 
	bpropy-ablation-inference-freq-max-iter \
		--config bpropy/configs/ablation/inference_freq_max_iter_ablation/mis_n100_p0.15_lbp_fce_seed$i.json \
		--bpropy-root $HOME/bpropy-jz/bpropy/ \
		--max-instances 20
done

for i in {0..2}; do 
	bpropy-ablation-inference-freq-max-iter \
		--config bpropy/configs/ablation/inference_freq_max_iter_ablation/mis_n100_p0.15_nmf_fce_seed$i.json \
		--bpropy-root $HOME/bpropy-jz/bpropy/ \
		--max-instances 20
done

# MIS (n=200-300, p=0.15)
for i in {0..2}; do 
	bpropy-ablation-inference-freq-max-iter \
		--config bpropy/configs/ablation/inference_freq_max_iter_ablation/mis_n200-300_p0.15_lbp_fce_seed$i.json \
		--bpropy-root $HOME/bpropy-jz/bpropy/ \
		--max-instances 20
done

for i in {0..2}; do 
	bpropy-ablation-inference-freq-max-iter \
		--config bpropy/configs/ablation/inference_freq_max_iter_ablation/mis_n200-300_p0.15_nmf_fce_seed$i.json \
		--bpropy-root $HOME/bpropy-jz/bpropy/ \
		--max-instances 20
done

# SCP(n=50, m=50, k=5)
for i in {0..2}; do 
	bpropy-ablation-inference-freq-max-iter \
		--config bpropy/configs/ablation/inference_freq_max_iter_ablation/scp_n50_m50_k5_lbp_fce_seed$i.json \
		--bpropy-root $HOME/bpropy-jz/bpropy/ \
		--max-instances 20
done

for i in {0..2}; do 
	bpropy-ablation-inference-freq-max-iter \
		--config bpropy/configs/ablation/inference_freq_max_iter_ablation/scp_n50_m50_k5_nmf_fce_seed$i.json \
		--bpropy-root $HOME/bpropy-jz/bpropy/ \
		--max-instances 20
done

# SCP(n=120-180, m=150-250, k=5)
for i in {0..2}; do 
	bpropy-ablation-inference-freq-max-iter \
		--config bpropy/configs/ablation/inference_freq_max_iter_ablation/scp_n120-180_m150-250_k5_lbp_fce_seed$i.json \
		--bpropy-root $HOME/bpropy-jz/bpropy/ \
		--max-instances 20
done

for i in {0..2}; do 
	bpropy-ablation-inference-freq-max-iter \
		--config bpropy/configs/ablation/inference_freq_max_iter_ablation/scp_n120-180_m150-250_k5_nmf_fce_seed$i.json \
		--bpropy-root $HOME/bpropy-jz/bpropy/ \
		--max-instances 20
done

