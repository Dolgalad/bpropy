#!/usr/bin/env bash
#SBATCH --job-name=learning_mis_solution_edge_cover
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./learning_mis_solution_edge_cover.out
#SBATCH --error=./learning_mis_solution_edge_cover.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export PYTHONUNBUFFERED=TRUE
cd scripts
srun python learning_mis_solution_edge_cover.py /data1/schulz/datasets_archive/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_inmem_mrf_CP_train.pkl /data1/schulz/datasets_archive/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_inmem_mrf_CP_validation.pkl /data1/schulz/datasets_archive/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_inmem_mrf_CP_test.pkl /data1/schulz/logs/mis_nmf_edge_cover_gat_high_patience --model_args '{"mapping_aggr": "max", "inference": "naive_mean_field", "gnn_f": "GAT"}' -p 1000
