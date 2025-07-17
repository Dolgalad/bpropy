#!/usr/bin/env bash
#SBATCH --job-name=learning_mis_solution_mrf
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./learning_mis_solution_mrf.out
#SBATCH --error=./learning_mis_solution_mrf.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

srun python learning_mis_solution_mrf.py /data3/schulz/datasets/mis_er_n700-800_p0.15_tl1000.0_pooltl1000.0_mrf_indicators /data3/schulz/models/mis_er_n700-800_p0.15_tl1000.0_pooltl1000.0_mrf /data3/schulz/logs
