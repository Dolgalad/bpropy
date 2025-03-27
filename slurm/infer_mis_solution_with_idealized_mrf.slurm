#!/usr/bin/env bash
#SBATCH --job-name=infer_mis_solution_with_idealized_mrf
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./slurm_outputs/infer_mis_solution_with_idealized_mrf.out
#SBATCH --error=./slurm_outputs/infer_mis_solution_with_idealized_mrf.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

cd ./scripts
srun python infer_mis_solutions_with_idealized_mrf.py /data1/schulz/datasets_archive/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_inmem_mrf_EB_test.pkl infer_mis_solution_with_idealized_mrf.csv
srun python infer_mis_solutions_with_idealized_mrf.py /data1/schulz/datasets_archive/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_inmem_mrf_CC_test.pkl infer_mis_solution_with_idealized_mrf.csv
srun python infer_mis_solutions_with_idealized_mrf.py /data1/schulz/datasets_archive/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_inmem_mrf_CF_test.pkl infer_mis_solution_with_idealized_mrf.csv
srun python infer_mis_solutions_with_idealized_mrf.py /data1/schulz/datasets_archive/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_inmem_mrf_EC_test.pkl infer_mis_solution_with_idealized_mrf.csv
srun python infer_mis_solutions_with_idealized_mrf.py /data1/schulz/datasets_archive/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_inmem_mrf_CP_test.pkl infer_mis_solution_with_idealized_mrf.csv

