#!/bin/bash
#SBATCH -A bnq@v100
#SBATCH --time=19:00:00
#SBATCH --job-name=gs_learning_mis_solution_nair_gcn_v2_gap0
#SBATCH --output=slurm_outputs/gs_learning_mis_solution_nair_gcn_v2_gap0/gs_learning_mis_solution_nair_gcn_v2_gap0-%a.out
#SBATCH --error=slurm_outputs/gs_learning_mis_solution_nair_gcn_v2_gap0/gs_learning_mis_solution_nair_gcn_v2_gap0-%a.err
#SBATCH --array=1-3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

echo SLURM_JOB_ID $SLURM_JOB_ID
echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID

srun python gs_learning_mis_solution_nair_gcn_v2_gap0.py $WORK/datasets/mis $WORK/logs/mis $SLURM_ARRAY_TASK_ID
