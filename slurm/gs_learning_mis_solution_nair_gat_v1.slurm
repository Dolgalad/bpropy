#!/bin/bash
#SBATCH -A bnq@v100
#SBATCH --time=19:00:00
#SBATCH --job-name=gs_learning_mis_solution_nair_gat_v1
#SBATCH --output=slurm_outputs/gs_learning_mis_solution_nair_gat_v1/gs_learning_mis_solution_nair_gat_v1-%a.out
#SBATCH --error=slurm_outputs/gs_learning_mis_solution_nair_gat_v1/gs_learning_mis_solution_nair_gat_v1-%a.err
#SBATCH --array=1-9
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

echo SLURM_JOB_ID $SLURM_JOB_ID
echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID

srun python gs_learning_mis_solution_nair_gat_v1.py $WORK/datasets/mis $WORK/logs/mis $SLURM_ARRAY_TASK_ID
