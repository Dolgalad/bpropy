#!/bin/bash
#SBATCH -A bnq@v100
#SBATCH --time=19:00:00
#SBATCH --job-name=gs_learning_mis_solution_nair_loss_all_factors
#SBATCH --output=slurm_outputs/gs_learning_mis_solution_nair_loss_all_factors/gs_learning_mis_solution_nair_loss_all_factors-%a.out
#SBATCH --error=slurm_outputs/gs_learning_mis_solution_nair_loss_all_factors/gs_learning_mis_solution_nair_loss_all_factors-%a.err
#SBATCH --array=1-6
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

echo SLURM_JOB_ID $SLURM_JOB_ID
echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID

srun python gs_learning_mis_solution_nair_loss_all_factors.py $WORK/datasets/mis_mrf $WORK/logs/mis_mrf $SLURM_ARRAY_TASK_ID
