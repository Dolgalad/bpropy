#!/bin/bash
#SBATCH --job-name=gs_learning_mis_solution_nair_gcn_v2
#SBATCH --output=slurm_outputs/gs_learning_mis_solution_nair_gcn_v2/gs_learning_mis_solution_nair_gcn_v2-%a.out
#SBATCH --error=slurm_outputs/gs_learning_mis_solution_nair_gcn_v2/gs_learning_mis_solution_nair_gcn_v2-%a.err
#SBATCH --mail-type=end
#SBATCH --mail-user=schulz@lipn.univ-paris13.fr
#SBATCH --array=1-54
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=lipn-rtx1

echo SLURM_JOB_ID $SLURM_JOB_ID
echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID

srun python gs_learning_mis_solution_nair_gcn_v2.py /data3/schulz/datasets/mis /data3/schulz/logs/mis $SLURM_ARRAY_TASK_ID
