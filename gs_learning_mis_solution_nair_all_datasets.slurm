#!/bin/bash
#SBATCH --job-name=gs_learning_mis_solution_nair_all_datasets
#SBATCH --output=slurm_outputs/gs_learning_mis_solution_nair_all_datasets-%a.out
#SBATCH --error=slurm_outputs/gs_learning_mis_solution_nair_all_datasets-%a.err
#SBATCH --mail-type=end
#SBATCH --mail-user=schulz@lipn.univ-paris13.fr
#SBATCH --array=1-15
#SBATCH --ntasks=1
#SBATCH --gpus 1

echo SLURM_JOB_ID $SLURM_JOB_ID
echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID

srun python gs_learning_mis_solution_nair_all_datasets.py /data1/schulz/datasets/ /data1/schulz/models/mis /data1/schulz/logs/mis $SLURM_ARRAY_TASK_ID
