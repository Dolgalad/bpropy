#!/bin/bash
#SBATCH --job-name=gs_learning_mis_solution_nair_cc
#SBATCH --output=slurm_outputs/gs_learning_mis_solution_nair_cc-%a.out
#SBATCH --error=slurm_outputs/gs_learning_mis_solution_nair_cc-%a.err
#SBATCH --mail-type=end
#SBATCH --mail-user=schulz@lipn.univ-paris13.fr
#SBATCH --array=1-24
#SBATCH --ntasks=1
#SBATCH --gpus 1

echo SLURM_JOB_ID $SLURM_JOB_ID
echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID

srun python gs_learning_mis_solution_nair.py /data1/schulz/datasets/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_cc_train/ /data1/schulz/datasets/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_cc_validation/ MIS100_CC /data1/schulz/models/mis /data1/schulz/logs/mis $SLURM_ARRAY_TASK_ID
