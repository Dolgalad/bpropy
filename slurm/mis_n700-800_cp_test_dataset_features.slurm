#!/bin/bash
#SBATCH --job-name=mis_n700-800_cp_test_dataset_features
#SBATCH --output=slurm_outputs/mis_n700-800_cp_test_dataset_features-%a.out
#SBATCH --error=slurm_outputs/mis_n700-800_cp_test_dataset_features-%a.err
#SBATCH --mail-type=end
#SBATCH --mail-user=schulz@lipn.univ-paris13.fr
#SBATCH --array=1-100
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH -A bnq@v100
#SBATCH --time=19:00:00
#SBATCH --exclusive

export PYTHONUNBUFFERED=TRUE

cd scripts
srun python mis_dataset_features.py $WORK/datasets_archive/mis_er_n700-800_p0.15_tl1000.0_pooltl1000.0/test $WORK/datasets/mis_er_n700-800_p0.15_tl1000.0_pooltl1000.0_cp_test/ CP $SLURM_ARRAY_TASK_ID 100
