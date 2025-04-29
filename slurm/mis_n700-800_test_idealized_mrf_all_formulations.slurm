#!/bin/bash
#SBATCH --job-name=mis_n700-800_test_idealized_mrf_all_formulations
#SBATCH --output=slurm_outputs/mis_n700-800_test_idealized_mrf_all_formulations-%a.out
#SBATCH --error=slurm_outputs/mis_n700-800_test_idealized_mrf_all_formulations-%a.err
#SBATCH --mail-type=end
#SBATCH --mail-user=schulz@lipn.univ-paris13.fr
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH -A bnq@v100
#SBATCH --time=10:00:00
#SBATCH --array=1-100
#SBATCH --exclusive

export PYTHONUNBUFFERED=TRUE
cd scripts
srun python infer_mis_solutions_with_idealized_mrf_with_prep.py $WORK/datasets_archive/mis_er_n700-800_p0.15_tl1000.0_pooltl1000.0_split.tar.gz infer_mis_solutions_with_idealized_mrf_n700-800 $SLURM_ARRAY_TASK_ID 100 100
