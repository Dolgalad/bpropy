#!/bin/bash
#SBATCH --job-name=mis_n700-800_eb_test_idealized_mrf
#SBATCH --output=slurm_outputs/mis_n700-800_eb_test_idealized_mrf.out
#SBATCH --error=slurm_outputs/mis_n700-800_eb_test_idealized_mrf.err
#SBATCH --mail-type=end
#SBATCH --mail-user=schulz@lipn.univ-paris13.fr
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH -A bnq@v100
#SBATCH --time=19:00:00

cd scripts
srun python infer_mis_solutions_with_idealized_mrf.py $WORK/datasets/mis/mis_er_n700-800_p0.15_tl1000.0_pooltl1000.0_eb_test/ infer_mis_solutions_with_idealized_mrf_n700-800.csv 10000
