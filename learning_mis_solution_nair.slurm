#!/bin/bash
#SBATCH -A bnq@v100
#SBATCH --time=19:00:00
#SBATCH --job-name=learning_mis_solution_nair
#SBATCH --output=slurm_outputs/learning_mis_solution_nair.out
#SBATCH --error=slurm_outputs/learning_mis_solution_nair.err
#SBATCH --mail-type=end
#SBATCH --mail-user=schulz@lipn.univ-paris13.fr
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

srun python learning_mis_solution_nair.py $WORK/datasets/mis MIS100_EB $WORK/logs/mis -D '{"unravel_pool":false, "solution_limit":1, "aggregate":false, "gap_limit": 0.0}' -M GCN_v3 -A '{"hidden_dim": 64, "dropout_p": 0.25, "n_layers": 4}' --debug 


