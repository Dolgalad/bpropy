#!/bin/bash
#SBATCH --job-name=test_nmf_park_all_datasets
##SBATCH --partition=<your-partition>
#SBATCH --gpus 1
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/test/test_nmf_park_all_datasets.out
#SBATCH --error=slurm_outputs/test/test_nmf_park_all_datasets.err
##SBATCH --exclusive
#SBATCH -A bnq@v100

set -euo pipefail
export PYTHONUNBUFFERED=TRUE

cd scripts/test
mkdir -p test_nmf_park_all_datasets

## SCP(n=50,m=50,k=5)
#mkdir -p test_nmf_park_all_datasets/scp
#srun python -u test_nmf_park.py "$WORK/datasets/scp/scp_n50_m50_k5" \
#	"$WORK/datasets/scp/scp_n50_m50_k5/processed_lbp" \
#	test_nmf_park_all_datasets/scp/scp_n50_m50_k5.csv \
#	--solver cbc

# SCP(scp_n80:120_m100:150_k5)
mkdir -p test_nmf_park_all_datasets/scp
srun python -u test_nmf_park.py "$WORK/datasets/scp/scp_n80:120_m100:150_k5" \
	"$WORK/datasets/scp/scp_n80:120_m100:150_k5/processed_lbp" \
	"test_nmf_park_all_datasets/scp/scp_n80:120_m100:150_k5.csv" \
	--solver cbc

## MIS(n=100,p=0.15)
#mkdir -p test_nmf_park_all_datasets/mis
#srun python -u test_nmf_park.py "$WORK/datasets/mis/mis_n100_p0.15" \
#	"$WORK/datasets/mis/mis_n100_p0.15/processed_lbp" \
#	"test_nmf_park_all_datasets/mis/mis_n100_p0.15.csv" \
#	--solver kamis
#
## MIS(n in [200,300],p=0.15)
#srun python -u test_nmf_park.py "$WORK/datasets/mis/mis_n200:300_p0.15_maxdeg15" \
#	"$WORK/datasets/mis/mis_n200:300_p0.15_maxdeg15/processed_lbp" \
#	"test_nmf_park_all_datasets/mis/mis_n200:300_p0.15_maxdeg15.csv" \
#	--solver kamis
#
## MIS(n in [700,800],p=0.15)
#srun python -u test_nmf_park.py "$WORK/datasets/mis/mis_n700:800_p0.15_maxdeg15" \
#	"$WORK/datasets/mis/mis_n700:800_p0.15_maxdeg15/processed_lbp" \
#	"test_nmf_park_all_datasets/mis/mis_n700:800_p0.15_maxdeg15.csv" \
#	--solver kamis
