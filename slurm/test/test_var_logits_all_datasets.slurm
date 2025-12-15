#!/bin/bash
#SBATCH --job-name=test_var_logits_all_datasets
##SBATCH --partition=<your-partition>
#SBATCH --gpus 1
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/test/test_var_logits_all_datasets.out
#SBATCH --error=slurm_outputs/test/test_var_logits_all_datasets.err
##SBATCH --exclusive
#SBATCH -A bnq@v100

set -euo pipefail
export PYTHONUNBUFFERED=TRUE

TESTDIR="test_var_logits_all_datasets"

cd scripts/test
mkdir -p $TESTDIR

## SCP(n=50,m=50,k=5)
#mkdir -p $TESTDIR/scp
#srun python -u test_var_logits.py "$WORK/datasets/scp/scp_n50_m50_k5" \
#	"$TESTDIR/scp/scp_n50_m50_k5.csv" \
#	"$WORK/logs/scp/scp_n50_m50_k5/var_logits_bce/HeteroGCNVarLogits_hetero_1757660049" \
#	"$WORK/logs/scp/scp_n50_m50_k5/var_logits_bce/HeteroGCNVarLogits_hetero_1757490155" \
#	"$WORK/logs/scp/scp_n50_m50_k5/var_logits_bce/HeteroGCNVarLogits_hetero_1757574334" \
#	--processed_dir "$WORK/datasets/scp/scp_n50_m50_k5/processed_pb" \
#	--solver cbc

# SCP(scp_n80:120_m100:150_k5)
mkdir -p $TESTDIR/scp
srun python -u test_var_logits.py "$WORK/datasets/scp/scp_n80:120_m100:150_k5" \
	"$TESTDIR/scp/scp_n80:120_m100:150_k5.csv" \
	"$WORK/logs/scp/scp_n50_m50_k5/var_logits_bce/HeteroGCNVarLogits_hetero_1757660049" \
	"$WORK/logs/scp/scp_n50_m50_k5/var_logits_bce/HeteroGCNVarLogits_hetero_1757490155" \
	"$WORK/logs/scp/scp_n50_m50_k5/var_logits_bce/HeteroGCNVarLogits_hetero_1757574334" \
	--processed_dir "$WORK/datasets/scp/scp_n80:120_m100:150_k5/processed_pb" \
	--solver cbc

# MIS(n=100,p=0.15) DONE
#mkdir -p $TESTDIR/mis
#srun python -u test_var_logits.py "$WORK/datasets/mis/mis_n100_p0.15" \
#	"$TESTDIR/mis/mis_n100_p0.15.csv" \
#	"$WORK/logs/mis/mis_n100_p0.15/var_logits_bce/HeteroGCNVarLogits_hetero_1757561337" \
#	"$WORK/logs/mis/mis_n100_p0.15/var_logits_bce/HeteroGCNVarLogits_hetero_1757561342" \
#	"$WORK/logs/mis/mis_n100_p0.15/var_logits_bce/HeteroGCNVarLogits_hetero_1757561419" \
#	--processed_dir "$WORK/datasets/mis/mis_n100_p0.15/processed_pb" \
#	--solver kamis
#
## MIS(n in [200,300],p=0.15) DONE
#srun python -u test_var_logits.py "$WORK/datasets/mis/mis_n200:300_p0.15_maxdeg15" \
#	"$TESTDIR/mis/mis_n200:300_p0.15_maxdeg15.csv" \
#	"$WORK/logs/mis/mis_n200:300_p0.15_maxdeg15/var_logits_bce/HeteroGCNVarLogits_hetero_1757561576" \
#	"$WORK/logs/mis/mis_n200:300_p0.15_maxdeg15/var_logits_bce/HeteroGCNVarLogits_hetero_1756793919" \
#	"$WORK/logs/mis/mis_n200:300_p0.15_maxdeg15/var_logits_bce/HeteroGCNVarLogits_hetero_1757561633" \
#	--processed_dir "$WORK/datasets/mis/mis_n200:300_p0.15_maxdeg15/processed_pb" \
#	--solver kamis

## MIS(n in [700,800],p=0.15)
#srun python -u test_var_logits.py "$WORK/datasets/mis/mis_n700:800_p0.15_maxdeg15" \
#	"$TESTDIR/mis/mis_n700:800_p0.15_maxdeg15.csv" \
#	--processed_dir "$WORK/datasets/mis/mis_n700:800_p0.15_maxdeg15/processed_pb" \
#	--solver kamis
