# Shared defaults for all jobs
#export SLURM_DEF_PARTITION="gpu"
export SLURM_DEF_QOS="qos_gpu"              # empty means “don’t set”
export SLURM_DEF_ACCOUNT="bnq@v100"
export SLURM_DEF_TIME="20:00:00"
export SLURM_DEF_CPUS_PER_TASK="8"
#export SLURM_DEF_MEM="32G"
export SLURM_DEF_GPUS="1"
#export SLURM_DEF_CONSTRAINT=""
export SLURM_DEF_JOB_NAME="bpropy_train"
export SLURM_DEF_OUTPUT="slurm/out/%x_%j.out"
export SLURM_DEF_ERROR="slurm/err/%x_%j.err"

export SEEDS_JSON="[2023,2024,2025]"
