#!/bin/bash


echo "===== MY JOBS (detailed) ====="
squeue -u $USER -o "%.18i %.9P %.20j %.8T %.10M %.6D %R"

echo -e "\n===== PRIORITY OF MY JOBS ====="
sprio -u $USER

echo -e "\n===== PARTITION STATUS (gpu_p13) ====="
sinfo -p gpu_p13 -o "%P %a %l %D %t %N"

echo -e "\n===== RUNNING JOBS IN gpu_p13 ====="
squeue -p gpu_p13 -t R -o "%.18i %.8u %.10M %.6D %R"

echo -e "\n===== PENDING JOBS AHEAD OF ME (gpu_p13) ====="
squeue -p gpu_p13 -t PD -o "%.18i %.8u %.10Q %.20j %R" | sort -k3 -nr | head -20

echo -e "\n===== CLUSTER FAIRSHARE ====="
sshare -u $USER

echo -e "\n===== ONE JOB FULL INFO ====="
scontrol show job 413608
