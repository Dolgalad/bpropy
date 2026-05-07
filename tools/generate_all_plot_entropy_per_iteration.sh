#!/bin/bash

set -euo pipefail

: "${WORK:?WORK must be set}"

CMAP="GNN EGG=#1f77b4,Tree MRF EGG=#ff7f0e,Tree MRF GD=#2ca02c,Pair MRF EGG=#d62728,Pair MRF GD=#9467bd,MRF EGG=#8c564b,MRF GD=#e377c2"
FORMAT=png

python tools/plot_gd_entropy_per_iteration.py \
  --gurobi-dir "$WORK/gurobi_solutions/gurobi_scp_n50_m50_k5_test" \
  --title "SCP(n=50, m=50, k=5)" \
  --output-prefix scp_n50_mixed_2 \
  --colors "$CMAP" \
  --output-format "$FORMAT" \
  --series "GNN EGG:EGG=$WORK/test_fix_and_solve/scp_n50_m50_k5/gnn" \
  --series "Tree MRF EGG:EGG=$WORK/test_fix_and_solve/scp_n50_m50_k5/gnn_lbp_tree" \
  --series "Tree MRF GD:GD=$WORK/test_fix_and_solve/scp_n50_m50_k5/gnn_lbp_tree" \
  --series "Pair MRF EGG:EGG=$WORK/test_fix_and_solve/scp_n50_m50_k5/gnn_lbp_pair" \
  --series "Pair MRF GD:GD=$WORK/test_fix_and_solve/scp_n50_m50_k5/gnn_lbp_pair" \
  --series "MRF EGG:EGG=$WORK/test_fix_and_solve/scp_n50_m50_k5/gnn_lbp_full" \
  --series "MRF GD:GD=$WORK/test_fix_and_solve/scp_n50_m50_k5/gnn_lbp_full"

python tools/plot_gd_entropy_per_iteration.py \
  --gurobi-dir "$WORK/gurobi_solutions/gurobi_scp_n120-180_m150-250_k5_test" \
  --title "SCP(n=[120,180], m=[150,250], k=5)" \
  --output-prefix scp_n120_mixed_2 \
  --colors "$CMAP" \
  --output-format "$FORMAT" \
  --series "GNN EGG:EGG=$WORK/test_fix_and_solve/scp_n120-180_m150-250_k5/gnn" \
  --series "Tree MRF EGG:EGG=$WORK/test_fix_and_solve/scp_n120-180_m150-250_k5/gnn_lbp_tree" \
  --series "Tree MRF GD:GD=$WORK/test_fix_and_solve/scp_n120-180_m150-250_k5/gnn_lbp_tree" \
  --series "Pair MRF EGG:EGG=$WORK/test_fix_and_solve/scp_n120-180_m150-250_k5/gnn_lbp_pair" \
  --series "Pair MRF GD:GD=$WORK/test_fix_and_solve/scp_n120-180_m150-250_k5/gnn_lbp_pair" \
  --series "MRF EGG:EGG=$WORK/test_fix_and_solve/scp_n120-180_m150-250_k5/gnn_lbp_full" \
  --series "MRF GD:GD=$WORK/test_fix_and_solve/scp_n120-180_m150-250_k5/gnn_lbp_full"

#python tools/generate_gd_entropy_per_iteration.py \
#  --gurobi-dir "$WORK/gurobi_solutions/gurobi_scp_n150-300_m150-300_k15_test" \
#  --title "SCP(n=[150,300], m=[150,300], k=15)" \
#  --output-prefix scp_n150_mixed_2 \
#  --colors "$CMAP" \
#  --exclude-seed0 \
#  --output-format "$FORMAT" \
#  --series "GNN EGG:GD=$WORK/test_fix_and_solve/scp_n150/gnn" \
#  --series "Tree MRF EGG:GD=$WORK/test_fix_and_solve/scp_n150/gnn_lbp_tree" \
#  --series "Tree MRF GD:GRD=$WORK/test_fix_and_solve/scp_n150/gnn_lbp_tree" \
#  --series "Pair MRF EGG:GD=$WORK/test_fix_and_solve/scp_n150/gnn_lbp_pair" \
#  --series "Pair MRF GD:GRD=$WORK/test_fix_and_solve/scp_n150/gnn_lbp_pair"
