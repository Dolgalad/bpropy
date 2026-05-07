#!/bin/bash

CMAP="GNN EGG=#1f77b4,Tree MRF EGG=#ff7f0e,Tree MRF GD=#2ca02c,Pair MRF EGG=#d62728,Pair MRF GD=#9467bd,MRF EGG=#8c564b,MRF GD=#e377c2"
FORMAT=png

python tools/generate_scp_plots.py --title "SCP(n=50, m=50, k=5)" --output-prefix scp_n50_mixed --colors "$CMAP" --output-format $FORMAT \
	--series "GNN EGG:GD=./analysis_plot_data/scp_n50/gnn" \
	--series "Tree MRF EGG:GD=./analysis_plot_data/scp_n50/gnn_lbp_tree" \
	--series "Tree MRF GD:GRD=./analysis_plot_data/scp_n50/gnn_lbp_tree" \
	--series "Pair MRF EGG:GD=./analysis_plot_data/scp_n50/gnn_lbp_pair" \
	--series "Pair MRF GD:GRD=./analysis_plot_data/scp_n50/gnn_lbp_pair" \
	--series "MRF EGG:GD=./analysis_plot_data/scp_n50/gnn_lbp_full" \
	--series "MRF GD:GRD=./analysis_plot_data/scp_n50/gnn_lbp_full" 

python tools/generate_scp_plots.py --title "SCP(n=[120,180], m=[150,250], k=5)" --output-prefix scp_n120_mixed --colors "$CMAP" --output-format $FORMAT \
	--series "GNN EGG:GD=./analysis_plot_data/scp_n120/gnn"  \
	--series "Tree MRF EGG:GD=./analysis_plot_data/scp_n120/gnn_lbp_tree" \
	--series "Tree MRF GD:GRD=./analysis_plot_data/scp_n120/gnn_lbp_tree" \
	--series "Pair MRF EGG:GD=./analysis_plot_data/scp_n120/gnn_lbp_pair" \
	--series "Pair MRF GD:GRD=./analysis_plot_data/scp_n120/gnn_lbp_pair" \
	--series "MRF EGG:GD=./analysis_plot_data/scp_n120/gnn_lbp_full" \
	--series "MRF GD:GRD=./analysis_plot_data/scp_n120/gnn_lbp_full"

python tools/generate_scp_plots.py --title "SCP(n=[150,300], m=[150,300], k=15)" --output-prefix scp_n150_mixed --colors "$CMAP" --exclude-seed0 --output-format $FORMAT \
	--series "GNN EGG:GD=./analysis_plot_data/scp_n150/gnn" \
	--series "Tree MRF EGG:GD=./analysis_plot_data/scp_n150/gnn_lbp_tree" \
	--series "Tree MRF GD:GRD=./analysis_plot_data/scp_n150/gnn_lbp_tree" \
	--series "Pair MRF EGG:GD=./analysis_plot_data/scp_n150/gnn_lbp_pair" \
	--series "Pair MRF GD:GRD=./analysis_plot_data/scp_n150/gnn_lbp_pair"

