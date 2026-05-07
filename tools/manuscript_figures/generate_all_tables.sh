#!/bin/bash
BPROPY_ROOT="$HOME/bpropy-jz/bpropy"

# Chapter 3
echo "Chapter 3 / mrf_nair_inference_results"
python $BPROPY_ROOT/tools/manuscript_figures/generate_latex_results_table.py \
	--config $BPROPY_ROOT/configs/manuscript_resources/tables/chapter_3/mrf_nair_inference_results.json \
	--root-dir $HOME/bpropy-jz \
	--output $BPROPY_ROOT/resources/tab/chapter_3/mrf_nair_inference_results.tex

echo "Chapter 3 / mrf_nair_inference_grouped_bar_plot_optimality"
python $BPROPY_ROOT/tools/manuscript_figures/generate_latex_grouped_bar_figure.py \
	--config $BPROPY_ROOT/configs/manuscript_resources/tikz/chapter_3/mrf_nair_inference_grouped_bar_optimality.json \
	--root-dir $HOME/bpropy-jz \
	--output $BPROPY_ROOT/resources/tikz/chapter_3/mrf_nair_inference_grouped_bar_optimality.tex

echo "Chapter 3 / mrf_nair_inference_stacked_grouped_bar_plot_runtime"
python $BPROPY_ROOT/tools/manuscript_figures/generate_latex_stacked_grouped_bar_figure.py \
        --config $BPROPY_ROOT/configs/manuscript_resources/tikz/chapter_3/mrf_nair_inference_stacked_grouped_bar_runtime.json \
        --root-dir $HOME/bpropy-jz \
        --output $BPROPY_ROOT/resources/tikz/chapter_3/mrf_nair_inference_stacked_grouped_bar_runtime.tex

#echo "Chapter 3 / inference_freq_max_iter_ablation_pareto"
#python $BPROPY_ROOT/tools/manuscript_figures/generate_inference_freq_max_iter_ablation_pareto_figure.py \
#	--config $BPROPY_ROOT/configs/manuscript_resources/tikz/chapter_3/inference_freq_max_iter_ablation_pareto.json \
#	--output $BPROPY_ROOT/resources/tikz/chapter_3/inference_freq_max_iter_ablation_pareto.tex

echo "Chapter 3 / inference_freq_max_iter_ablation_pareto"
python $BPROPY_ROOT/tools/manuscript_figures/generate_inference_freq_max_iter_ablation_pareto_figure_grid.py \
	--config $BPROPY_ROOT/configs/manuscript_resources/tikz/chapter_3/inference_freq_max_iter_ablation_pareto.json \
	--output $BPROPY_ROOT/resources/tikz/chapter_3/inference_freq_max_iter_ablation_pareto.tex
echo "Chapter 3 / inference_theta_distribution"
python $BPROPY_ROOT/tools/manuscript_figures/generate_inference_theta_distribution_figure_grid.py \
        --config $BPROPY_ROOT/configs/manuscript_resources/tikz/chapter_3/inference_theta_distribution.json \
        --output $BPROPY_ROOT/resources/tikz/chapter_3/inference_theta_distribution.tex
echo "Chapter 3 / inference_test_nair_pareto"
python $BPROPY_ROOT/tools/manuscript_figures/generate_inference_test_nair_pareto.py \
	--config $BPROPY_ROOT/configs/manuscript_resources/tikz/chapter_3/inference_test_nair_pareto.json \
	--results-dir "$HOME/bpropy-jz/test_inference_fix_and_solve" \
	--gurobi-solutions-dir "$HOME/bpropy-jz/gurobi_solutions" \
        --output $BPROPY_ROOT/resources/tikz/chapter_3/inference_test_nair_pareto.tex
echo "Chapter 3 / inference_test_nair_gap_bar"
python $BPROPY_ROOT/tools/manuscript_figures/generate_inference_test_nair_gap_bar.py \
        --config "$BPROPY_ROOT/configs/manuscript_resources/tikz/chapter_3/inference_test_nair_gap_bar.json" \
        --results-dir "$HOME/bpropy-jz/test_inference_fix_and_solve" \
        --gurobi-solutions-dir "$HOME/bpropy-jz/gurobi_solutions" \
        --output "$BPROPY_ROOT/resources/tikz/chapter_3/inference_test_nair_gap_bar.tex"
echo "Chapter 3 / inference_test_nair_runtime_stacked_bar"
python $BPROPY_ROOT/tools/manuscript_figures/generate_inference_test_nair_runtime_stacked_bar.py \
    --config "$BPROPY_ROOT/configs/manuscript_resources/tikz/chapter_3/inference_test_nair_runtime_stacked_bar.json" \
    --results-dir "$HOME/bpropy-jz/test_inference_fix_and_solve" \
    --output "$BPROPY_ROOT/resources/tikz/chapter_3/inference_test_nair_runtime_stacked_bar.tex"
