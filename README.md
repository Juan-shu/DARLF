DARLF (`Disentangled Aging Representation Learning Framework`) is a trimmed repository focused on one ablation result bundle: `ablation_health_binary_agefirst_reusefull_v1_run_01`. The repository keeps the core training package, the ablation configuration, a small set of analysis scripts, and the result files needed to inspect the calibrated age-prediction and disease-aware ablation findings.

## Scope

This repository is intentionally narrowed to the files that are directly useful for the selected ablation study:

- `src/aging_discovery/`: core package required by the ablation runner
- `scripts/run_ablation.py`: reruns the ablation experiment
- `scripts/analyze_ablation_significance.py`: recomputes paired significance analysis from saved result files
- `scripts/render_biology_manifold_program_figures.py`: redraws the biology/manifold summary figure from saved CSV tables
- `configs/ablation_health_binary_agefirst_reusefull_v1.yaml`: ablation configuration used for this bundle
- `results/ablation_health_binary_agefirst_reusefull_v1_run_01/`: selected tables, calibrated prediction summaries, statistical outputs, and final figure assets

## Installation

```bash
pip install -r requirements.txt
```

## Result Bundle

The preserved result snapshot includes:

- summary tables: `ablation_formal_table.csv`, `ablation_summary.json`, `DETAILED_RESULTS_REPORT.txt`
- full-model outputs needed for downstream reading: `metrics.json`, `posthoc_summary.json`, `eval_predictions_with_manifold_distance.csv`, `test_predictions_calibrated.csv`
- ablation-variant comparison files for `no_health_supervision` and `no_disease_and_adversarial`
- figure assets in `nature_ablation_figures/`, `biology_manifold_program_figures/`, and `significance_analysis/`

## Usage

Re-run the ablation:

```bash
python scripts/run_ablation.py --config configs/ablation_health_binary_agefirst_reusefull_v1.yaml --output-dir results/reproduced_ablation
```

Recompute significance analysis on the saved bundle:

```bash
python scripts/analyze_ablation_significance.py
```

Redraw the biology/manifold summary figure from the saved `full` outputs:

```bash
python scripts/render_biology_manifold_program_figures.py --model-dir results/ablation_health_binary_agefirst_reusefull_v1_run_01/full --output-dir results/ablation_health_binary_agefirst_reusefull_v1_run_01/biology_manifold_program_figures
```

