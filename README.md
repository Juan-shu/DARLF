DARLF (Disentangled Aging Representation Learning Framework) is a disease-aware bulk transcriptomics framework for biological age modeling. This repository highlights a compact training package together with a curated result bundle that demonstrates three key strengths of the model: accurate transcriptomic age estimation, strong health-state separation, and structured representations that remain interpretable after residual calibration.

## Scope

This repository is intentionally narrowed to the files that are directly useful for the DARLF training and result package:

- `src/aging_discovery/`: core package for data processing, model construction, training, and residual calibration
- `scripts/run_darlf_training_bundle.py`: launches the main DARLF training bundle with the retained structured-variant setup
- `scripts/analyze_darlf_significance.py`: recomputes paired significance statistics from saved result files
- `scripts/render_biology_manifold_program_figures.py`: redraws the biology/manifold summary figure from saved CSV tables
- `configs/darlf_disease_aware_joint_training_v1.yaml`: training configuration for the retained DARLF result package
- `results/`: selected tables, calibrated prediction summaries, statistical outputs, and final figure assets

## Installation

```bash
pip install -r requirements.txt
```

## Highlights

The preserved result snapshot emphasizes the main advantages of DARLF:

- calibrated age prediction with strong post-hoc error reduction
- disease-aware latent structure that separates healthy and diseased states with high AUROC and AUPRC
- interpretable pathway and manifold summaries from the saved `full` outputs
- paper-ready figures and statistical summaries preserved under `results/`

The result bundle includes:

- summary tables and an integrated text report
- full-model outputs needed for downstream reading, including calibrated predictions and manifold summaries
- comparison outputs for `no_health_supervision` and `no_disease_and_adversarial`
- figure assets, pathway summaries, and statistical outputs preserved under `results/`

## Usage

Run the retained DARLF training bundle:

```bash
python scripts/run_darlf_training_bundle.py --config configs/darlf_disease_aware_joint_training_v1.yaml --output-dir results/darlf_training_run
```

Recompute significance analysis on the saved bundle:

```bash
python scripts/analyze_darlf_significance.py
```

Redraw the biology/manifold summary figure from the saved `full` outputs:

```bash
python scripts/render_biology_manifold_program_figures.py --model-dir results/<result_bundle>/full --output-dir results/<result_bundle>/biology_manifold_program_figures
```


