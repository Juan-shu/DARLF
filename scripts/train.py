from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aging_discovery.analysis import run_posthoc_analyses
from aging_discovery.ablation import (
    DiseaseAwareNoTissueModel,
    SharedOnlyAgingModel,
)
from aging_discovery.data import DiscoveryDataBundle, load_discovery_bundle
from aging_discovery.model import DisentangledAgingModel, ModelConfig
from aging_discovery.residual_calibration import run_residual_calibration
from aging_discovery.trainer import (
    AgingDataset,
    evaluate_model,
    predict_age_only,
    predict_dataset,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the disentangled multi-tissue aging discovery model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "default_run"),
        help="Directory for model checkpoints and outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Force device selection. Use 'cuda' to require GPU.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload, handle, ensure_ascii=False, indent=2
        )


def build_label_maps(
    metadata: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    metadata = metadata.copy()
    if "sex_group" not in metadata.columns:
        metadata["sex_group"] = metadata["sex"].map(
            lambda value: value
            if str(value).strip().lower() in {"male", "female"}
            else "unknown"
        )
    label_maps: dict[str, dict[str, int]] = {}
    for column in [
        "tissue_family",
        "disease_group",
        "dataset",
        "sex_group",
    ]:
        unique_values = sorted(
            metadata[column].astype(str).unique().tolist()
        )
        label_maps[column] = {
            value: idx for idx, value in enumerate(unique_values)
        }
    label_maps["health_label"] = {"Healthy": 0, "Disease": 1}
    return label_maps


def attach_label_indices(
    metadata: pd.DataFrame,
    label_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["sex_group"] = metadata["sex"].map(
        lambda value: value
        if str(value).strip().lower() in {"male", "female"}
        else "unknown"
    )
    metadata["tissue_idx"] = metadata["tissue_family"].map(
        label_maps["tissue_family"]
    )
    metadata["disease_idx"] = metadata["disease_group"].map(
        label_maps["disease_group"]
    )
    metadata["batch_idx"] = metadata["dataset"].map(
        label_maps["dataset"]
    )
    metadata["sex_idx"] = metadata["sex_group"].map(
        label_maps["sex_group"]
    )
    metadata["health_label"] = metadata["is_disease"].astype(int)
    return metadata


def split_train_val(
    train_mask: np.ndarray,
    metadata: pd.DataFrame,
    seed: int,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(train_mask)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if len(indices) < 4:
        val_size = max(1, len(indices) // 2)
    else:
        val_size = max(
            1, int(round(len(indices) * val_fraction))
        )
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        train_idx = val_idx[:1]
        val_idx = val_idx[1:]
    sub_train_mask = np.zeros(len(metadata), dtype=bool)
    sub_val_mask = np.zeros(len(metadata), dtype=bool)
    sub_train_mask[train_idx] = True
    sub_val_mask[val_idx] = True
    return sub_train_mask, sub_val_mask


def make_dataset(
    bundle: DiscoveryDataBundle,
    metadata: pd.DataFrame,
    mask: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> AgingDataset:
    subset = metadata.loc[mask].copy()
    feature_idx = subset.index.to_numpy()
    return AgingDataset(
        features=bundle.features[feature_idx],
        ages=subset["age"].to_numpy(dtype=np.float32),
        tissue_idx=subset["tissue_idx"].to_numpy(
            dtype=np.int64
        ),
        health_label=subset["health_label"].to_numpy(
            dtype=np.int64
        ),
        disease_idx=subset["disease_idx"].to_numpy(
            dtype=np.int64
        ),
        batch_idx=subset["batch_idx"].to_numpy(
            dtype=np.int64
        ),
        sex_idx=subset["sex_idx"].to_numpy(dtype=np.int64),
        sample_weight=sample_weight,
    )


def build_train_kwargs(
    config: dict,
    device: str,
    *,
    verbose: bool,
    log_prefix: str,
    overrides: dict | None = None,
) -> dict:
    overrides = overrides or {}
    age_weighting = config.get("age_weighting", {})
    return {
        "learning_rate": overrides.get(
            "learning_rate", config["optim"]["learning_rate"]
        ),
        "weight_decay": overrides.get(
            "weight_decay", config["optim"]["weight_decay"]
        ),
        "batch_size": overrides.get(
            "batch_size", config["optim"]["batch_size"]
        ),
        "max_epochs": overrides.get(
            "max_epochs", config["optim"]["max_epochs"]
        ),
        "patience": overrides.get(
            "patience", config["optim"]["patience"]
        ),
        "feature_weight": config["loss"]["feature_weight"],
        "health_weight": config["loss"]["health_weight"],
        "monotonic_weight": config["loss"]["monotonic_weight"],
        "orthogonality_weight": config["loss"][
            "orthogonality_weight"
        ],
        "deviation_weight": config["loss"]["deviation_weight"],
        "disease_weight": config["loss"].get("disease_weight", 0.4),
        "tissue_weight": config["loss"].get("tissue_weight", 0.25),
        "nuisance_batch_weight": config["loss"].get(
            "nuisance_batch_weight", 0.15
        ),
        "adversarial_batch_weight": config["loss"].get(
            "adversarial_batch_weight", 0.10
        ),
        "sex_weight": config["loss"].get("sex_weight", 0.0),
        "age_expert_supervision_weight": config["loss"].get(
            "age_expert_supervision_weight", 0.0
        ),
        "residual_weight": config["loss"].get(
            "residual_weight", 0.0
        ),
        "seed": config["run"]["seed"],
        "device": device,
        "age_only_finetune_epochs": overrides.get(
            "age_only_finetune_epochs",
            config["optim"].get("age_only_finetune_epochs", 0),
        ),
        "age_only_lr_scale": overrides.get(
            "age_only_lr_scale",
            config["optim"].get("age_only_lr_scale", 0.25),
        ),
        "disease_age_weight": config["loss"].get(
            "disease_age_weight", 1.0
        ),
        "low_age_threshold": age_weighting.get(
            "low_age_threshold"
        ),
        "high_age_threshold": age_weighting.get(
            "high_age_threshold"
        ),
        "extreme_age_weight": age_weighting.get(
            "extreme_age_weight", 1.0
        ),
        "use_amp": overrides.get(
            "use_amp", config["run"].get("use_amp", False)
        ),
        "selection_metric": overrides.get(
            "selection_metric",
            config["optim"].get("selection_metric", "val_loss"),
        ),
        "lr_scheduler_patience": overrides.get(
            "lr_scheduler_patience",
            config["optim"].get("lr_scheduler_patience", 0),
        ),
        "lr_scheduler_factor": overrides.get(
            "lr_scheduler_factor",
            config["optim"].get("lr_scheduler_factor", 0.5),
        ),
        "min_learning_rate": overrides.get(
            "min_learning_rate",
            config["optim"].get("min_learning_rate", 1e-5),
        ),
        "adversarial_warmup_epochs": overrides.get(
            "adversarial_warmup_epochs",
            config["optim"].get("adversarial_warmup_epochs", 0),
        ),
        "orthogonality_warmup_epochs": overrides.get(
            "orthogonality_warmup_epochs",
            config["optim"].get("orthogonality_warmup_epochs", 0),
        ),
        "tissue_warmup_epochs": overrides.get(
            "tissue_warmup_epochs",
            config["optim"].get("tissue_warmup_epochs", 0),
        ),
        "verbose": verbose,
        "log_prefix": log_prefix,
    }


def build_focused_sample_weights(
    metadata: pd.DataFrame,
    config: dict,
) -> tuple[np.ndarray, dict[str, float | int | dict[str, float]]]:
    focused_cfg = config.get("focused_reweight", {})
    weights = np.ones(len(metadata), dtype=np.float32)
    summary: dict[str, float | int | dict[str, float]] = {
        "enabled": bool(focused_cfg.get("enabled", False)),
        "n_samples": int(len(metadata)),
    }
    if not focused_cfg.get("enabled", False):
        summary["mean_weight"] = 1.0
        return weights, summary

    ages = metadata["age"].to_numpy(dtype=np.float32)
    high_age_threshold = focused_cfg.get("high_age_threshold")
    high_age_weight = float(
        focused_cfg.get("high_age_weight", 1.0)
    )
    if high_age_threshold is not None and abs(high_age_weight - 1.0) > 1e-6:
        high_age_mask = ages >= float(high_age_threshold)
        weights *= np.where(
            high_age_mask, high_age_weight, 1.0
        ).astype(np.float32)
        summary["high_age_threshold"] = float(high_age_threshold)
        summary["high_age_weight"] = high_age_weight
        summary["high_age_fraction"] = float(high_age_mask.mean())

    disease_group_weights = {
        str(key): float(value)
        for key, value in focused_cfg.get(
            "disease_group_weights", {}
        ).items()
    }
    disease_groups = metadata["disease_group"].astype(str)
    for group_name, group_weight in disease_group_weights.items():
        if abs(group_weight - 1.0) <= 1e-6:
            continue
        group_mask = disease_groups.eq(group_name).to_numpy()
        if not np.any(group_mask):
            continue
        weights[group_mask] *= np.float32(group_weight)
    if disease_group_weights:
        summary["disease_group_weights"] = disease_group_weights

    tissue_family_weights = {
        str(key): float(value)
        for key, value in focused_cfg.get(
            "tissue_family_weights", {}
        ).items()
    }
    tissue_families = metadata["tissue_family"].astype(str)
    for tissue_name, tissue_weight in tissue_family_weights.items():
        if abs(tissue_weight - 1.0) <= 1e-6:
            continue
        tissue_mask = tissue_families.eq(tissue_name).to_numpy()
        if not np.any(tissue_mask):
            continue
        weights[tissue_mask] *= np.float32(tissue_weight)
    if tissue_family_weights:
        summary["tissue_family_weights"] = tissue_family_weights

    summary["mean_weight"] = float(weights.mean())
    summary["max_weight"] = float(weights.max())
    summary["min_weight"] = float(weights.min())
    summary["fraction_upweighted"] = float((weights > 1.001).mean())
    return weights, summary


def build_residual_sample_weights(
    dataset: AgingDataset,
    age_pred: np.ndarray,
    residual_cfg: dict,
) -> tuple[np.ndarray, pd.DataFrame]:
    age_true = dataset.ages.detach().cpu().numpy().astype(np.float32)
    health_label = (
        dataset.health_label.detach().cpu().numpy().astype(np.int64)
    )
    residual = np.abs(age_pred.astype(np.float32) - age_true)
    weights = np.ones_like(residual, dtype=np.float32)

    def apply_group_weights(
        mask: np.ndarray,
        threshold: float,
        floor_weight: float,
        decay_window: float,
        drop_threshold: float | None,
    ) -> None:
        if not np.any(mask):
            return
        group_residual = residual[mask]
        group_weight = np.ones_like(group_residual, dtype=np.float32)
        excess = np.clip(
            group_residual - float(threshold), a_min=0.0, a_max=None
        )
        if float(decay_window) > 1e-6:
            group_weight = np.maximum(
                float(floor_weight),
                1.0 - excess / float(decay_window),
            ).astype(np.float32)
        else:
            group_weight = np.where(
                group_residual > float(threshold),
                np.float32(floor_weight),
                np.float32(1.0),
            )
        if drop_threshold is not None:
            group_weight = np.where(
                group_residual > float(drop_threshold),
                np.float32(0.0),
                group_weight,
            )
        weights[mask] = group_weight

    healthy_mask = health_label == 0
    disease_mask = health_label == 1
    apply_group_weights(
        healthy_mask,
        threshold=residual_cfg.get(
            "healthy_residual_threshold", 18.0
        ),
        floor_weight=residual_cfg.get("healthy_floor_weight", 0.5),
        decay_window=residual_cfg.get("healthy_decay_window", 12.0),
        drop_threshold=residual_cfg.get("healthy_drop_threshold"),
    )
    apply_group_weights(
        disease_mask,
        threshold=residual_cfg.get(
            "disease_residual_threshold", 15.0
        ),
        floor_weight=residual_cfg.get("disease_floor_weight", 0.25),
        decay_window=residual_cfg.get("disease_decay_window", 10.0),
        drop_threshold=residual_cfg.get("disease_drop_threshold"),
    )

    if residual_cfg.get("normalize_mean", True):
        positive_mean = float(weights.mean())
        if positive_mean > 1e-6:
            weights = weights / positive_mean

    frame = pd.DataFrame(
        {
            "age_true": age_true,
            "age_pred_warmup": age_pred.astype(np.float32),
            "abs_residual_warmup": residual,
            "health_label": health_label,
            "sample_weight": weights,
            "is_downweighted": (weights < 0.999).astype(np.int64),
            "is_zero_weight": (weights <= 1e-8).astype(np.int64),
        }
    )
    return weights.astype(np.float32), frame


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["run"]["seed"]))

    bundle = load_discovery_bundle(
        data_root=config["data"]["data_root"],
        min_samples_per_tissue=config["data"][
            "min_samples_per_tissue"
        ],
        normalization=config["data"]["normalization"],
        test_fraction=config["split"]["test_fraction"],
        seed=config["run"]["seed"],
        split_mode=config["split"].get(
            "mode", "group_by_dataset"
        ),
        top_hvgs=config["data"].get("top_hvgs", 2500),
        add_rank_features=config["data"].get(
            "add_rank_features", True
        ),
        add_pathway_scores=config["data"].get(
            "add_pathway_scores", True
        ),
        add_cell_scores=config["data"].get(
            "add_cell_scores", True
        ),
        age_corr_weight=config["data"].get(
            "age_corr_weight", 2.0
        ),
        within_tissue_age_weight=config["data"].get(
            "within_tissue_age_weight", 2.5
        ),
        tissue_specificity_penalty=config["data"].get(
            "tissue_specificity_penalty", 1.5
        ),
        cell_marker_penalty=config["data"].get(
            "cell_marker_penalty", 1.0
        ),
        exclude_cell_marker_genes=config["data"].get(
            "exclude_cell_marker_genes", False
        ),
        min_gene_detection_rate=config["data"].get(
            "min_gene_detection_rate", 0.05
        ),
        min_tissue_samples_for_corr=config["data"].get(
            "min_tissue_samples_for_corr", 25
        ),
        regress_out_cell_scores=config["data"].get(
            "regress_out_cell_scores", False
        ),
        cell_residual_ridge=config["data"].get(
            "cell_residual_ridge", 1e-4
        ),
        n_gene_pcs=config["data"].get(
            "n_gene_pcs", 0
        ),
        include_tissues=config["data"].get("include_tissues", []),
    )
    label_maps = build_label_maps(bundle.metadata)
    metadata = attach_label_indices(
        bundle.metadata, label_maps
    )
    train_mask, val_mask = split_train_val(
        bundle.train_mask,
        metadata,
        seed=config["run"]["seed"],
        val_fraction=config["split"]["val_fraction"],
    )
    test_mask = bundle.test_mask

    train_weights, train_weight_summary = build_focused_sample_weights(
        metadata.loc[train_mask],
        config,
    )
    val_weights, val_weight_summary = build_focused_sample_weights(
        metadata.loc[val_mask],
        config,
    )
    test_weights, test_weight_summary = build_focused_sample_weights(
        metadata.loc[test_mask],
        config,
    )

    train_dataset = make_dataset(
        bundle, metadata, train_mask, sample_weight=train_weights
    )
    val_dataset = make_dataset(
        bundle, metadata, val_mask, sample_weight=val_weights
    )
    test_dataset = make_dataset(
        bundle, metadata, test_mask, sample_weight=test_weights
    )

    save_json(
        output_dir / "focused_reweight_summary.json",
        {
            "train": train_weight_summary,
            "validation": val_weight_summary,
            "test": test_weight_summary,
        },
    )

    targeted_tissue_names = [
        str(name)
        for name in config["model"].get(
            "targeted_tissue_residual_tissues", []
        )
    ]
    targeted_tissue_indices = tuple(
        label_maps["tissue_family"][name]
        for name in targeted_tissue_names
        if name in label_maps["tissue_family"]
    )
    model_config = ModelConfig(
        input_dim=bundle.features.shape[1],
        n_tissues=len(label_maps["tissue_family"]),
        n_disease_groups=len(label_maps["disease_group"]),
        n_batches=len(label_maps["dataset"]),
        n_health_classes=len(label_maps["health_label"]),
        n_sexes=len(label_maps["sex_group"]),
        hidden_dim=config["model"]["hidden_dim"],
        shared_dim=config["model"]["shared_dim"],
        tissue_dim=config["model"]["tissue_dim"],
        disease_dim=config["model"]["disease_dim"],
        nuisance_dim=config["model"]["nuisance_dim"],
        dropout=config["model"]["dropout"],
        grl_lambda=config["model"]["grl_lambda"],
        n_pathways=len(bundle.pathway_names),
        n_age_experts=config["model"].get("n_age_experts", 1),
        encoder_layers=config["model"].get("encoder_layers", 3),
        use_sex_condition=config["model"].get(
            "use_sex_condition", False
        ),
        targeted_tissue_indices=targeted_tissue_indices,
        use_joint_residual_branch=config["model"].get(
            "use_joint_residual_branch", False
        ),
        residual_branch_hidden_dim=config["model"].get(
            "residual_branch_hidden_dim", 0
        ),
        use_separate_targeted_tissue_heads=config["model"].get(
            "use_separate_targeted_tissue_heads", False
        ),
        use_tissue_in_age_head=config["model"].get(
            "use_tissue_in_age_head", True
        ),
        use_disease_in_age_head=config["model"].get(
            "use_disease_in_age_head", False
        ),
        raw_context_dim=config["model"].get("raw_context_dim", 0),
        use_raw_context_in_classifiers=config["model"].get(
            "use_raw_context_in_classifiers", False
        ),
        use_raw_context_in_residual=config["model"].get(
            "use_raw_context_in_residual", False
        ),
        architecture=config["model"].get(
            "architecture", "disentangled"
        ),
    )
    requested_device = (
        args.device.strip().lower()
        if args.device.strip()
        else str(config["run"]["device"]).strip().lower()
    )
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available in this environment."
            )
        device = "cuda"
    elif requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    print(
        json.dumps(
            {
                "device": device,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
            },
            ensure_ascii=False,
        )
        ,
        flush=True,
    )
    residual_cfg = config.get("residual_reweight", {})
    residual_summary: dict[str, float | int | bool] | None = None
    if residual_cfg.get("enabled", False):
        warmup_overrides = residual_cfg.get("warmup_optim", {})
        warmup_overrides = {
            "max_epochs": warmup_overrides.get("max_epochs", 80),
            "patience": warmup_overrides.get("patience", 20),
            "age_only_finetune_epochs": warmup_overrides.get(
                "age_only_finetune_epochs", 0
            ),
            "learning_rate": warmup_overrides.get(
                "learning_rate", config["optim"]["learning_rate"]
            ),
            "selection_metric": warmup_overrides.get(
                "selection_metric", "val_mae"
            ),
            "lr_scheduler_patience": warmup_overrides.get(
                "lr_scheduler_patience",
                config["optim"].get("lr_scheduler_patience", 0),
            ),
            "use_amp": warmup_overrides.get(
                "use_amp", config["run"].get("use_amp", False)
            ),
        }
        warmup_model = DisentangledAgingModel(model_config)
        warmup_artifacts = train_model(
            warmup_model,
            train_dataset,
            val_dataset,
            **build_train_kwargs(
                config,
                device,
                verbose=True,
                log_prefix="reweight-warmup",
                overrides=warmup_overrides,
            ),
        )
        warmup_model.load_state_dict(warmup_artifacts.model_state)
        warmup_age_pred = predict_age_only(
            warmup_model,
            train_dataset,
            batch_size=config["optim"]["batch_size"],
            device=device,
        )
        train_weights, train_weight_frame = (
            build_residual_sample_weights(
                train_dataset,
                warmup_age_pred,
                residual_cfg,
            )
        )
        base_train_weights = (
            train_dataset.sample_weight.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        final_train_weights = base_train_weights * train_weights
        train_dataset.set_sample_weights(final_train_weights)
        train_weight_frame["sample_weight_base"] = base_train_weights
        train_weight_frame["sample_weight_residual"] = train_weights
        train_weight_frame["sample_weight_final"] = final_train_weights
        train_weight_frame.to_csv(
            output_dir / "train_sample_weights.csv", index=False
        )
        residual_summary = {
            "enabled": True,
            "warmup_best_epoch": int(warmup_artifacts.best_epoch),
            "mean_sample_weight": float(final_train_weights.mean()),
            "fraction_downweighted": float(
                (final_train_weights < 0.999).mean()
            ),
            "fraction_zero_weight": float(
                (final_train_weights <= 1e-8).mean()
            ),
            "healthy_residual_threshold": float(
                residual_cfg.get("healthy_residual_threshold", 18.0)
            ),
            "disease_residual_threshold": float(
                residual_cfg.get("disease_residual_threshold", 15.0)
            ),
        }
        save_json(
            output_dir / "residual_reweight_summary.json",
            residual_summary,
        )

    architecture = str(
        config.get("model", {}).get("architecture", "disentangled")
    ).strip().lower()
    if architecture == "shared_only":
        model = SharedOnlyAgingModel(model_config)
    elif architecture == "disease_aware_no_tissue":
        model = DiseaseAwareNoTissueModel(model_config)
    elif architecture == "disentangled":
        model = DisentangledAgingModel(model_config)
    else:
        raise ValueError(
            f"Unsupported model architecture: {architecture}"
        )
    artifacts = train_model(
        model,
        train_dataset,
        val_dataset,
        **build_train_kwargs(
            config,
            device,
            verbose=True,
            log_prefix="train",
        ),
    )

    model.load_state_dict(artifacts.model_state)
    val_metrics = evaluate_model(
        model,
        val_dataset,
        batch_size=config["optim"]["batch_size"],
        device=device,
    )
    test_metrics = evaluate_model(
        model,
        test_dataset,
        batch_size=config["optim"]["batch_size"],
        device=device,
    )
    train_predictions = predict_dataset(
        model,
        train_dataset,
        metadata.loc[train_mask],
        batch_size=config["optim"]["batch_size"],
        device=device,
        feature_matrix=bundle.features[metadata.loc[train_mask].index.to_numpy()],
        feature_names=bundle.preprocessor.feature_names_,
    )
    val_predictions = predict_dataset(
        model,
        val_dataset,
        metadata.loc[val_mask],
        batch_size=config["optim"]["batch_size"],
        device=device,
        feature_matrix=bundle.features[metadata.loc[val_mask].index.to_numpy()],
        feature_names=bundle.preprocessor.feature_names_,
    )
    test_predictions = predict_dataset(
        model,
        test_dataset,
        metadata.loc[test_mask],
        batch_size=config["optim"]["batch_size"],
        device=device,
        feature_matrix=bundle.features[metadata.loc[test_mask].index.to_numpy()],
        feature_names=bundle.preprocessor.feature_names_,
    )
    all_predictions = pd.concat(
        [
            train_predictions,
            val_predictions,
            test_predictions,
        ],
        axis=0,
        ignore_index=True,
    )
    calibration_result = run_residual_calibration(
        train_predictions,
        val_predictions,
        test_predictions,
        output_dir=output_dir,
        config=config.get("residual_calibration", {}),
    )
    if calibration_result is not None:
        save_json(
            output_dir / "residual_calibration_summary.json",
            calibration_result.summary,
        )

    torch.save(
        artifacts.model_state, output_dir / "model.pt"
    )
    pd.DataFrame(artifacts.history).to_csv(
        output_dir / "history.csv", index=False
    )
    metadata.to_csv(
        output_dir / "metadata_used.csv", index=False
    )
    train_predictions.to_csv(
        output_dir / "train_predictions.csv", index=False
    )
    val_predictions.to_csv(
        output_dir / "val_predictions.csv", index=False
    )
    test_predictions.to_csv(
        output_dir / "test_predictions.csv", index=False
    )
    all_predictions.to_csv(
        output_dir / "all_predictions.csv", index=False
    )
    save_json(
        output_dir / "metrics.json",
        (
            {
                "validation": val_metrics,
                "test": test_metrics,
                "best_epoch": artifacts.best_epoch,
            }
            if calibration_result is None
            else {
                "validation": val_metrics,
                "validation_calibrated": {
                    **{
                        key: value
                        for key, value in val_metrics.items()
                        if key not in {"mae", "r2"}
                    },
                    "mae": calibration_result.calibrated_val_mae,
                    "r2": float(
                        r2_score(
                            calibration_result.val_predictions[
                                "age"
                            ].to_numpy(dtype=float),
                            calibration_result.val_predictions[
                                "age_pred_calibrated"
                            ].to_numpy(dtype=float),
                        )
                    ),
                },
                "test": test_metrics,
                "test_calibrated": {
                    **{
                        key: value
                        for key, value in test_metrics.items()
                        if key not in {"mae", "r2"}
                    },
                    "mae": calibration_result.calibrated_test_mae,
                    "r2": float(
                        r2_score(
                            calibration_result.test_predictions[
                                "age"
                            ].to_numpy(dtype=float),
                            calibration_result.test_predictions[
                                "age_pred_calibrated"
                            ].to_numpy(dtype=float),
                        )
                    ),
                },
                "residual_calibration": calibration_result.summary,
                "best_epoch": artifacts.best_epoch,
            }
        ),
    )
    save_json(
        output_dir / "feature_state.json",
        bundle.preprocessor.state_dict(),
    )
    save_json(
        output_dir / "label_maps.json", label_maps
    )
    save_json(
        output_dir / "manifest.json",
        {
            "n_samples": int(len(metadata)),
            "n_features": int(bundle.features.shape[1]),
            "n_tissues": int(
                len(label_maps["tissue_family"])
            ),
            "n_disease_groups": int(
                len(label_maps["disease_group"])
            ),
            "n_health_classes": int(
                len(label_maps["health_label"])
            ),
            "n_sexes": int(len(label_maps["sex_group"])),
            "n_batches": int(
                len(label_maps["dataset"])
            ),
            "pathway_names": bundle.pathway_names,
            "cell_score_names": bundle.cell_score_names,
            "age_weighting": config.get("age_weighting", {}),
            "focused_reweight": config.get("focused_reweight", {}),
            "residual_calibration": config.get(
                "residual_calibration", {}
            ),
            "targeted_tissue_residual_tissues": targeted_tissue_names,
            "use_separate_targeted_tissue_heads": bool(
                config["model"].get(
                    "use_separate_targeted_tissue_heads", False
                )
            ),
            "use_joint_residual_branch": bool(
                config["model"].get(
                    "use_joint_residual_branch", False
                )
            ),
            "residual_reweight_enabled": bool(
                residual_cfg.get("enabled", False)
            ),
        },
    )
    if residual_summary is not None:
        save_json(
            output_dir / "residual_reweight_summary.json",
            residual_summary,
        )
    save_json(
        output_dir / "benchmark_summary.json",
        {
            "device": device,
            "use_amp": bool(config["run"].get("use_amp", False))
            and device == "cuda",
            "training_seconds": artifacts.training_seconds,
            "mean_epoch_seconds": artifacts.mean_epoch_seconds,
            "train_examples_per_second": (
                artifacts.train_examples_per_second
            ),
            "batch_size": int(config["optim"]["batch_size"]),
            "selection_metric": str(
                config["optim"].get(
                    "selection_metric", "val_loss"
                )
            ),
            "lr_scheduler_patience": int(
                config["optim"].get(
                    "lr_scheduler_patience", 0
                )
            ),
            "min_learning_rate": float(
                config["optim"].get(
                    "min_learning_rate", 1e-5
                )
            ),
            "n_train_samples": int(train_mask.sum()),
            "n_val_samples": int(val_mask.sum()),
            "n_test_samples": int(test_mask.sum()),
            "n_features": int(bundle.features.shape[1]),
            "best_epoch": int(artifacts.best_epoch),
        },
    )
    posthoc_summary = run_posthoc_analyses(
        train_predictions=train_predictions,
        eval_predictions=pd.concat(
            [val_predictions, test_predictions],
            axis=0,
            ignore_index=True,
        ),
        output_dir=output_dir,
        pathway_names=bundle.pathway_names,
    )
    save_json(
        output_dir / "posthoc_summary.json",
        posthoc_summary,
    )

    print(
        "Validation metrics:",
        json.dumps(
            val_metrics, ensure_ascii=False, indent=2
        ),
        flush=True,
    )
    print(
        "Test metrics:",
        json.dumps(
            test_metrics, ensure_ascii=False, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
