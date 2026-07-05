from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .analysis import (
    compute_disease_deviation_ranking,
    compute_healthy_manifold_deviation,
    compute_tissue_pace_summary,
    run_posthoc_analyses,
)
from .model import (
    DisentangledAgingModel,
    DisentangledModelNoBatch,
    DisentangledModelNoDisease,
    DisentangledModelNoMonotonic,
    GradientReversal,
    MLP,
    ModelConfig,
    build_encoder,
)
from .residual_calibration import run_residual_calibration
from .trainer import (
    AgingDataset,
    evaluate_model,
    predict_dataset,
    set_seed,
    train_model,
)


class SharedOnlyAgingModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = build_encoder(
            config.input_dim,
            config.hidden_dim,
            config.dropout,
            config.encoder_layers,
        )
        self.shared_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.shared_dim,
            config.dropout,
        )
        self.age_head = MLP(
            config.shared_dim,
            max(64, config.hidden_dim // 2),
            1,
            config.dropout,
        )
        self.shared_age_head = MLP(
            config.shared_dim,
            max(64, config.hidden_dim // 2),
            1,
            config.dropout,
        )

    def forward(
        self,
        x: Tensor,
        _tissue_idx: Tensor,
        _sex_idx: Tensor | None = None,
    ) -> dict[str, Tensor]:
        hidden = self.encoder(x)
        z_shared = self.shared_proj(hidden)
        age_pred = self.age_head(z_shared).squeeze(1)
        return {
            "z_shared": z_shared,
            "age_pred": age_pred,
            "shared_age_pred": self.shared_age_head(z_shared).squeeze(1),
        }


class DiseaseAwareNoTissueModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = build_encoder(
            config.input_dim,
            config.hidden_dim,
            config.dropout,
            config.encoder_layers,
        )
        self.shared_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.shared_dim,
            config.dropout,
        )
        self.disease_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.disease_dim,
            config.dropout,
        )
        self.nuisance_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.nuisance_dim,
            config.dropout,
        )
        self.use_sex_condition = bool(
            config.use_sex_condition and config.n_sexes > 1
        )
        self.use_disease_in_age_head = bool(
            config.use_disease_in_age_head
        )
        if self.use_sex_condition:
            self.sex_embedding = nn.Embedding(
                config.n_sexes,
                min(8, max(3, config.shared_dim // 16)),
            )
            sex_embed_dim = self.sex_embedding.embedding_dim
            self.sex_classifier = MLP(
                config.shared_dim + config.disease_dim,
                config.hidden_dim // 2,
                config.n_sexes,
                config.dropout,
            )
        else:
            self.sex_embedding = None
            sex_embed_dim = 0
            self.sex_classifier = None
        age_input_dim = config.shared_dim
        if self.use_disease_in_age_head:
            age_input_dim += config.disease_dim
            if self.use_sex_condition:
                age_input_dim += sex_embed_dim
        self.age_head = MLP(
            age_input_dim,
            config.hidden_dim,
            1,
            config.dropout,
        )
        self.shared_age_head = MLP(
            config.shared_dim,
            max(64, config.hidden_dim // 2),
            1,
            config.dropout,
        )
        disease_input_dim = config.disease_dim + sex_embed_dim
        self.health_classifier = MLP(
            disease_input_dim,
            config.hidden_dim // 2,
            config.n_health_classes,
            config.dropout,
        )
        self.nuisance_batch_classifier = MLP(
            config.nuisance_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )
        self.grl = GradientReversal(config.grl_lambda)
        self.adversarial_batch_classifier = MLP(
            config.shared_dim + config.disease_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )

    def forward(
        self,
        x: Tensor,
        _tissue_idx: Tensor,
        sex_idx: Tensor | None = None,
    ) -> dict[str, Tensor]:
        hidden = self.encoder(x)
        z_shared = self.shared_proj(hidden)
        z_disease = self.disease_proj(hidden)
        z_nuisance = self.nuisance_proj(hidden)
        disease_inputs = [z_disease]
        if self.use_sex_condition:
            if sex_idx is None:
                sex_idx = torch.zeros(
                    z_shared.shape[0],
                    dtype=torch.long,
                    device=z_shared.device,
                )
            disease_inputs.append(self.sex_embedding(sex_idx))
        disease_input = torch.cat(disease_inputs, dim=1)
        age_inputs = [z_shared]
        if self.use_disease_in_age_head:
            age_inputs.append(z_disease)
            if self.use_sex_condition:
                age_inputs.append(disease_inputs[-1])
        age_input = (
            torch.cat(age_inputs, dim=1)
            if len(age_inputs) > 1
            else z_shared
        )
        outputs: dict[str, Tensor] = {
            "z_shared": z_shared,
            "z_disease": z_disease,
            "z_nuisance": z_nuisance,
            "age_pred": self.age_head(age_input).squeeze(1),
            "shared_age_pred": self.shared_age_head(z_shared).squeeze(1),
            "health_logits": self.health_classifier(disease_input),
            "nuisance_batch_logits": self.nuisance_batch_classifier(
                z_nuisance
            ),
            "adversarial_batch_logits": self.adversarial_batch_classifier(
                self.grl(torch.cat([z_shared, z_disease], dim=1))
            ),
        }
        if self.sex_classifier is not None:
            outputs["sex_logits"] = self.sex_classifier(
                torch.cat([z_shared, z_disease], dim=1)
            )
        return outputs


class NoDiseaseNoAdversarialModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = build_encoder(
            config.input_dim,
            config.hidden_dim,
            config.dropout,
            config.encoder_layers,
        )
        self.shared_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.shared_dim,
            config.dropout,
        )
        self.tissue_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.tissue_dim,
            config.dropout,
        )
        self.nuisance_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.nuisance_dim,
            config.dropout,
        )
        self.tissue_embedding = nn.Embedding(
            config.n_tissues,
            min(32, max(8, config.tissue_dim)),
        )
        age_input_dim = (
            config.shared_dim
            + config.tissue_dim
            + self.tissue_embedding.embedding_dim
        )
        self.age_head = MLP(
            age_input_dim, config.hidden_dim, 1, config.dropout
        )
        self.shared_age_head = MLP(
            config.shared_dim,
            max(64, config.hidden_dim // 2),
            1,
            config.dropout,
        )
        self.tissue_classifier = MLP(
            config.tissue_dim,
            config.hidden_dim // 2,
            config.n_tissues,
            config.dropout,
        )
        self.nuisance_batch_classifier = MLP(
            config.nuisance_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )

    def forward(
        self,
        x: Tensor,
        tissue_idx: Tensor,
        _sex_idx: Tensor | None = None,
    ) -> dict[str, Tensor]:
        hidden = self.encoder(x)
        z_shared = self.shared_proj(hidden)
        z_tissue = self.tissue_proj(hidden)
        z_nuisance = self.nuisance_proj(hidden)
        tissue_embed = self.tissue_embedding(tissue_idx)
        age_input = torch.cat(
            [z_shared, z_tissue, tissue_embed], dim=1
        )
        return {
            "z_shared": z_shared,
            "z_tissue": z_tissue,
            "z_nuisance": z_nuisance,
            "age_pred": self.age_head(age_input).squeeze(1),
            "shared_age_pred": self.shared_age_head(z_shared).squeeze(1),
            "tissue_logits": self.tissue_classifier(z_tissue),
            "nuisance_batch_logits": self.nuisance_batch_classifier(
                z_nuisance
            ),
        }


@dataclass(frozen=True)
class AblationSpec:
    key: str
    label: str
    builder: Callable[[ModelConfig], nn.Module]
    train_overrides: dict[str, Any] | None = None


def _clone_config(
    model_config: ModelConfig, **overrides: Any
) -> ModelConfig:
    return replace(model_config, **overrides)


def _build_full_model(model_config: ModelConfig) -> nn.Module:
    if model_config.architecture == "disease_aware_no_tissue":
        return DiseaseAwareNoTissueModel(model_config)
    if model_config.architecture == "shared_only":
        return SharedOnlyAgingModel(model_config)
    return DisentangledAgingModel(model_config)


def _build_no_disease_model(model_config: ModelConfig) -> nn.Module:
    return DisentangledModelNoDisease(
        _clone_config(model_config, use_sex_condition=False)
    )


def _build_no_batch_model(model_config: ModelConfig) -> nn.Module:
    return DisentangledModelNoBatch(
        _clone_config(model_config, use_sex_condition=False)
    )


def _build_no_disease_and_adversarial_model(
    model_config: ModelConfig,
) -> nn.Module:
    return NoDiseaseNoAdversarialModel(
        _clone_config(model_config, use_sex_condition=False)
    )


def _build_no_health_supervision_model(
    model_config: ModelConfig,
) -> nn.Module:
    return DiseaseAwareNoTissueModel(model_config)


def _build_no_monotonic_model(model_config: ModelConfig) -> nn.Module:
    return DisentangledModelNoMonotonic(
        _clone_config(model_config, use_sex_condition=False)
    )


def _build_no_tissue_model(model_config: ModelConfig) -> nn.Module:
    return DiseaseAwareNoTissueModel(model_config)


def _build_no_disentanglement_model(
    model_config: ModelConfig,
) -> nn.Module:
    return SharedOnlyAgingModel(
        _clone_config(model_config, use_sex_condition=False)
    )


def _build_no_joint_residual_model(
    model_config: ModelConfig,
) -> nn.Module:
    return DisentangledAgingModel(
        _clone_config(model_config, use_joint_residual_branch=False)
    )


def _build_no_targeted_residual_model(
    model_config: ModelConfig,
) -> nn.Module:
    return DisentangledAgingModel(
        _clone_config(
            model_config,
            use_separate_targeted_tissue_heads=False,
        )
    )


def _build_no_sex_conditioning_model(
    model_config: ModelConfig,
) -> nn.Module:
    return DisentangledAgingModel(
        _clone_config(model_config, use_sex_condition=False)
    )


ABLATION_VARIANTS: dict[str, AblationSpec] = {
    "full": AblationSpec(
        key="full",
        label="Full",
        builder=_build_full_model,
    ),
    "no_disentanglement": AblationSpec(
        key="no_disentanglement",
        label="No Disentanglement",
        builder=_build_no_disentanglement_model,
        train_overrides={
            "health_weight": 0.0,
            "disease_weight": 0.0,
            "tissue_weight": 0.0,
            "nuisance_batch_weight": 0.0,
            "adversarial_batch_weight": 0.0,
            "sex_weight": 0.0,
            "residual_weight": 0.0,
            "orthogonality_weight": 0.0,
            "deviation_weight": 0.0,
        },
    ),
    "no_tissue_specific": AblationSpec(
        key="no_tissue_specific",
        label="No Tissue-Specific",
        builder=_build_no_tissue_model,
        train_overrides={
            "tissue_weight": 0.0,
            "residual_weight": 0.0,
        },
    ),
    "no_disease": AblationSpec(
        key="no_disease",
        label="No Disease",
        builder=_build_no_disease_model,
        train_overrides={
            "health_weight": 0.0,
            "disease_weight": 0.0,
            "deviation_weight": 0.0,
            "sex_weight": 0.0,
            "residual_weight": 0.0,
        },
    ),
    "no_disease_and_adversarial": AblationSpec(
        key="no_disease_and_adversarial",
        label="No Disease & Adversarial",
        builder=_build_no_disease_and_adversarial_model,
        train_overrides={
            "health_weight": 0.0,
            "disease_weight": 0.0,
            "deviation_weight": 0.0,
            "nuisance_batch_weight": 0.01,
            "adversarial_batch_weight": 0.0,
            "sex_weight": 0.0,
            "residual_weight": 0.0,
        },
    ),
    "no_health_supervision": AblationSpec(
        key="no_health_supervision",
        label="No Health Supervision",
        builder=_build_no_health_supervision_model,
        train_overrides={
            "health_weight": 0.0,
            "disease_weight": 0.0,
            "deviation_weight": 0.0,
        },
    ),
    "no_adversarial_batch": AblationSpec(
        key="no_adversarial_batch",
        label="No Adversarial Batch",
        builder=_build_no_batch_model,
        train_overrides={
            "nuisance_batch_weight": 0.0,
            "adversarial_batch_weight": 0.0,
            "sex_weight": 0.0,
            "residual_weight": 0.0,
        },
    ),
    "no_monotonic": AblationSpec(
        key="no_monotonic",
        label="No Monotonic",
        builder=_build_no_monotonic_model,
        train_overrides={
            "monotonic_weight": 0.0,
            "sex_weight": 0.0,
            "residual_weight": 0.0,
        },
    ),
    "no_joint_residual": AblationSpec(
        key="no_joint_residual",
        label="No Joint Residual",
        builder=_build_no_joint_residual_model,
        train_overrides={},
    ),
    "no_targeted_tissue_residual": AblationSpec(
        key="no_targeted_tissue_residual",
        label="No Targeted Tissue Residual",
        builder=_build_no_targeted_residual_model,
        train_overrides={},
    ),
    "no_sex_conditioning": AblationSpec(
        key="no_sex_conditioning",
        label="No Sex Conditioning",
        builder=_build_no_sex_conditioning_model,
        train_overrides={"sex_weight": 0.0},
    ),
}

FORMAL_ABLATION_COLUMNS = [
    "variant",
    "variant_key",
    "val_mae",
    "val_health_accuracy",
    "val_health_auroc",
    "val_health_auprc",
    "test_mae",
    "test_calibrated_mae",
    "test_r2",
    "test_health_accuracy",
    "test_health_auroc",
    "test_health_auprc",
    "healthy_manifold_deviation_gap",
    "tissue_pace_spread",
    "best_epoch",
    "training_seconds",
]


def _write_formal_ablation_tables(
    output_dir: Path,
    results: dict[str, dict[str, float | str]],
) -> None:
    if not results:
        return
    summary_df = pd.DataFrame(results.values())
    for column in FORMAL_ABLATION_COLUMNS:
        if column not in summary_df.columns:
            summary_df[column] = np.nan
    formal_df = summary_df[FORMAL_ABLATION_COLUMNS].copy()
    formal_df.to_csv(
        output_dir / "ablation_formal_table.csv", index=False
    )
    md_df = formal_df.copy()
    numeric_cols = [
        col
        for col in md_df.columns
        if col not in {"variant", "variant_key"}
    ]
    for col in numeric_cols:
        md_df[col] = md_df[col].map(
            lambda x: ""
            if pd.isna(x)
            else f"{float(x):.4f}"
            if abs(float(x)) < 1000
            else f"{float(x):.1f}"
        )
    header = "| " + " | ".join(md_df.columns.tolist()) + " |"
    divider = "| " + " | ".join(["---"] * len(md_df.columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in md_df.itertuples(index=False, name=None)
    ]
    lines = [
        "# Ablation Formal Table",
        "",
        header,
        divider,
        *body,
        "",
    ]
    (output_dir / "ablation_formal_table.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _device_from_config(config: dict) -> str:
    requested = str(config["run"].get("device", "auto")).strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _copy_directory_contents(
    source_dir: Path, target_dir: Path
) -> None:
    for path in source_dir.iterdir():
        destination = target_dir / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)


def _load_reused_full_metrics(
    source_dir: Path,
) -> dict[str, float | str]:
    metrics_path = source_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing metrics.json in reused full dir: {source_dir}"
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    benchmark_summary = {}
    benchmark_path = source_dir / "benchmark_summary.json"
    if benchmark_path.exists():
        benchmark_summary = json.loads(
            benchmark_path.read_text(encoding="utf-8")
        )
    combined: dict[str, float | str] = {
        "variant": "Full",
        "variant_key": "full",
    }
    validation = payload.get("validation", {})
    test = payload.get("test", {})
    validation_calibrated = payload.get("validation_calibrated", {})
    test_calibrated = payload.get("test_calibrated", {})
    for key, value in validation.items():
        combined[f"val_{key}"] = value
    for key, value in test.items():
        combined[f"test_{key}"] = value
    if "mae" in validation_calibrated:
        combined["val_calibrated_mae"] = validation_calibrated["mae"]
    if "mae" in test_calibrated:
        combined["test_calibrated_mae"] = test_calibrated["mae"]
    if "best_epoch" in benchmark_summary:
        combined["best_epoch"] = float(benchmark_summary["best_epoch"])
    if "training_seconds" in benchmark_summary:
        combined["training_seconds"] = float(
            benchmark_summary["training_seconds"]
        )
    if "mean_epoch_seconds" in benchmark_summary:
        combined["mean_epoch_seconds"] = float(
            benchmark_summary["mean_epoch_seconds"]
        )
    if "train_examples_per_second" in benchmark_summary:
        combined["train_examples_per_second"] = float(
            benchmark_summary["train_examples_per_second"]
        )
    return combined


def _build_train_kwargs(
    config: dict,
    device: str,
    log_prefix: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    age_weighting = config.get("age_weighting", {})
    return {
        "learning_rate": config["optim"]["learning_rate"],
        "weight_decay": config["optim"]["weight_decay"],
        "batch_size": config["optim"]["batch_size"],
        "max_epochs": config["optim"]["max_epochs"],
        "patience": config["optim"]["patience"],
        "feature_weight": overrides.get(
            "feature_weight", config["loss"]["feature_weight"]
        ),
        "health_weight": overrides.get(
            "health_weight", config["loss"]["health_weight"]
        ),
        "monotonic_weight": overrides.get(
            "monotonic_weight", config["loss"]["monotonic_weight"]
        ),
        "orthogonality_weight": overrides.get(
            "orthogonality_weight",
            config["loss"]["orthogonality_weight"],
        ),
        "deviation_weight": overrides.get(
            "deviation_weight", config["loss"]["deviation_weight"]
        ),
        "disease_weight": overrides.get(
            "disease_weight", config["loss"].get("disease_weight", 0.4)
        ),
        "tissue_weight": overrides.get(
            "tissue_weight", config["loss"].get("tissue_weight", 0.25)
        ),
        "nuisance_batch_weight": overrides.get(
            "nuisance_batch_weight",
            config["loss"].get("nuisance_batch_weight", 0.15),
        ),
        "adversarial_batch_weight": overrides.get(
            "adversarial_batch_weight",
            config["loss"].get("adversarial_batch_weight", 0.10),
        ),
        "sex_weight": overrides.get(
            "sex_weight", config["loss"].get("sex_weight", 0.0)
        ),
        "age_expert_supervision_weight": overrides.get(
            "age_expert_supervision_weight",
            config["loss"].get("age_expert_supervision_weight", 0.0),
        ),
        "residual_weight": overrides.get(
            "residual_weight",
            config["loss"].get("residual_weight", 0.0),
        ),
        "seed": config["run"]["seed"],
        "device": device,
        "age_only_finetune_epochs": config["optim"].get(
            "age_only_finetune_epochs", 0
        ),
        "age_only_lr_scale": config["optim"].get(
            "age_only_lr_scale", 0.25
        ),
        "disease_age_weight": config["loss"].get(
            "disease_age_weight", 1.0
        ),
        "low_age_threshold": age_weighting.get("low_age_threshold"),
        "high_age_threshold": age_weighting.get("high_age_threshold"),
        "extreme_age_weight": age_weighting.get(
            "extreme_age_weight", 1.0
        ),
        "use_amp": config["run"].get("use_amp", False),
        "selection_metric": config["optim"].get(
            "selection_metric", "val_loss"
        ),
        "lr_scheduler_patience": config["optim"].get(
            "lr_scheduler_patience", 0
        ),
        "lr_scheduler_factor": config["optim"].get(
            "lr_scheduler_factor", 0.5
        ),
        "min_learning_rate": config["optim"].get(
            "min_learning_rate", 1e-5
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
        "verbose": True,
        "log_prefix": log_prefix,
    }


def _compute_per_tissue_mae(
    predictions: pd.DataFrame,
    pred_col: str,
) -> pd.DataFrame:
    if predictions.empty or pred_col not in predictions.columns:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["abs_err"] = (
        frame[pred_col].to_numpy(dtype=float)
        - frame["age"].to_numpy(dtype=float)
    )
    frame["abs_err"] = frame["abs_err"].abs()
    return (
        frame.groupby("tissue_family", dropna=False)["abs_err"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values(["mean", "count"], ascending=[False, False])
    )


def _compute_discovery_metrics(
    train_predictions: pd.DataFrame,
    eval_predictions: pd.DataFrame,
) -> dict[str, float]:
    eval_with_distance, _ = compute_healthy_manifold_deviation(
        train_predictions,
        eval_predictions,
    )
    tissue_pace = compute_tissue_pace_summary(
        pd.concat(
            [train_predictions, eval_predictions],
            axis=0,
            ignore_index=True,
        )
    )
    disease_rank = compute_disease_deviation_ranking(
        eval_with_distance
    )
    if eval_with_distance.empty:
        healthy_gap = float("nan")
    else:
        healthy_mean = float(
            eval_with_distance.loc[
                eval_with_distance["health_status"].eq("Healthy"),
                "healthy_manifold_distance",
            ].mean()
        )
        disease_mean = float(
            eval_with_distance.loc[
                eval_with_distance["health_status"].ne("Healthy"),
                "healthy_manifold_distance",
            ].mean()
        )
        healthy_gap = (
            disease_mean - healthy_mean
            if np.isfinite(disease_mean) and np.isfinite(healthy_mean)
            else float("nan")
        )
    return {
        "healthy_manifold_deviation_gap": healthy_gap,
        "tissue_pace_spread": float(
            tissue_pace["shared_age_slope"].std()
        )
        if not tissue_pace.empty
        else float("nan"),
        "n_tissue_pace_groups": float(len(tissue_pace))
        if not tissue_pace.empty
        else 0.0,
        "n_disease_groups_ranked": float(len(disease_rank))
        if not disease_rank.empty
        else 0.0,
    }


def run_ablation_study(
    config: dict,
    model_config: ModelConfig,
    train_dataset: AgingDataset,
    val_dataset: AgingDataset,
    test_dataset: AgingDataset,
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    train_feature_matrix: np.ndarray | None = None,
    val_feature_matrix: np.ndarray | None = None,
    test_feature_matrix: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    variants: list[str] | None = None,
    output_dir: str | Path = "outputs/ablation",
) -> dict[str, dict[str, float | str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if variants is None:
        variants = list(ABLATION_VARIANTS.keys())
    device = _device_from_config(config)
    results: dict[str, dict[str, float | str]] = {}
    ablation_cfg = config.get("ablation", {})
    reused_full_dir_cfg = ablation_cfg.get("reuse_full_from_dir")
    reused_full_dir = (
        Path(reused_full_dir_cfg)
        if str(reused_full_dir_cfg or "").strip()
        else None
    )

    for variant_key in variants:
        spec = ABLATION_VARIANTS.get(variant_key)
        if spec is None:
            continue
        variant_dir = output_dir / spec.key
        variant_dir.mkdir(parents=True, exist_ok=True)
        if (
            spec.key == "full"
            and reused_full_dir is not None
            and reused_full_dir.exists()
        ):
            _copy_directory_contents(reused_full_dir, variant_dir)
            combined = _load_reused_full_metrics(reused_full_dir)
            results[spec.key] = combined
            with (variant_dir / "metrics.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(combined, handle, ensure_ascii=False, indent=2)
            continue
        set_seed(int(config["run"]["seed"]))
        model = spec.builder(model_config)
        artifacts = train_model(
            model,
            train_dataset,
            val_dataset,
            **_build_train_kwargs(
                config=config,
                device=device,
                log_prefix=spec.key,
                overrides=spec.train_overrides,
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
            train_metadata.reset_index(drop=True),
            batch_size=config["optim"]["batch_size"],
            device=device,
            feature_matrix=train_feature_matrix,
            feature_names=feature_names,
        )
        val_predictions = predict_dataset(
            model,
            val_dataset,
            val_metadata.reset_index(drop=True),
            batch_size=config["optim"]["batch_size"],
            device=device,
            feature_matrix=val_feature_matrix,
            feature_names=feature_names,
        )
        test_predictions = predict_dataset(
            model,
            test_dataset,
            test_metadata.reset_index(drop=True),
            batch_size=config["optim"]["batch_size"],
            device=device,
            feature_matrix=test_feature_matrix,
            feature_names=feature_names,
        )
        pd.DataFrame(artifacts.history).to_csv(
            variant_dir / "history.csv", index=False
        )
        train_predictions.to_csv(
            variant_dir / "train_predictions.csv", index=False
        )
        val_predictions.to_csv(
            variant_dir / "val_predictions.csv", index=False
        )
        test_predictions.to_csv(
            variant_dir / "test_predictions.csv", index=False
        )
        pd.concat(
            [train_predictions, val_predictions, test_predictions],
            axis=0,
            ignore_index=True,
        ).to_csv(variant_dir / "all_predictions.csv", index=False)

        calibration_result = run_residual_calibration(
            train_predictions=train_predictions,
            val_predictions=val_predictions,
            test_predictions=test_predictions,
            output_dir=variant_dir,
            config=config.get("residual_calibration", {}),
        )
        if calibration_result is not None:
            with (
                variant_dir / "residual_calibration_summary.json"
            ).open("w", encoding="utf-8") as handle:
                json.dump(
                    calibration_result.summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

        test_tissue_mae = _compute_per_tissue_mae(
            test_predictions,
            "age_pred",
        )
        if not test_tissue_mae.empty:
            test_tissue_mae.to_csv(
                variant_dir / "test_tissue_mae.csv", index=False
            )
        if calibration_result is not None:
            calibrated_tissue_mae = _compute_per_tissue_mae(
                calibration_result.test_predictions,
                "age_pred_calibrated",
            )
            if not calibrated_tissue_mae.empty:
                calibrated_tissue_mae.to_csv(
                    variant_dir / "test_tissue_mae_calibrated.csv",
                    index=False,
                )

        run_posthoc_analyses(
            train_predictions=train_predictions,
            eval_predictions=pd.concat(
                [val_predictions, test_predictions],
                axis=0,
                ignore_index=True,
            ),
            output_dir=variant_dir,
        )
        discovery_metrics = _compute_discovery_metrics(
            train_predictions=train_predictions,
            eval_predictions=pd.concat(
                [val_predictions, test_predictions],
                axis=0,
                ignore_index=True,
            ),
        )
        combined: dict[str, float | str] = {
            "variant": spec.label,
            "variant_key": spec.key,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **discovery_metrics,
            "best_epoch": float(artifacts.best_epoch),
            "training_seconds": float(artifacts.training_seconds),
            "mean_epoch_seconds": float(
                artifacts.mean_epoch_seconds
            ),
            "train_examples_per_second": float(
                artifacts.train_examples_per_second
            ),
        }
        if calibration_result is not None:
            combined["val_calibrated_mae"] = float(
                calibration_result.calibrated_val_mae
            )
            combined["test_calibrated_mae"] = float(
                calibration_result.calibrated_test_mae
            )
        results[spec.key] = combined
        with (variant_dir / "metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(combined, handle, ensure_ascii=False, indent=2)

    summary_df = pd.DataFrame(results.values())
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)
    with (output_dir / "ablation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    _write_formal_ablation_tables(output_dir, results)
    return results
