from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aging_discovery.ablation import run_ablation_study
from aging_discovery.data import load_discovery_bundle
from aging_discovery.model import ModelConfig
from aging_discovery.trainer import AgingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run systematic ablation experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "ablation.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "ablation"),
        help="Directory for ablation outputs.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=None,
        help="Specific ablation variants to run. If empty, runs all.",
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
            value: idx
            for idx, value in enumerate(unique_values)
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
    metadata["health_label"] = metadata["is_disease"].astype(
        int
    )
    return metadata


def build_focused_sample_weights(
    metadata: pd.DataFrame,
    config: dict,
) -> np.ndarray:
    focused_cfg = config.get("focused_reweight", {})
    weights = np.ones(len(metadata), dtype=np.float32)
    if not focused_cfg.get("enabled", False):
        return weights
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
        mask = tissue_families.eq(tissue_name).to_numpy()
        if np.any(mask):
            weights[mask] *= np.float32(tissue_weight)
    return weights


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    )
    label_maps = build_label_maps(bundle.metadata)
    metadata = attach_label_indices(
        bundle.metadata, label_maps
    )

    train_indices = np.flatnonzero(bundle.train_mask)
    val_size = max(
        1,
        int(
            round(
                len(train_indices)
                * config["split"]["val_fraction"]
            )
        ),
    )
    rng = np.random.default_rng(config["run"]["seed"])
    rng.shuffle(train_indices)
    val_indices = train_indices[:val_size]
    inner_train_indices = train_indices[val_size:]
    test_indices = np.flatnonzero(bundle.test_mask)

    def _make_ds(indices, sample_weight=None):
        sub = metadata.iloc[indices]
        return AgingDataset(
            features=bundle.features[indices],
            ages=sub["age"].to_numpy(dtype=np.float32),
            tissue_idx=sub["tissue_idx"].to_numpy(
                dtype=np.int64
            ),
            health_label=sub["health_label"].to_numpy(
                dtype=np.int64
            ),
            disease_idx=sub["disease_idx"].to_numpy(
                dtype=np.int64
            ),
            batch_idx=sub["batch_idx"].to_numpy(
                dtype=np.int64
            ),
            sex_idx=sub["sex_idx"].to_numpy(dtype=np.int64),
            sample_weight=sample_weight,
        )

    train_dataset = _make_ds(
        inner_train_indices,
        sample_weight=build_focused_sample_weights(
            metadata.iloc[inner_train_indices], config
        ),
    )
    val_dataset = _make_ds(
        val_indices,
        sample_weight=build_focused_sample_weights(
            metadata.iloc[val_indices], config
        ),
    )
    test_dataset = _make_ds(
        test_indices,
        sample_weight=build_focused_sample_weights(
            metadata.iloc[test_indices], config
        ),
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
        config["run"]["device"] = "cuda"
    elif requested_device == "auto":
        config["run"]["device"] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    elif requested_device:
        config["run"]["device"] = requested_device

    variants = args.variants if args.variants else None
    results = run_ablation_study(
        config=config,
        model_config=model_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_metadata=metadata.iloc[inner_train_indices].copy(),
        val_metadata=metadata.iloc[val_indices].copy(),
        test_metadata=metadata.iloc[test_indices].copy(),
        train_feature_matrix=bundle.features[inner_train_indices],
        val_feature_matrix=bundle.features[val_indices],
        test_feature_matrix=bundle.features[test_indices],
        feature_names=bundle.preprocessor.feature_names_,
        variants=variants,
        output_dir=output_dir,
    )
    print(
        "Ablation study complete. Summary:",
        json.dumps(
            {
                k: {
                    sk: sv
                    for sk, sv in v.items()
                    if "mae" in sk
                }
                for k, v in results.items()
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
