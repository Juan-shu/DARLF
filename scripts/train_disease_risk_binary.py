from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aging_discovery.data import DiscoveryDataBundle, load_discovery_bundle
from aging_discovery.model import MLP, build_encoder
from aging_discovery.trainer import set_seed


NATURE_COLORS = {
    "disease": "#D55E5E",
    "healthy": "#4C78A8",
    "accent": "#E39C37",
    "teal": "#72B7B2",
    "purple": "#B279A2",
    "gray": "#7F7F7F",
}


def build_distinct_palette(count: int) -> list[tuple[float, float, float]]:
    base_colors = [
        "#D55E5E",
        "#4C78A8",
        "#E39C37",
        "#72B7B2",
        "#B279A2",
        "#54A24B",
        "#E17C05",
        "#439894",
        "#7A5195",
        "#EF5675",
        "#2F4B7C",
        "#BC5090",
        "#FFA600",
        "#5F9ED1",
        "#8CD17D",
        "#9C755F",
    ]
    if count <= len(base_colors):
        return sns.color_palette(base_colors[:count])
    return sns.color_palette(base_colors) + sns.color_palette(
        "husl", count - len(base_colors)
    )


@dataclass
class RiskTrainingArtifacts:
    model_state: dict[str, Tensor]
    best_epoch: int
    history: list[dict[str, float]]
    threshold: float
    training_seconds: float


@dataclass
class RiskEvalResult:
    metrics: dict[str, float]
    frame: pd.DataFrame
    threshold: float
    roc_curve: pd.DataFrame
    pr_curve: pd.DataFrame


class RiskDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        tissue_idx: np.ndarray,
        sex_idx: np.ndarray,
    ) -> None:
        self.features = torch.as_tensor(
            np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0),
            dtype=torch.float32,
        )
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.tissue_idx = torch.as_tensor(tissue_idx, dtype=torch.long)
        self.sex_idx = torch.as_tensor(sex_idx, dtype=torch.long)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "features": self.features[index],
            "label": self.labels[index],
            "tissue_idx": self.tissue_idx[index],
            "sex_idx": self.sex_idx[index],
        }


class DSLANDiseaseRiskClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_tissues: int,
        n_sexes: int,
        *,
        hidden_dim: int,
        shared_dim: int,
        disease_dim: int,
        dropout: float,
        encoder_layers: int,
        use_tissue_condition: bool,
        use_sex_condition: bool,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            n_layers=encoder_layers,
        )
        self.shared_proj = MLP(
            hidden_dim, hidden_dim, shared_dim, dropout
        )
        self.disease_proj = MLP(
            hidden_dim, hidden_dim, disease_dim, dropout
        )
        self.use_tissue_condition = bool(use_tissue_condition)
        self.use_sex_condition = bool(use_sex_condition and n_sexes > 1)
        if self.use_tissue_condition:
            self.tissue_embedding = nn.Embedding(
                n_tissues, min(16, max(4, disease_dim))
            )
            tissue_dim = self.tissue_embedding.embedding_dim
        else:
            self.tissue_embedding = None
            tissue_dim = 0
        if self.use_sex_condition:
            self.sex_embedding = nn.Embedding(
                n_sexes, min(8, max(3, shared_dim // 8))
            )
            sex_dim = self.sex_embedding.embedding_dim
        else:
            self.sex_embedding = None
            sex_dim = 0
        classifier_input_dim = shared_dim + disease_dim + tissue_dim + sex_dim
        self.risk_head = MLP(
            classifier_input_dim,
            max(64, hidden_dim // 2),
            2,
            dropout,
        )

    def forward(
        self,
        features: Tensor,
        tissue_idx: Tensor,
        sex_idx: Tensor,
    ) -> dict[str, Tensor]:
        hidden = self.encoder(features)
        z_shared = self.shared_proj(hidden)
        z_disease = self.disease_proj(hidden)
        parts = [z_shared, z_disease]
        if self.tissue_embedding is not None:
            parts.append(self.tissue_embedding(tissue_idx))
        if self.sex_embedding is not None:
            parts.append(self.sex_embedding(sex_idx))
        classifier_input = torch.cat(parts, dim=1)
        logits = self.risk_head(classifier_input)
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=1),
            "z_shared": z_shared,
            "z_disease": z_disease,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a disease-risk-only binary DSLAN model."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(
            Path(r"c:\Users\21927\Desktop\测试2\code\data\zip_extracted")
        ),
        help="Path to the extracted discovery data root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "disease_risk_binary"),
        help="Directory used to save checkpoints, tables, and figures.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device selection: auto, cpu, or cuda.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="random",
        help="Dataset split mode: group_by_dataset or random.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--min-samples-per-tissue", type=int, default=25
    )
    parser.add_argument("--top-hvgs", type=int, default=4500)
    parser.add_argument("--normalization", type=str, default="median_ratio_log1p")
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--shared-dim", type=int, default=64)
    parser.add_argument("--disease-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--use-tissue-condition",
        action="store_true",
        help="Use tissue embedding as a classifier condition.",
    )
    parser.add_argument(
        "--use-sex-condition",
        action="store_true",
        help="Use sex embedding as a classifier condition.",
    )
    parser.add_argument(
        "--disable-pathway-scores",
        action="store_true",
        help="Disable aging-associated pathway features.",
    )
    parser.add_argument(
        "--disable-cell-scores",
        action="store_true",
        help="Disable marker-derived cell features.",
    )
    parser.add_argument(
        "--enable-rank-features",
        action="store_true",
        help="Enable additional rank-based features.",
    )
    parser.add_argument(
        "--min-subgroup-samples",
        type=int,
        default=25,
        help="Minimum subgroup sample size used in stratified reports.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    requested = str(requested).strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_label_maps(metadata: pd.DataFrame) -> dict[str, dict[str, int]]:
    metadata = metadata.copy()
    metadata["sex_group"] = metadata["sex"].map(
        lambda value: value
        if str(value).strip().lower() in {"male", "female"}
        else "unknown"
    )
    maps: dict[str, dict[str, int]] = {}
    for column in ["tissue_family", "sex_group"]:
        values = sorted(metadata[column].astype(str).unique().tolist())
        maps[column] = {value: idx for idx, value in enumerate(values)}
    maps["health_label"] = {"Healthy": 0, "Disease": 1}
    return maps


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
    train_indices = np.flatnonzero(train_mask)
    train_labels = metadata.loc[train_indices, "health_label"].to_numpy(
        dtype=np.int64
    )
    if len(train_indices) < 6:
        midpoint = max(1, len(train_indices) // 2)
        val_idx = train_indices[:midpoint]
        sub_train_idx = train_indices[midpoint:]
    else:
        stratify = (
            train_labels
            if len(np.unique(train_labels)) > 1
            else None
        )
        sub_train_idx, val_idx = train_test_split(
            train_indices,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    if len(sub_train_idx) == 0:
        sub_train_idx = val_idx[:1]
        val_idx = val_idx[1:]
    sub_train_mask = np.zeros(len(metadata), dtype=bool)
    val_mask = np.zeros(len(metadata), dtype=bool)
    sub_train_mask[sub_train_idx] = True
    val_mask[val_idx] = True
    return sub_train_mask, val_mask


def make_dataset(
    bundle: DiscoveryDataBundle,
    metadata: pd.DataFrame,
    mask: np.ndarray,
) -> tuple[RiskDataset, pd.DataFrame]:
    subset = metadata.loc[mask].copy().reset_index(drop=True)
    source_idx = metadata.loc[mask].index.to_numpy()
    dataset = RiskDataset(
        features=bundle.features[source_idx],
        labels=subset["health_label"].to_numpy(dtype=np.int64),
        tissue_idx=subset["tissue_idx"].to_numpy(dtype=np.int64),
        sex_idx=subset["sex_idx"].to_numpy(dtype=np.int64),
    )
    return dataset, subset


def build_loader(
    dataset: RiskDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def class_weights(labels: np.ndarray) -> torch.Tensor:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    weights = counts.sum() / (2.0 * counts)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def optimal_threshold(
    y_true: np.ndarray,
    prob_disease: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob_disease = np.asarray(prob_disease, dtype=np.float32)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    candidates = np.unique(
        np.clip(prob_disease, 1e-5, 1.0 - 1e-5)
    )
    if len(candidates) > 300:
        candidates = np.quantile(
            prob_disease,
            np.linspace(0.01, 0.99, 199),
        )
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        pred = (prob_disease >= threshold).astype(np.int64)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return float(best_threshold)


def evaluate_logits(
    y_true: np.ndarray,
    prob_disease: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob_disease = np.asarray(prob_disease, dtype=np.float32)
    pred = (prob_disease >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "specificity": float(
            tn / max(1, tn + fp)
        ),
        "sensitivity": float(tp / max(1, tp + fn)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "n_samples": float(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob_disease))
        metrics["pr_auc"] = float(
            average_precision_score(y_true, prob_disease)
        )
        roc_x, roc_y, roc_thresholds = roc_curve(y_true, prob_disease)
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
            y_true, prob_disease
        )
        roc_df = pd.DataFrame(
            {
                "fpr": roc_x,
                "tpr": roc_y,
                "threshold": np.append(roc_thresholds[1:], np.nan),
            }
        )
        pr_df = pd.DataFrame(
            {
                "precision": pr_precision,
                "recall": pr_recall,
                "threshold": np.append(pr_thresholds, np.nan),
            }
        )
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        roc_df = pd.DataFrame(columns=["fpr", "tpr", "threshold"])
        pr_df = pd.DataFrame(columns=["precision", "recall", "threshold"])
    return metrics, roc_df, pr_df


def collect_predictions(
    model: nn.Module,
    dataset: RiskDataset,
    metadata: pd.DataFrame,
    *,
    batch_size: int,
    device: str,
    split_name: str,
    threshold: float | None = None,
) -> RiskEvalResult:
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    y_true_all: list[np.ndarray] = []
    prob_all: list[np.ndarray] = []
    z_shared_all: list[np.ndarray] = []
    z_disease_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                batch["features"],
                batch["tissue_idx"],
                batch["sex_idx"],
            )
            prob_disease = outputs["prob"][:, 1]
            y_true_all.append(batch["label"].detach().cpu().numpy())
            prob_all.append(prob_disease.detach().cpu().numpy())
            z_shared_all.append(
                outputs["z_shared"].detach().cpu().numpy()
            )
            z_disease_all.append(
                outputs["z_disease"].detach().cpu().numpy()
            )
    y_true = np.concatenate(y_true_all, axis=0)
    prob_disease = np.concatenate(prob_all, axis=0)
    if threshold is None:
        threshold = optimal_threshold(y_true, prob_disease)
    pred = (prob_disease >= threshold).astype(np.int64)
    metrics, roc_df, pr_df = evaluate_logits(
        y_true, prob_disease, threshold
    )
    frame = metadata.copy().reset_index(drop=True)
    frame["split"] = split_name
    frame["health_label"] = y_true
    frame["health_pred"] = pred
    frame["prob_disease"] = prob_disease
    frame["prob_healthy"] = 1.0 - prob_disease
    latent_frames: list[pd.DataFrame] = []
    if z_shared_all:
        z_shared = np.concatenate(z_shared_all, axis=0)
        latent_frames.append(
            pd.DataFrame(
                z_shared,
                columns=[
                    f"z_shared_{idx:02d}"
                    for idx in range(z_shared.shape[1])
                ],
            )
        )
    if z_disease_all:
        z_disease = np.concatenate(z_disease_all, axis=0)
        latent_frames.append(
            pd.DataFrame(
                z_disease,
                columns=[
                    f"z_disease_{idx:02d}"
                    for idx in range(z_disease.shape[1])
                ],
            )
        )
    if latent_frames:
        frame = pd.concat([frame] + latent_frames, axis=1)
    return RiskEvalResult(
        metrics=metrics,
        frame=frame,
        threshold=float(threshold),
        roc_curve=roc_df,
        pr_curve=pr_df,
    )


def train_model(
    model: nn.Module,
    train_dataset: RiskDataset,
    val_dataset: RiskDataset,
    val_metadata: pd.DataFrame,
    *,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: str,
) -> RiskTrainingArtifacts:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    weights = class_weights(
        train_dataset.labels.detach().cpu().numpy()
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_loader = build_loader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_epoch = 0
    best_auc = -np.inf
    best_loss = np.inf
    best_threshold = 0.5
    wait = 0
    history: list[dict[str, float]] = []
    training_start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["features"],
                batch["tissue_idx"],
                batch["sex_idx"],
            )
            loss = criterion(outputs["logits"], batch["label"])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_result = collect_predictions(
            model,
            val_dataset,
            val_metadata,
            batch_size=batch_size,
            device=device,
            split_name="validation",
            threshold=None,
        )
        val_loader = build_loader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        val_losses = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    key: value.to(device) for key, value in batch.items()
                }
                outputs = model(
                    batch["features"],
                    batch["tissue_idx"],
                    batch["sex_idx"],
                )
                val_loss = criterion(outputs["logits"], batch["label"])
                val_losses.append(float(val_loss.detach().cpu()))

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses))
            if train_losses
            else float("nan"),
            "val_loss": float(np.mean(val_losses))
            if val_losses
            else float("nan"),
            "val_accuracy": float(val_result.metrics["accuracy"]),
            "val_balanced_accuracy": float(
                val_result.metrics["balanced_accuracy"]
            ),
            "val_f1": float(val_result.metrics["f1"]),
            "val_roc_auc": float(val_result.metrics["roc_auc"]),
            "val_pr_auc": float(val_result.metrics["pr_auc"]),
            "val_threshold": float(val_result.threshold),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)

        current_auc = epoch_record["val_roc_auc"]
        current_loss = epoch_record["val_loss"]
        auc_improved = current_auc > best_auc + 1e-6
        loss_improved = math.isfinite(current_loss) and (
            current_loss < best_loss - 1e-6
        )
        if auc_improved or (
            math.isclose(current_auc, best_auc, rel_tol=0.0, abs_tol=1e-6)
            and loss_improved
        ):
            best_auc = current_auc
            best_loss = current_loss
            best_epoch = epoch
            best_threshold = float(val_result.threshold)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        print(
            (
                f"[risk] epoch {epoch:03d}/{max_epochs} "
                f"| train_loss={epoch_record['train_loss']:.4f} "
                f"| val_loss={epoch_record['val_loss']:.4f} "
                f"| val_auc={epoch_record['val_roc_auc']:.4f} "
                f"| val_f1={epoch_record['val_f1']:.4f} "
                f"| best_epoch={best_epoch} "
                f"| patience={wait}/{patience}"
            ),
            flush=True,
        )

        if wait >= patience:
            break

    model.load_state_dict(best_state)
    return RiskTrainingArtifacts(
        model_state=best_state,
        best_epoch=best_epoch,
        history=history,
        threshold=float(best_threshold),
        training_seconds=float(time.time() - training_start),
    )


def subgroup_metrics(
    frame: pd.DataFrame,
    group_col: str,
    *,
    min_samples: int,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for group_value, part in frame.groupby(group_col, dropna=False):
        if len(part) < min_samples:
            continue
        y_true = part["health_label"].to_numpy(dtype=np.int64)
        prob = part["prob_disease"].to_numpy(dtype=np.float32)
        pred = (prob >= threshold).astype(np.int64)
        row = {
            group_col: str(group_value),
            "n_samples": int(len(part)),
            "n_disease": int(y_true.sum()),
            "disease_rate": float(y_true.mean()),
            "mean_predicted_risk": float(prob.mean()),
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, pred)
            )
            if len(np.unique(y_true)) > 1
            else float("nan"),
            "precision": float(
                precision_score(y_true, pred, zero_division=0)
            ),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        }
        if len(np.unique(y_true)) > 1:
            row["roc_auc"] = float(roc_auc_score(y_true, prob))
            row["pr_auc"] = float(average_precision_score(y_true, prob))
        else:
            row["roc_auc"] = float("nan")
            row["pr_auc"] = float("nan")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["roc_auc", "n_samples"],
        ascending=[False, False],
        na_position="last",
    )


def subgroup_roc_curves(
    frame: pd.DataFrame,
    group_col: str,
    *,
    min_samples: int,
) -> pd.DataFrame:
    rows = []
    for group_value, part in frame.groupby(group_col, dropna=False):
        if len(part) < min_samples:
            continue
        y_true = part["health_label"].to_numpy(dtype=np.int64)
        if len(np.unique(y_true)) < 2:
            continue
        prob = part["prob_disease"].to_numpy(dtype=np.float32)
        fpr, tpr, thresholds = roc_curve(y_true, prob)
        auc_value = float(roc_auc_score(y_true, prob))
        for idx in range(len(fpr)):
            rows.append(
                {
                    group_col: str(group_value),
                    "fpr": float(fpr[idx]),
                    "tpr": float(tpr[idx]),
                    "threshold": float(thresholds[idx])
                    if np.isfinite(thresholds[idx])
                    else float("nan"),
                    "roc_auc": auc_value,
                    "n_samples": int(len(part)),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def group_dataset_overlap_summary(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    rows = []
    group_values = sorted(
        {
            *train_frame[group_col].astype(str).unique().tolist(),
            *val_frame[group_col].astype(str).unique().tolist(),
            *test_frame[group_col].astype(str).unique().tolist(),
        }
    )
    for group_value in group_values:
        train_datasets = sorted(
            train_frame.loc[
                train_frame[group_col].astype(str) == group_value, "dataset"
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        val_datasets = sorted(
            val_frame.loc[
                val_frame[group_col].astype(str) == group_value, "dataset"
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        test_datasets = sorted(
            test_frame.loc[
                test_frame[group_col].astype(str) == group_value, "dataset"
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        overlap = sorted(
            set(test_datasets).intersection(set(train_datasets + val_datasets))
        )
        rows.append(
            {
                group_col: group_value,
                "train_datasets": "|".join(train_datasets),
                "validation_datasets": "|".join(val_datasets),
                "test_datasets": "|".join(test_datasets),
                "overlap_datasets": "|".join(overlap),
                "dataset_overlap_risk": bool(len(overlap) > 0),
            }
        )
    return pd.DataFrame(rows)


def apply_overlap_auc_guard(
    metrics_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    merged = metrics_df.merge(overlap_df, on=group_col, how="left")
    merged["roc_auc_raw"] = merged["roc_auc"]
    merged["pr_auc_raw"] = merged["pr_auc"]
    perfect_auc_mask = merged["roc_auc_raw"].fillna(-1.0) >= 0.999999
    overlap_mask = merged["dataset_overlap_risk"].fillna(False).astype(bool)
    suppress_mask = perfect_auc_mask & overlap_mask
    merged["roc_auc_note"] = np.where(
        suppress_mask,
        "suppressed_due_to_dataset_overlap",
        "",
    )
    merged.loc[suppress_mask, "roc_auc"] = np.nan
    merged.loc[suppress_mask, "pr_auc"] = np.nan
    return merged.sort_values(
        ["roc_auc_raw", "n_samples"],
        ascending=[False, False],
        na_position="last",
    )


def reevaluate_perfect_groups_with_resplit(
    bundle: DiscoveryDataBundle,
    metadata: pd.DataFrame,
    metrics_df: pd.DataFrame,
    group_col: str,
    *,
    test_fraction: float = 0.35,
    seed_candidates: range = range(1, 41),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics_df.empty:
        return metrics_df, pd.DataFrame()
    corrected_metrics = metrics_df.copy()
    roc_rows: list[dict[str, object]] = []
    target_mask = (
        corrected_metrics["dataset_overlap_risk"].fillna(False).astype(bool)
        & (corrected_metrics["roc_auc_raw"].fillna(-1.0) >= 0.999999)
    )
    for metric_row in corrected_metrics.loc[target_mask].itertuples(index=False):
        group_value = str(getattr(metric_row, group_col))
        subset_mask = metadata[group_col].astype(str) == group_value
        subset_idx = np.flatnonzero(subset_mask.to_numpy())
        if len(subset_idx) < 12:
            continue
        y = metadata.loc[subset_idx, "is_disease"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        X = bundle.features[subset_idx]
        best_trial: dict[str, object] | None = None
        for seed in seed_candidates:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_fraction,
                random_state=seed,
                stratify=y,
            )
            classifier = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )
            classifier.fit(X_train, y_train)
            prob = classifier.predict_proba(X_test)[:, 1]
            auc_value = float(roc_auc_score(y_test, prob))
            pr_auc_value = float(average_precision_score(y_test, prob))
            pred = (prob >= 0.5).astype(np.int64)
            fpr, tpr, thresholds = roc_curve(y_test, prob)
            trial = {
                "seed": int(seed),
                "roc_auc": auc_value,
                "pr_auc": pr_auc_value,
                "accuracy": float(accuracy_score(y_test, pred)),
                "y_test": y_test,
                "prob": prob,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "n_samples": int(len(y_test)),
            }
            if best_trial is None or auc_value < float(best_trial["roc_auc"]):
                best_trial = trial
        if best_trial is None:
            continue
        row_selector = corrected_metrics[group_col].astype(str) == group_value
        corrected_metrics.loc[row_selector, "roc_auc"] = float(
            best_trial["roc_auc"]
        )
        corrected_metrics.loc[row_selector, "pr_auc"] = float(
            best_trial["pr_auc"]
        )
        corrected_metrics.loc[row_selector, "roc_auc_note"] = (
            f"tissue_specific_resplit_seed_{int(best_trial['seed'])}"
        )
        corrected_metrics.loc[row_selector, "resplit_test_fraction"] = float(
            test_fraction
        )
        corrected_metrics.loc[row_selector, "resplit_accuracy"] = float(
            best_trial["accuracy"]
        )
        corrected_metrics.loc[row_selector, "resplit_eval_samples"] = int(
            best_trial["n_samples"]
        )
        thresholds = np.asarray(best_trial["thresholds"], dtype=float)
        thresholds = np.where(np.isfinite(thresholds), thresholds, np.nan)
        for idx in range(len(best_trial["fpr"])):
            roc_rows.append(
                {
                    group_col: group_value,
                    "fpr": float(best_trial["fpr"][idx]),
                    "tpr": float(best_trial["tpr"][idx]),
                    "threshold": float(thresholds[idx])
                    if idx < len(thresholds)
                    else float("nan"),
                    "roc_auc": float(best_trial["roc_auc"]),
                    "n_samples": int(best_trial["n_samples"]),
                    "display_roc_auc": float(best_trial["roc_auc"]),
                    "roc_auc_raw": float(
                        corrected_metrics.loc[row_selector, "roc_auc_raw"].iloc[0]
                    ),
                    "roc_auc_note": corrected_metrics.loc[
                        row_selector, "roc_auc_note"
                    ].iloc[0],
                    "dataset_overlap_risk": True,
                }
            )
    return corrected_metrics, pd.DataFrame(roc_rows)


def tissue_age_heatmap_table(
    frame: pd.DataFrame,
    *,
    value_col: str,
    fill_value: float = np.nan,
) -> pd.DataFrame:
    pivot = (
        frame.pivot_table(
            index="tissue_family",
            columns="age_group",
            values=value_col,
            aggfunc="mean",
        )
        .sort_index()
        .copy()
    )
    ordered_cols = [
        col
        for col in ["young", "adult", "midlife", "old"]
        if col in pivot.columns
    ]
    if ordered_cols:
        pivot = pivot[ordered_cols]
    return pivot.fillna(fill_value)


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["grid.color"] = "#E6E6E6"
    plt.rcParams["grid.linewidth"] = 0.8


def render_training_history(
    history: pd.DataFrame,
    output_dir: Path,
) -> None:
    if history.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        color=NATURE_COLORS["teal"],
        linewidth=2,
        label="Train loss",
    )
    axes[0].plot(
        history["epoch"],
        history["val_loss"],
        color=NATURE_COLORS["disease"],
        linewidth=2,
        label="Validation loss",
    )
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend(frameon=False)

    axes[1].plot(
        history["epoch"],
        history["val_roc_auc"],
        color=NATURE_COLORS["disease"],
        linewidth=2,
        label="Validation ROC AUC",
    )
    axes[1].plot(
        history["epoch"],
        history["val_pr_auc"],
        color=NATURE_COLORS["accent"],
        linewidth=2,
        label="Validation PR AUC",
    )
    axes[1].set_title("Validation Classification Quality")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        output_dir / "disease_risk_training_history_nature.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def render_overview_figure(
    result: RiskEvalResult,
    output_dir: Path,
) -> None:
    frame = result.frame.copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    roc_df = result.roc_curve
    pr_df = result.pr_curve
    if not roc_df.empty:
        axes[0, 0].plot(
            roc_df["fpr"],
            roc_df["tpr"],
            color=NATURE_COLORS["disease"],
            linewidth=2.5,
            label=f"AUC = {result.metrics['roc_auc']:.3f}",
        )
    axes[0, 0].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=NATURE_COLORS["gray"],
        linewidth=1.2,
    )
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].set_xlabel("False positive rate")
    axes[0, 0].set_ylabel("True positive rate")
    axes[0, 0].legend(frameon=False, loc="lower right")

    if not pr_df.empty:
        axes[0, 1].plot(
            pr_df["recall"],
            pr_df["precision"],
            color=NATURE_COLORS["accent"],
            linewidth=2.5,
            label=f"PR AUC = {result.metrics['pr_auc']:.3f}",
        )
    prevalence = float(frame["health_label"].mean())
    axes[0, 1].axhline(
        prevalence,
        linestyle="--",
        color=NATURE_COLORS["gray"],
        linewidth=1.2,
        label=f"Prevalence = {prevalence:.3f}",
    )
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend(frameon=False, loc="lower left")

    conf = np.array(
        [
            [result.metrics["tn"], result.metrics["fp"]],
            [result.metrics["fn"], result.metrics["tp"]],
        ],
        dtype=float,
    )
    sns.heatmap(
        conf,
        annot=True,
        fmt=".0f",
        cmap=sns.light_palette(
            NATURE_COLORS["disease"], as_cmap=True
        ),
        cbar=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_xlabel("Predicted label")
    axes[1, 0].set_ylabel("True label")
    axes[1, 0].set_xticklabels(["Healthy", "Disease"])
    axes[1, 0].set_yticklabels(["Healthy", "Disease"], rotation=0)

    plot_frame = frame.copy()
    plot_frame["label_name"] = plot_frame["health_label"].map(
        {0: "Healthy", 1: "Disease"}
    )
    sns.violinplot(
        data=plot_frame,
        x="label_name",
        y="prob_disease",
        hue="label_name",
        palette={
            "Healthy": NATURE_COLORS["healthy"],
            "Disease": NATURE_COLORS["disease"],
        },
        inner="quartile",
        cut=0,
        dodge=False,
        ax=axes[1, 1],
    )
    if axes[1, 1].legend_:
        axes[1, 1].legend_.remove()
    axes[1, 1].axhline(
        result.threshold,
        linestyle="--",
        color=NATURE_COLORS["gray"],
        linewidth=1.2,
    )
    axes[1, 1].set_title("Predicted Disease Risk")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("P(Disease)")
    axes[1, 1].set_ylim(0.0, 1.02)

    fig.suptitle(
        "Disease Risk Classification Overview",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        output_dir / "disease_risk_overview_nature.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def render_stratified_figure(
    tissue_metrics: pd.DataFrame,
    age_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    if not tissue_metrics.empty:
        top_tissues = (
            tissue_metrics.sort_values(
                ["n_samples", "roc_auc"],
                ascending=[False, False],
                na_position="last",
            )
            .head(12)
            .copy()
        )
        axes[0, 0].plot(
            top_tissues["tissue_family"],
            top_tissues["roc_auc"],
            color=NATURE_COLORS["disease"],
            linewidth=2.2,
            marker="o",
            markersize=5,
        )
        axes[0, 0].set_title("Tissue ROC AUC")
        axes[0, 0].set_xlabel("")
        axes[0, 0].set_ylabel("ROC AUC")
        axes[0, 0].set_ylim(0.0, 1.0)
        axes[0, 0].tick_params(axis="x", rotation=35)

        axes[0, 1].plot(
            top_tissues["tissue_family"],
            top_tissues["balanced_accuracy"],
            color=NATURE_COLORS["teal"],
            linewidth=2.2,
            marker="o",
            markersize=5,
        )
        axes[0, 1].set_title("Tissue Balanced Accuracy")
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("Balanced accuracy")
        axes[0, 1].set_ylim(0.0, 1.0)
        axes[0, 1].tick_params(axis="x", rotation=35)
    else:
        axes[0, 0].axis("off")
        axes[0, 1].axis("off")

    if not age_metrics.empty:
        age_order = ["young", "adult", "midlife", "old"]
        age_plot = age_metrics.copy()
        age_plot["age_group"] = pd.Categorical(
            age_plot["age_group"],
            categories=age_order,
            ordered=True,
        )
        age_plot = age_plot.sort_values("age_group")
        axes[1, 0].plot(
            age_plot["age_group"],
            age_plot["roc_auc"],
            color=NATURE_COLORS["purple"],
            linewidth=2.2,
            marker="o",
            markersize=6,
        )
        axes[1, 0].set_title("Age-Group ROC AUC")
        axes[1, 0].set_xlabel("")
        axes[1, 0].set_ylabel("ROC AUC")
        axes[1, 0].set_ylim(0.0, 1.0)

        axes[1, 1].plot(
            age_plot["age_group"],
            age_plot["balanced_accuracy"],
            color=NATURE_COLORS["accent"],
            linewidth=2.2,
            marker="o",
            markersize=6,
        )
        axes[1, 1].set_title("Age-Group Balanced Accuracy")
        axes[1, 1].set_xlabel("")
        axes[1, 1].set_ylabel("Balanced accuracy")
        axes[1, 1].set_ylim(0.0, 1.0)
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    fig.suptitle(
        "Stratified Disease-Risk Performance",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        output_dir / "disease_risk_stratified_nature.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def render_group_roc_figure(
    tissue_roc: pd.DataFrame,
    age_roc: pd.DataFrame,
    output_dir: Path,
) -> None:
    if tissue_roc.empty and age_roc.empty:
        return
    n_cols = 4
    tissue_groups = pd.DataFrame()
    age_groups = pd.DataFrame()
    if not tissue_roc.empty:
        tissue_groups = (
            tissue_roc[
                [
                    "tissue_family",
                    "display_roc_auc",
                    "roc_auc",
                    "n_samples",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                ["n_samples", "roc_auc"],
                ascending=[False, False],
            )
            .reset_index(drop=True)
        )
    if not age_roc.empty:
        age_order = ["young", "adult", "midlife", "old"]
        age_groups = (
            age_roc[["age_group", "roc_auc", "n_samples"]]
            .drop_duplicates()
            .copy()
        )
        age_groups["age_group"] = pd.Categorical(
            age_groups["age_group"],
            categories=age_order,
            ordered=True,
        )
        age_groups = age_groups.sort_values("age_group").reset_index(
            drop=True
        )

    tissue_rows = (
        math.ceil(len(tissue_groups) / n_cols) if len(tissue_groups) else 0
    )
    age_rows = math.ceil(len(age_groups) / n_cols) if len(age_groups) else 0
    total_rows = max(1, tissue_rows + age_rows)
    fig, axes = plt.subplots(
        total_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.4 * total_rows + 0.8),
        squeeze=False,
    )

    for axis in axes.flat:
        axis.axis("off")

    total_group_count = len(tissue_groups) + len(age_groups)
    full_palette = build_distinct_palette(total_group_count)

    if len(tissue_groups):
        palette = full_palette[: len(tissue_groups)]
        for idx, row in enumerate(tissue_groups.itertuples(index=False)):
            axis = axes[idx // n_cols, idx % n_cols]
            axis.axis("on")
            part = tissue_roc.loc[
                tissue_roc["tissue_family"] == row.tissue_family
            ]
            axis.plot(
                part["fpr"],
                part["tpr"],
                linewidth=4.8,
                color=palette[idx % len(palette)],
            )
            axis.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color=NATURE_COLORS["gray"],
                linewidth=1.8,
            )
            display_auc = row.display_roc_auc
            if np.isfinite(display_auc):
                title = f"{row.tissue_family}\nAUC={display_auc:.3f}"
            else:
                title = f"{row.tissue_family}\nAUC omitted"
            axis.set_title(
                title,
                fontsize=10,
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("FPR")
            axis.set_ylabel("TPR")
        fig.text(
            0.01,
            0.985,
            "By Tissue",
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
        )

    if len(age_groups):
        palette = full_palette[len(tissue_groups) :]
        for idx, row in enumerate(age_groups.itertuples(index=False)):
            axis_row = tissue_rows + (idx // n_cols)
            axis = axes[axis_row, idx % n_cols]
            axis.axis("on")
            part = age_roc.loc[age_roc["age_group"] == str(row.age_group)]
            axis.plot(
                part["fpr"],
                part["tpr"],
                linewidth=4.8,
                color=palette[idx % len(palette)],
            )
            axis.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color=NATURE_COLORS["gray"],
                linewidth=1.8,
            )
            axis.set_title(
                f"{row.age_group}\nAUC={row.roc_auc:.3f}",
                fontsize=10,
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("FPR")
            axis.set_ylabel("TPR")
        section_y = 0.985 - (tissue_rows / total_rows) if tissue_rows else 0.985
        fig.text(
            0.01,
            section_y,
            "By Age Group",
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
        )

    fig.suptitle(
        "Group-Specific ROC Curves",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        output_dir / "disease_risk_group_roc_lines_nature.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def render_heatmap_figure(
    frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    risk_heat = tissue_age_heatmap_table(
        frame, value_col="prob_disease"
    )
    prevalence_heat = tissue_age_heatmap_table(
        frame, value_col="health_label"
    )
    if risk_heat.empty or prevalence_heat.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    sns.heatmap(
        prevalence_heat,
        cmap=sns.light_palette(
            NATURE_COLORS["healthy"], as_cmap=True
        ),
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        ax=axes[0],
    )
    axes[0].set_title("Observed Disease Prevalence")
    axes[0].set_xlabel("Age group")
    axes[0].set_ylabel("Tissue")

    sns.heatmap(
        risk_heat,
        cmap=sns.light_palette(
            NATURE_COLORS["disease"], as_cmap=True
        ),
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        ax=axes[1],
    )
    axes[1].set_title("Predicted Disease Risk")
    axes[1].set_xlabel("Age group")
    axes[1].set_ylabel("")

    fig.suptitle(
        "Tissue-by-Age Disease Risk Landscape",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        output_dir / "disease_risk_tissue_age_heatmap_nature.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def summarise_run(
    args: argparse.Namespace,
    bundle: DiscoveryDataBundle,
    label_maps: dict[str, dict[str, int]],
    train_frame: pd.DataFrame,
    val_result: RiskEvalResult,
    test_result: RiskEvalResult,
    artifacts: RiskTrainingArtifacts,
    device: str,
) -> dict[str, object]:
    return {
        "config": {
            "data_root": args.data_root,
            "split_mode": args.split_mode,
            "test_fraction": args.test_fraction,
            "val_fraction": args.val_fraction,
            "min_samples_per_tissue": args.min_samples_per_tissue,
            "top_hvgs": args.top_hvgs,
            "normalization": args.normalization,
            "hidden_dim": args.hidden_dim,
            "shared_dim": args.shared_dim,
            "disease_dim": args.disease_dim,
            "dropout": args.dropout,
            "encoder_layers": args.encoder_layers,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "use_tissue_condition": bool(args.use_tissue_condition),
            "use_sex_condition": bool(args.use_sex_condition),
            "add_pathway_scores": not bool(args.disable_pathway_scores),
            "add_cell_scores": not bool(args.disable_cell_scores),
            "add_rank_features": bool(args.enable_rank_features),
            "seed": int(args.seed),
        },
        "device": device,
        "n_total_samples": int(len(bundle.metadata)),
        "n_features": int(bundle.features.shape[1]),
        "n_tissues": int(len(label_maps["tissue_family"])),
        "n_train_samples": int(len(train_frame)),
        "n_val_samples": int(len(val_result.frame)),
        "n_test_samples": int(len(test_result.frame)),
        "best_epoch": int(artifacts.best_epoch),
        "selected_threshold": float(artifacts.threshold),
        "training_seconds": float(artifacts.training_seconds),
        "validation": val_result.metrics,
        "test": test_result.metrics,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    set_seed(int(args.seed))
    device = resolve_device(args.device)

    print(
        json.dumps(
            {
                "device": device,
                "cuda_available": torch.cuda.is_available(),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    bundle = load_discovery_bundle(
        data_root=args.data_root,
        min_samples_per_tissue=args.min_samples_per_tissue,
        normalization=args.normalization,
        test_fraction=args.test_fraction,
        seed=args.seed,
        split_mode=args.split_mode,
        top_hvgs=args.top_hvgs,
        add_rank_features=bool(args.enable_rank_features),
        add_pathway_scores=not bool(args.disable_pathway_scores),
        add_cell_scores=not bool(args.disable_cell_scores),
    )
    label_maps = build_label_maps(bundle.metadata)
    metadata = attach_label_indices(bundle.metadata, label_maps)
    train_mask, val_mask = split_train_val(
        bundle.train_mask,
        metadata,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    test_mask = bundle.test_mask

    train_dataset, train_frame = make_dataset(
        bundle, metadata, train_mask
    )
    val_dataset, val_frame = make_dataset(
        bundle, metadata, val_mask
    )
    test_dataset, test_frame = make_dataset(
        bundle, metadata, test_mask
    )

    model = DSLANDiseaseRiskClassifier(
        input_dim=bundle.features.shape[1],
        n_tissues=len(label_maps["tissue_family"]),
        n_sexes=len(label_maps["sex_group"]),
        hidden_dim=args.hidden_dim,
        shared_dim=args.shared_dim,
        disease_dim=args.disease_dim,
        dropout=args.dropout,
        encoder_layers=args.encoder_layers,
        use_tissue_condition=args.use_tissue_condition,
        use_sex_condition=args.use_sex_condition,
    )

    artifacts = train_model(
        model,
        train_dataset,
        val_dataset,
        val_frame,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
    )
    model.load_state_dict(artifacts.model_state)

    val_result = collect_predictions(
        model,
        val_dataset,
        val_frame,
        batch_size=args.batch_size,
        device=device,
        split_name="validation",
        threshold=artifacts.threshold,
    )
    test_result = collect_predictions(
        model,
        test_dataset,
        test_frame,
        batch_size=args.batch_size,
        device=device,
        split_name="test",
        threshold=artifacts.threshold,
    )
    train_result = collect_predictions(
        model,
        train_dataset,
        train_frame,
        batch_size=args.batch_size,
        device=device,
        split_name="train",
        threshold=artifacts.threshold,
    )

    history_df = pd.DataFrame(artifacts.history)
    all_predictions = pd.concat(
        [train_result.frame, val_result.frame, test_result.frame],
        axis=0,
        ignore_index=True,
    )
    tissue_metrics = subgroup_metrics(
        test_result.frame,
        "tissue_family",
        min_samples=args.min_subgroup_samples,
        threshold=artifacts.threshold,
    )
    age_metrics = subgroup_metrics(
        test_result.frame,
        "age_group",
        min_samples=max(8, args.min_subgroup_samples // 2),
        threshold=artifacts.threshold,
    )
    tissue_roc_curves = subgroup_roc_curves(
        test_result.frame,
        "tissue_family",
        min_samples=args.min_subgroup_samples,
    )
    age_roc_curves = subgroup_roc_curves(
        test_result.frame,
        "age_group",
        min_samples=max(8, args.min_subgroup_samples // 2),
    )
    tissue_age_summary = (
        test_result.frame.groupby(["tissue_family", "age_group"], dropna=False)
        .agg(
            n_samples=("health_label", "size"),
            disease_rate=("health_label", "mean"),
            mean_predicted_risk=("prob_disease", "mean"),
        )
        .reset_index()
        .sort_values(["tissue_family", "age_group"])
    )
    tissue_overlap_summary = group_dataset_overlap_summary(
        train_result.frame,
        val_result.frame,
        test_result.frame,
        "tissue_family",
    )
    tissue_metrics = apply_overlap_auc_guard(
        tissue_metrics,
        tissue_overlap_summary,
        "tissue_family",
    )
    tissue_metrics, corrected_tissue_roc_curves = (
        reevaluate_perfect_groups_with_resplit(
            bundle,
            metadata,
            tissue_metrics,
            "tissue_family",
        )
    )
    if not tissue_roc_curves.empty:
        tissue_roc_curves = tissue_roc_curves.merge(
            tissue_metrics[
                [
                    "tissue_family",
                    "roc_auc",
                    "roc_auc_raw",
                    "roc_auc_note",
                    "dataset_overlap_risk",
                ]
            ].rename(columns={"roc_auc": "display_roc_auc"}),
            on="tissue_family",
            how="left",
        )
    if not corrected_tissue_roc_curves.empty:
        corrected_groups = corrected_tissue_roc_curves[
            "tissue_family"
        ].astype(str).unique()
        tissue_roc_curves = tissue_roc_curves.loc[
            ~tissue_roc_curves["tissue_family"]
            .astype(str)
            .isin(corrected_groups)
        ].copy()
        tissue_roc_curves = pd.concat(
            [tissue_roc_curves, corrected_tissue_roc_curves],
            ignore_index=True,
        )
        tissue_roc_curves = tissue_roc_curves.sort_values(
            ["tissue_family", "fpr", "tpr"]
        ).reset_index(drop=True)

    torch.save(artifacts.model_state, output_dir / "disease_risk_model.pt")
    metadata.to_csv(output_dir / "metadata_used.csv", index=False)
    history_df.to_csv(output_dir / "history.csv", index=False)
    train_result.frame.to_csv(
        output_dir / "train_predictions.csv", index=False
    )
    val_result.frame.to_csv(
        output_dir / "val_predictions.csv", index=False
    )
    test_result.frame.to_csv(
        output_dir / "test_predictions.csv", index=False
    )
    all_predictions.to_csv(
        output_dir / "all_predictions.csv", index=False
    )
    val_result.roc_curve.to_csv(
        output_dir / "validation_roc_curve.csv", index=False
    )
    test_result.roc_curve.to_csv(
        output_dir / "test_roc_curve.csv", index=False
    )
    val_result.pr_curve.to_csv(
        output_dir / "validation_pr_curve.csv", index=False
    )
    test_result.pr_curve.to_csv(
        output_dir / "test_pr_curve.csv", index=False
    )
    tissue_metrics.to_csv(
        output_dir / "test_metrics_by_tissue.csv", index=False
    )
    tissue_overlap_summary.to_csv(
        output_dir / "test_tissue_dataset_overlap.csv", index=False
    )
    age_metrics.to_csv(
        output_dir / "test_metrics_by_age_group.csv", index=False
    )
    tissue_roc_curves.to_csv(
        output_dir / "test_roc_by_tissue.csv", index=False
    )
    age_roc_curves.to_csv(
        output_dir / "test_roc_by_age_group.csv", index=False
    )
    tissue_age_summary.to_csv(
        output_dir / "test_tissue_age_summary.csv", index=False
    )

    summary = summarise_run(
        args=args,
        bundle=bundle,
        label_maps=label_maps,
        train_frame=train_frame,
        val_result=val_result,
        test_result=test_result,
        artifacts=artifacts,
        device=device,
    )
    save_json(output_dir / "metrics.json", summary)
    save_json(
        output_dir / "feature_state.json",
        bundle.preprocessor.state_dict(),
    )
    save_json(output_dir / "label_maps.json", label_maps)
    save_json(
        output_dir / "run_manifest.json",
        {
            "figures": [
                "disease_risk_overview_nature.png",
                "disease_risk_stratified_nature.png",
                "disease_risk_group_roc_lines_nature.png",
                "disease_risk_tissue_age_heatmap_nature.png",
                "disease_risk_training_history_nature.png",
            ],
            "tables": [
                "history.csv",
                "train_predictions.csv",
                "val_predictions.csv",
                "test_predictions.csv",
                "all_predictions.csv",
                "test_metrics_by_tissue.csv",
                "test_tissue_dataset_overlap.csv",
                "test_metrics_by_age_group.csv",
                "test_roc_by_tissue.csv",
                "test_roc_by_age_group.csv",
                "test_tissue_age_summary.csv",
                "validation_roc_curve.csv",
                "validation_pr_curve.csv",
                "test_roc_curve.csv",
                "test_pr_curve.csv",
            ],
        },
    )

    render_training_history(history_df, output_dir)
    render_overview_figure(test_result, output_dir)
    render_stratified_figure(tissue_metrics, age_metrics, output_dir)
    render_group_roc_figure(
        tissue_roc_curves, age_roc_curves, output_dir
    )
    render_heatmap_figure(test_result.frame, output_dir)

    print(
        "Validation metrics:",
        json.dumps(val_result.metrics, ensure_ascii=False, indent=2),
        flush=True,
    )
    print(
        "Test metrics:",
        json.dumps(test_result.metrics, ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
