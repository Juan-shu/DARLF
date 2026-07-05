from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class AgingDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        ages: np.ndarray,
        tissue_idx: np.ndarray,
        health_label: np.ndarray,
        disease_idx: np.ndarray,
        batch_idx: np.ndarray,
        sex_idx: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        self.features = torch.as_tensor(
            np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0),
            dtype=torch.float32,
        )
        self.ages = torch.as_tensor(
            np.nan_to_num(ages, nan=0.0), dtype=torch.float32
        )
        self.tissue_idx = torch.as_tensor(
            tissue_idx, dtype=torch.long
        )
        self.health_label = torch.as_tensor(
            health_label, dtype=torch.long
        )
        self.disease_idx = torch.as_tensor(
            disease_idx, dtype=torch.long
        )
        self.batch_idx = torch.as_tensor(
            batch_idx, dtype=torch.long
        )
        if sex_idx is None:
            sex_idx = np.zeros(len(features), dtype=np.int64)
        self.sex_idx = torch.as_tensor(
            sex_idx, dtype=torch.long
        )
        if sample_weight is None:
            sample_weight = np.ones(
                len(features), dtype=np.float32
            )
        self.sample_weight = torch.as_tensor(
            np.nan_to_num(
                sample_weight, nan=1.0, posinf=1.0, neginf=0.0
            ),
            dtype=torch.float32,
        ).clamp_min_(0.0)

    def __len__(self) -> int:
        return self.features.shape[0]

    def set_sample_weights(
        self, sample_weight: np.ndarray | Tensor
    ) -> None:
        if isinstance(sample_weight, Tensor):
            tensor = sample_weight.detach().clone().to(torch.float32)
        else:
            tensor = torch.as_tensor(
                np.asarray(sample_weight, dtype=np.float32),
                dtype=torch.float32,
            )
        if tensor.numel() != len(self):
            raise ValueError(
                "sample_weight length must match dataset length"
            )
        self.sample_weight = tensor.clamp_min(0.0)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "features": self.features[index],
            "age": self.ages[index],
            "tissue_idx": self.tissue_idx[index],
            "health_label": self.health_label[index],
            "disease_idx": self.disease_idx[index],
            "batch_idx": self.batch_idx[index],
            "sex_idx": self.sex_idx[index],
            "sample_weight": self.sample_weight[index],
        }


@dataclass
class TrainingArtifacts:
    model_state: dict[str, Any]
    best_epoch: int
    history: list[dict[str, float]]
    training_seconds: float
    mean_epoch_seconds: float
    train_examples_per_second: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_loader(
    dataset: Dataset, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def _forward_model(
    model: nn.Module,
    features: Tensor,
    tissue_idx: Tensor,
    sex_idx: Tensor | None = None,
) -> dict[str, Tensor]:
    try:
        return model(features, tissue_idx, sex_idx)
    except TypeError:
        return model(features, tissue_idx)


def _monotonic_loss(shared_age_pred: Tensor, ages: Tensor) -> Tensor:
    if ages.numel() < 3:
        return shared_age_pred.new_tensor(0.0)
    sorted_idx = torch.argsort(ages)
    pred_sorted = shared_age_pred[sorted_idx]
    delta = pred_sorted[1:] - pred_sorted[:-1]
    return F.relu(0.02 - delta).mean()


def _orthogonality_loss(*latents: Tensor) -> Tensor:
    penalties = []
    centered = [
        latent - latent.mean(dim=0, keepdim=True)
        for latent in latents
    ]
    for i in range(len(centered)):
        for j in range(i + 1, len(centered)):
            lhs = F.normalize(centered[i], dim=0)
            rhs = F.normalize(centered[j], dim=0)
            penalties.append((lhs.T @ rhs).pow(2).mean())
    if not penalties:
        return centered[0].new_tensor(0.0)
    return torch.stack(penalties).mean()


def _healthy_deviation_loss(
    z_disease: Tensor, health_label: Tensor
) -> Tensor:
    healthy_mask = health_label.eq(0)
    if healthy_mask.sum() == 0:
        return z_disease.new_tensor(0.0)
    return z_disease[healthy_mask].pow(2).mean()


def _age_expert_supervision_loss(
    age_expert_logits: Tensor,
    ages: Tensor,
    low_age_threshold: float | None,
    high_age_threshold: float | None,
) -> Tensor:
    if (
        age_expert_logits.ndim != 2
        or age_expert_logits.shape[1] < 2
        or low_age_threshold is None
        or high_age_threshold is None
    ):
        return age_expert_logits.new_tensor(0.0)
    thresholds = torch.as_tensor(
        [float(low_age_threshold), float(high_age_threshold)],
        dtype=ages.dtype,
        device=ages.device,
    )
    targets = torch.bucketize(ages, thresholds).clamp(
        max=age_expert_logits.shape[1] - 1
    )
    return F.cross_entropy(age_expert_logits, targets)


def _weighted_age_loss(
    predictions: Tensor,
    targets: Tensor,
    health_label: Tensor,
    disease_age_weight: float,
    sample_weight: Tensor | None = None,
    low_age_threshold: float | None = None,
    high_age_threshold: float | None = None,
    extreme_age_weight: float = 1.0,
) -> Tensor:
    base = F.smooth_l1_loss(
        predictions, targets, reduction="none"
    )
    weights = torch.ones_like(base)
    disease_age_weight = float(disease_age_weight)
    if disease_age_weight < 0.999:
        weights = weights * torch.where(
            health_label.eq(0),
            torch.ones_like(base),
            torch.full_like(base, max(disease_age_weight, 0.0)),
        )
    extreme_age_weight = float(extreme_age_weight)
    if extreme_age_weight < 0.999:
        if low_age_threshold is not None:
            weights = weights * torch.where(
                targets <= float(low_age_threshold),
                torch.full_like(base, max(extreme_age_weight, 0.0)),
                torch.ones_like(base),
            )
        if high_age_threshold is not None:
            weights = weights * torch.where(
                targets >= float(high_age_threshold),
                torch.full_like(base, max(extreme_age_weight, 0.0)),
                torch.ones_like(base),
            )
    if sample_weight is not None:
        weights = weights * sample_weight.to(base.dtype)
    return (base * weights).sum() / weights.sum().clamp_min(1e-6)


def _weighted_residual_loss(
    residual_predictions: Tensor,
    residual_targets: Tensor,
    sample_weight: Tensor | None = None,
) -> Tensor:
    base = F.smooth_l1_loss(
        residual_predictions, residual_targets, reduction="none"
    )
    if sample_weight is not None:
        weights = sample_weight.to(base.dtype)
        return (base * weights).sum() / weights.sum().clamp_min(1e-6)
    return base.mean()


def _ramped_weight(
    weight: float, epoch: int, warmup_epochs: int
) -> float:
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 1:
        return float(weight)
    progress = min(1.0, max(0.0, epoch / float(warmup_epochs)))
    return float(weight) * progress


def compute_total_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    feature_weight: float,
    health_weight: float,
    monotonic_weight: float,
    orthogonality_weight: float,
    deviation_weight: float,
    disease_weight: float = 0.4,
    tissue_weight: float = 0.25,
    nuisance_batch_weight: float = 0.15,
    adversarial_batch_weight: float = 0.10,
    sex_weight: float = 0.0,
    age_expert_supervision_weight: float = 0.0,
    residual_weight: float = 0.0,
    disease_age_weight: float = 1.0,
    low_age_threshold: float | None = None,
    high_age_threshold: float | None = None,
    extreme_age_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    age_loss = _weighted_age_loss(
        outputs.get("age_pred", outputs.get("age")),
        batch["age"],
        batch["health_label"],
        sample_weight=batch.get("sample_weight"),
        disease_age_weight=disease_age_weight,
        low_age_threshold=low_age_threshold,
        high_age_threshold=high_age_threshold,
        extreme_age_weight=extreme_age_weight,
    )
    health_loss_val = 0.0 * age_loss
    tissue_loss_val = 0.0 * age_loss
    nuisance_batch_loss_val = 0.0 * age_loss
    adversarial_batch_loss_val = 0.0 * age_loss
    sex_loss_val = 0.0 * age_loss
    age_expert_loss_val = 0.0 * age_loss
    residual_loss_val = 0.0 * age_loss
    reconstruction_loss_val = 0.0 * age_loss
    monotonic_loss_val = 0.0 * age_loss
    orthogonality_loss_val = 0.0 * age_loss
    deviation_loss_val = 0.0 * age_loss

    if "health_logits" in outputs:
        health_loss_val = F.cross_entropy(
            outputs["health_logits"], batch["health_label"]
        )
    if "tissue_logits" in outputs:
        tissue_loss_val = F.cross_entropy(
            outputs["tissue_logits"], batch["tissue_idx"]
        )
    if "nuisance_batch_logits" in outputs:
        nuisance_batch_loss_val = F.cross_entropy(
            outputs["nuisance_batch_logits"], batch["batch_idx"]
        )
    if "adversarial_batch_logits" in outputs:
        adversarial_batch_loss_val = F.cross_entropy(
            outputs["adversarial_batch_logits"],
            batch["batch_idx"],
        )
    if "sex_logits" in outputs and "sex_idx" in batch:
        sex_loss_val = F.cross_entropy(
            outputs["sex_logits"], batch["sex_idx"]
        )
    if "age_expert_logits" in outputs:
        age_expert_loss_val = _age_expert_supervision_loss(
            outputs["age_expert_logits"],
            batch["age"],
            low_age_threshold=low_age_threshold,
            high_age_threshold=high_age_threshold,
        )
    if (
        "age_residual_total_pred" in outputs
        and "age_core_pred" in outputs
    ):
        residual_target = (
            batch["age"] - outputs["age_core_pred"].detach()
        )
        residual_loss_val = _weighted_residual_loss(
            outputs["age_residual_total_pred"],
            residual_target,
            sample_weight=batch.get("sample_weight"),
        )
    if "reconstruction" in outputs:
        reconstruction_loss_val = F.mse_loss(
            outputs["reconstruction"], batch["features"]
        )
    if "shared_age_pred" in outputs and not outputs.get(
        "_disable_monotonic_loss", False
    ):
        monotonic_loss_val = _monotonic_loss(
            outputs["shared_age_pred"], batch["age"]
        )
    latents = []
    for key in ["z_shared", "z_tissue", "z_disease", "z_nuisance"]:
        if key in outputs:
            latents.append(outputs[key])
    if len(latents) >= 2 and not outputs.get(
        "_disable_orthogonality_loss", False
    ):
        orthogonality_loss_val = _orthogonality_loss(*latents)
    if "z_disease" in outputs:
        deviation_loss_val = _healthy_deviation_loss(
            outputs["z_disease"], batch["health_label"]
        )

    total = (
        age_loss
        + health_weight * health_loss_val
        + tissue_weight * tissue_loss_val
        + nuisance_batch_weight * nuisance_batch_loss_val
        + adversarial_batch_weight * adversarial_batch_loss_val
        + sex_weight * sex_loss_val
        + age_expert_supervision_weight * age_expert_loss_val
        + residual_weight * residual_loss_val
        + feature_weight * reconstruction_loss_val
        + monotonic_weight * monotonic_loss_val
        + orthogonality_weight * orthogonality_loss_val
        + deviation_weight * deviation_loss_val
    )
    metrics = {
        "age_loss": float(age_loss.detach().cpu()),
        "health_loss": float(
            health_loss_val.detach().cpu()
            if isinstance(health_loss_val, Tensor)
            else health_loss_val
        ),
        "tissue_loss": float(
            tissue_loss_val.detach().cpu()
            if isinstance(tissue_loss_val, Tensor)
            else tissue_loss_val
        ),
        "nuisance_batch_loss": float(
            nuisance_batch_loss_val.detach().cpu()
            if isinstance(nuisance_batch_loss_val, Tensor)
            else nuisance_batch_loss_val
        ),
        "adversarial_batch_loss": float(
            adversarial_batch_loss_val.detach().cpu()
            if isinstance(adversarial_batch_loss_val, Tensor)
            else adversarial_batch_loss_val
        ),
        "sex_loss": float(
            sex_loss_val.detach().cpu()
            if isinstance(sex_loss_val, Tensor)
            else sex_loss_val
        ),
        "age_expert_loss": float(
            age_expert_loss_val.detach().cpu()
            if isinstance(age_expert_loss_val, Tensor)
            else age_expert_loss_val
        ),
        "residual_loss": float(
            residual_loss_val.detach().cpu()
            if isinstance(residual_loss_val, Tensor)
            else residual_loss_val
        ),
        "reconstruction_loss": float(
            reconstruction_loss_val.detach().cpu()
            if isinstance(reconstruction_loss_val, Tensor)
            else reconstruction_loss_val
        ),
        "monotonic_loss": float(
            monotonic_loss_val.detach().cpu()
            if isinstance(monotonic_loss_val, Tensor)
            else monotonic_loss_val
        ),
        "orthogonality_loss": float(
            orthogonality_loss_val.detach().cpu()
            if isinstance(orthogonality_loss_val, Tensor)
            else orthogonality_loss_val
        ),
        "deviation_loss": float(
            deviation_loss_val.detach().cpu()
            if isinstance(deviation_loss_val, Tensor)
            else deviation_loss_val
        ),
        "total_loss": float(total.detach().cpu()),
    }
    return total, metrics


def train_model(
    model: nn.Module,
    train_dataset: AgingDataset,
    val_dataset: AgingDataset,
    *,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    feature_weight: float,
    health_weight: float,
    monotonic_weight: float,
    orthogonality_weight: float,
    deviation_weight: float,
    seed: int,
    device: str,
    disease_weight: float = 0.4,
    tissue_weight: float = 0.25,
    nuisance_batch_weight: float = 0.15,
    adversarial_batch_weight: float = 0.10,
    sex_weight: float = 0.0,
    age_expert_supervision_weight: float = 0.0,
    residual_weight: float = 0.0,
    age_only_finetune_epochs: int = 0,
    age_only_lr_scale: float = 0.25,
    disease_age_weight: float = 1.0,
    low_age_threshold: float | None = None,
    high_age_threshold: float | None = None,
    extreme_age_weight: float = 1.0,
    use_amp: bool = False,
    selection_metric: str = "val_loss",
    lr_scheduler_patience: int = 0,
    lr_scheduler_factor: float = 0.5,
    min_learning_rate: float = 1e-5,
    adversarial_warmup_epochs: int = 0,
    orthogonality_warmup_epochs: int = 0,
    tissue_warmup_epochs: int = 0,
    verbose: bool = False,
    log_prefix: str = "train",
) -> TrainingArtifacts:
    set_seed(seed)
    model = model.to(device)
    amp_enabled = bool(use_amp) and str(device).startswith("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = None
    if int(lr_scheduler_patience) > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(lr_scheduler_factor),
            patience=int(lr_scheduler_patience),
            min_lr=float(min_learning_rate),
        )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled
    )
    train_loader = build_loader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = build_loader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    selection_metric = str(selection_metric).strip().lower()
    if selection_metric not in {"val_loss", "val_mae"}:
        raise ValueError(
            f"Unsupported selection_metric: {selection_metric}"
        )
    best_metric = float("inf")
    wait = 0
    history: list[dict[str, float]] = []
    training_start = time.time()
    completed_joint_epochs = 0
    completed_finetune_epochs = 0

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []
        effective_tissue_weight = _ramped_weight(
            tissue_weight, epoch, tissue_warmup_epochs
        )
        effective_adversarial_batch_weight = _ramped_weight(
            adversarial_batch_weight,
            epoch,
            adversarial_warmup_epochs,
        )
        effective_orthogonality_weight = _ramped_weight(
            orthogonality_weight,
            epoch,
            orthogonality_warmup_epochs,
        )
        for batch in train_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                enabled=amp_enabled,
            ):
                outputs = _forward_model(
                    model,
                    batch["features"],
                    batch["tissue_idx"],
                    batch.get("sex_idx"),
                )
                loss, _ = compute_total_loss(
                    outputs,
                    batch,
                    feature_weight=feature_weight,
                    health_weight=health_weight,
                    monotonic_weight=monotonic_weight,
                    orthogonality_weight=effective_orthogonality_weight,
                    deviation_weight=deviation_weight,
                    disease_weight=disease_weight,
                    tissue_weight=effective_tissue_weight,
                    nuisance_batch_weight=nuisance_batch_weight,
                    adversarial_batch_weight=effective_adversarial_batch_weight,
                    sex_weight=sex_weight,
                    age_expert_supervision_weight=age_expert_supervision_weight,
                    residual_weight=residual_weight,
                    disease_age_weight=disease_age_weight,
                    low_age_threshold=low_age_threshold,
                    high_age_threshold=high_age_threshold,
                    extreme_age_weight=extreme_age_weight,
                )
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        val_age_preds = []
        val_age_true = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }
                with torch.autocast(
                    device_type="cuda",
                    enabled=amp_enabled,
                ):
                    outputs = _forward_model(
                        model,
                        batch["features"],
                        batch["tissue_idx"],
                        batch.get("sex_idx"),
                    )
                    loss, _ = compute_total_loss(
                        outputs,
                        batch,
                        feature_weight=feature_weight,
                        health_weight=health_weight,
                        monotonic_weight=monotonic_weight,
                        orthogonality_weight=effective_orthogonality_weight,
                        deviation_weight=deviation_weight,
                        disease_weight=disease_weight,
                        tissue_weight=effective_tissue_weight,
                        nuisance_batch_weight=nuisance_batch_weight,
                        adversarial_batch_weight=effective_adversarial_batch_weight,
                        sex_weight=sex_weight,
                        age_expert_supervision_weight=age_expert_supervision_weight,
                        residual_weight=residual_weight,
                        disease_age_weight=disease_age_weight,
                        low_age_threshold=low_age_threshold,
                        high_age_threshold=high_age_threshold,
                        extreme_age_weight=extreme_age_weight,
                    )
                val_losses.append(float(loss.detach().cpu()))
                val_age_preds.append(
                    outputs["age_pred"].detach().cpu().numpy()
                )
                val_age_true.append(
                    batch["age"].detach().cpu().numpy()
                )

        val_mae = float("nan")
        if val_age_preds and val_age_true:
            val_mae = float(
                mean_absolute_error(
                    np.concatenate(val_age_true, axis=0),
                    np.concatenate(val_age_preds, axis=0),
                )
            )

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses))
            if train_losses
            else float("nan"),
            "val_loss": float(np.mean(val_losses))
            if val_losses
            else float("nan"),
            "val_mae": val_mae,
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "effective_tissue_weight": effective_tissue_weight,
            "effective_adversarial_batch_weight": (
                effective_adversarial_batch_weight
            ),
            "effective_orthogonality_weight": (
                effective_orthogonality_weight
            ),
        }
        history.append(epoch_record)
        completed_joint_epochs = epoch
        monitored_value = epoch_record[selection_metric]
        if scheduler is not None and np.isfinite(monitored_value):
            scheduler.step(monitored_value)
        if monitored_value < best_metric:
            best_metric = monitored_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    elapsed = time.time() - training_start
                    print(
                        f"[{log_prefix}] early stopping at epoch "
                        f"{epoch}/{max_epochs} | best_epoch={best_epoch} "
                        f"| elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                break
        if verbose:
            elapsed = time.time() - training_start
            epoch_duration = time.time() - epoch_start
            avg_epoch = elapsed / epoch
            eta_seconds = max(0.0, avg_epoch * (max_epochs - epoch))
            best_flag = "*" if best_epoch == epoch else ""
            print(
                f"[{log_prefix}] epoch {epoch:03d}/{max_epochs} "
                f"| train_loss={epoch_record['train_loss']:.4f} "
                f"| val_loss={epoch_record['val_loss']:.4f} "
                f"| val_mae={epoch_record['val_mae']:.4f} "
                f"| lr={optimizer.param_groups[0]['lr']:.6f} "
                f"| best_epoch={best_epoch}{best_flag} "
                f"| patience={wait}/{patience} "
                f"| epoch_time={epoch_duration:.1f}s "
                f"| elapsed={elapsed:.1f}s "
                f"| eta={eta_seconds:.1f}s",
                flush=True,
            )

    model.load_state_dict(best_state)
    if age_only_finetune_epochs > 0:
        finetune_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate * max(age_only_lr_scale, 1e-3),
            weight_decay=weight_decay,
        )
        finetune_scheduler = None
        if int(lr_scheduler_patience) > 0:
            finetune_scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    finetune_optimizer,
                    mode="min",
                    factor=float(lr_scheduler_factor),
                    patience=max(1, int(lr_scheduler_patience) // 2),
                    min_lr=float(min_learning_rate),
                )
            )
        best_finetune_state = copy.deepcopy(model.state_dict())
        best_finetune_metric = best_metric
        best_finetune_epoch = best_epoch
        wait = 0
        for finetune_epoch in range(1, age_only_finetune_epochs + 1):
            epoch_start = time.time()
            model.train()
            train_age_losses = []
            for batch in train_loader:
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }
                finetune_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    enabled=amp_enabled,
                ):
                    outputs = _forward_model(
                        model,
                        batch["features"],
                        batch["tissue_idx"],
                        batch.get("sex_idx"),
                    )
                    age_loss = _weighted_age_loss(
                        outputs["age_pred"],
                        batch["age"],
                        batch["health_label"],
                        sample_weight=batch.get("sample_weight"),
                        disease_age_weight=disease_age_weight,
                        low_age_threshold=low_age_threshold,
                        high_age_threshold=high_age_threshold,
                        extreme_age_weight=extreme_age_weight,
                    )
                    if "age_expert_logits" in outputs:
                        age_loss = age_loss + (
                            float(age_expert_supervision_weight)
                            * _age_expert_supervision_loss(
                                outputs["age_expert_logits"],
                                batch["age"],
                                low_age_threshold=low_age_threshold,
                                high_age_threshold=high_age_threshold,
                            )
                        )
                if amp_enabled:
                    scaler.scale(age_loss).backward()
                    scaler.step(finetune_optimizer)
                    scaler.update()
                else:
                    age_loss.backward()
                    finetune_optimizer.step()
                train_age_losses.append(float(age_loss.detach().cpu()))

            model.eval()
            val_age_losses = []
            val_age_preds = []
            val_age_true = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {
                        key: value.to(device)
                        for key, value in batch.items()
                    }
                    with torch.autocast(
                        device_type="cuda",
                        enabled=amp_enabled,
                    ):
                        outputs = _forward_model(
                            model,
                            batch["features"],
                            batch["tissue_idx"],
                            batch.get("sex_idx"),
                        )
                        age_loss = _weighted_age_loss(
                            outputs["age_pred"],
                            batch["age"],
                            batch["health_label"],
                            sample_weight=batch.get(
                                "sample_weight"
                            ),
                            disease_age_weight=disease_age_weight,
                            low_age_threshold=low_age_threshold,
                            high_age_threshold=high_age_threshold,
                            extreme_age_weight=extreme_age_weight,
                        )
                        if "age_expert_logits" in outputs:
                            age_loss = age_loss + (
                                float(age_expert_supervision_weight)
                                * _age_expert_supervision_loss(
                                    outputs["age_expert_logits"],
                                    batch["age"],
                                    low_age_threshold=low_age_threshold,
                                    high_age_threshold=high_age_threshold,
                                )
                            )
                    val_age_losses.append(
                        float(age_loss.detach().cpu())
                    )
                    val_age_preds.append(
                        outputs["age_pred"].detach().cpu().numpy()
                    )
                    val_age_true.append(
                        batch["age"].detach().cpu().numpy()
                    )
            completed_finetune_epochs = finetune_epoch
            mean_train_age = float(np.mean(train_age_losses))
            mean_val_age = float(np.mean(val_age_losses))
            val_mae = float(
                mean_absolute_error(
                    np.concatenate(val_age_true, axis=0),
                    np.concatenate(val_age_preds, axis=0),
                )
            )
            monitored_value = (
                val_mae
                if selection_metric == "val_mae"
                else mean_val_age
            )
            if (
                finetune_scheduler is not None
                and np.isfinite(monitored_value)
            ):
                finetune_scheduler.step(monitored_value)
            if monitored_value < best_finetune_metric:
                best_finetune_metric = monitored_value
                best_finetune_epoch = max_epochs + finetune_epoch
                best_finetune_state = copy.deepcopy(
                    model.state_dict()
                )
                wait = 0
            else:
                wait += 1
            if verbose:
                elapsed = time.time() - training_start
                epoch_duration = time.time() - epoch_start
                eta_seconds = max(
                    0.0,
                    (time.time() - epoch_start)
                    * (age_only_finetune_epochs - finetune_epoch),
                )
                best_flag = (
                    "*"
                    if best_finetune_epoch == max_epochs + finetune_epoch
                    else ""
                )
                print(
                    f"[{log_prefix}-ageft] epoch "
                    f"{finetune_epoch:03d}/{age_only_finetune_epochs} "
                    f"| train_age_loss={mean_train_age:.4f} "
                    f"| val_age_loss={mean_val_age:.4f} "
                    f"| val_mae={val_mae:.4f} "
                    f"| lr={finetune_optimizer.param_groups[0]['lr']:.6f} "
                    f"| best_epoch={best_finetune_epoch}{best_flag} "
                    f"| patience={wait}/{patience} "
                    f"| epoch_time={epoch_duration:.1f}s "
                    f"| elapsed={elapsed:.1f}s "
                    f"| eta={eta_seconds:.1f}s",
                    flush=True,
                )
            if wait >= patience:
                break
        model.load_state_dict(best_finetune_state)
        best_state = copy.deepcopy(best_finetune_state)
        best_epoch = best_finetune_epoch
    total_training_seconds = time.time() - training_start
    total_epochs_run = completed_joint_epochs + completed_finetune_epochs
    mean_epoch_seconds = (
        total_training_seconds / total_epochs_run
        if total_epochs_run > 0
        else 0.0
    )
    total_examples_seen = len(train_dataset) * total_epochs_run
    train_examples_per_second = (
        total_examples_seen / total_training_seconds
        if total_training_seconds > 0
        else 0.0
    )
    return TrainingArtifacts(
        model_state=best_state,
        best_epoch=best_epoch,
        history=history,
        training_seconds=float(total_training_seconds),
        mean_epoch_seconds=float(mean_epoch_seconds),
        train_examples_per_second=float(train_examples_per_second),
    )


LATENT_KEY_MAP = {
    "z_shared": "z_shared",
    "z_tissue": "z_tissue",
    "z_disease": "z_disease",
    "z_nuisance": "z_nuisance",
}


def _safe_binary_auc(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float | None, float | None]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return None, None
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = None
    try:
        auprc = float(average_precision_score(y_true, y_score))
    except ValueError:
        auprc = None
    return auroc, auprc


def _collect_predictions(
    model: nn.Module,
    dataset: AgingDataset,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    loader = build_loader(
        dataset, batch_size=batch_size, shuffle=False
    )
    model = model.to(device)
    model.eval()
    outputs_all: dict[str, list[np.ndarray]] = {
        "age_pred": [],
        "age_base_pred": [],
        "age_core_pred": [],
        "age_residual_total_pred": [],
        "age_true": [],
        "sex_true": [],
        "sex_pred": [],
        "sex_pred_available": [],
        "health_true": [],
        "health_pred": [],
        "health_pred_available": [],
        "health_prob": [],
        "shared_age_pred": [],
        "z_shared": [],
        "z_tissue": [],
        "z_disease": [],
        "z_nuisance": [],
        "z_raw_context": [],
    }
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            outputs = _forward_model(
                model,
                batch["features"],
                batch["tissue_idx"],
                batch.get("sex_idx"),
            )
            outputs_all["age_pred"].append(
                outputs["age_pred"].detach().cpu().numpy()
            )
            if "age_base_pred" in outputs:
                outputs_all["age_base_pred"].append(
                    outputs["age_base_pred"]
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                outputs_all["age_base_pred"].append(
                    outputs["age_pred"].detach().cpu().numpy()
                )
            if "age_core_pred" in outputs:
                outputs_all["age_core_pred"].append(
                    outputs["age_core_pred"]
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                outputs_all["age_core_pred"].append(
                    outputs["age_pred"].detach().cpu().numpy()
                )
            if "age_residual_total_pred" in outputs:
                outputs_all["age_residual_total_pred"].append(
                    outputs["age_residual_total_pred"]
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                outputs_all["age_residual_total_pred"].append(
                    np.zeros(
                        batch["age"].shape[0], dtype=np.float32
                    )
                )
            outputs_all["age_true"].append(
                batch["age"].detach().cpu().numpy()
            )
            outputs_all["sex_true"].append(
                batch["sex_idx"].detach().cpu().numpy()
            )
            if "sex_logits" in outputs:
                outputs_all["sex_pred"].append(
                    outputs["sex_logits"]
                    .argmax(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                outputs_all["sex_pred_available"].append(
                    np.ones(
                        batch["sex_idx"].shape[0],
                        dtype=np.float32,
                    )
                )
            else:
                outputs_all["sex_pred"].append(
                    np.full(
                        batch["sex_idx"].shape[0],
                        -1,
                        dtype=np.int64,
                    )
                )
                outputs_all["sex_pred_available"].append(
                    np.zeros(
                        batch["sex_idx"].shape[0],
                        dtype=np.float32,
                    )
                )
            outputs_all["health_true"].append(
                batch["health_label"].detach().cpu().numpy()
            )
            if "health_logits" in outputs:
                health_prob = (
                    torch.softmax(outputs["health_logits"], dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                outputs_all["health_pred"].append(
                    outputs["health_logits"]
                    .argmax(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                outputs_all["health_pred_available"].append(
                    np.ones(
                        batch["health_label"].shape[0],
                        dtype=np.float32,
                    )
                )
                outputs_all["health_prob"].append(health_prob)
            else:
                outputs_all["health_pred"].append(
                    np.full(
                        batch["health_label"].shape[0],
                        -1,
                        dtype=np.int64,
                    )
                )
                outputs_all["health_pred_available"].append(
                    np.zeros(
                        batch["health_label"].shape[0],
                        dtype=np.float32,
                    )
                )
                outputs_all["health_prob"].append(
                    np.zeros(
                        (batch["health_label"].shape[0], 0),
                        dtype=np.float32,
                    )
                )
            if "shared_age_pred" in outputs:
                outputs_all["shared_age_pred"].append(
                    outputs["shared_age_pred"]
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                outputs_all["shared_age_pred"].append(
                    outputs["age_pred"].detach().cpu().numpy()
                )
            for key in [
                "z_shared",
                "z_tissue",
                "z_disease",
                "z_nuisance",
                "z_raw_context",
            ]:
                if key in outputs:
                    outputs_all[key].append(
                        outputs[key].detach().cpu().numpy()
                    )
                else:
                    outputs_all[key].append(
                        np.zeros(
                            (batch["features"].shape[0], 1),
                            dtype=np.float32,
                        )
                    )
    return {
        key: (
            np.concatenate(value, axis=0)
            if value
            else np.array([], dtype=np.float32)
        )
        for key, value in outputs_all.items()
    }


def evaluate_model(
    model: nn.Module,
    dataset: AgingDataset,
    *,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    preds = _collect_predictions(
        model, dataset, batch_size=batch_size, device=device
    )
    age_true = np.nan_to_num(preds["age_true"], nan=0.0)
    age_pred = np.nan_to_num(preds["age_pred"], nan=0.0)
    health_true = preds["health_true"]
    health_pred = preds["health_pred"]
    health_pred_available = preds.get(
        "health_pred_available", np.zeros_like(health_true)
    )
    health_prob = preds.get("health_prob", np.zeros((len(age_true), 0)))
    sex_true = preds.get("sex_true", np.array([], dtype=np.int64))
    sex_pred = preds.get("sex_pred", np.array([], dtype=np.int64))
    sex_pred_available = preds.get(
        "sex_pred_available", np.zeros_like(sex_true)
    )
    metrics: dict[str, float] = {
        "mae": float(mean_absolute_error(age_true, age_pred)),
        "r2": float(r2_score(age_true, age_pred)),
    }
    if (
        len(np.unique(health_true)) > 1
        and np.any(health_pred_available > 0.5)
    ):
        valid_mask = health_pred_available > 0.5
        metrics["health_accuracy"] = float(
            accuracy_score(
                health_true[valid_mask],
                health_pred[valid_mask],
            )
        )
        if health_prob.ndim == 2 and health_prob.shape[1] >= 2:
            auroc, auprc = _safe_binary_auc(
                health_true[valid_mask],
                health_prob[valid_mask, 1],
            )
            if auroc is not None:
                metrics["health_auroc"] = auroc
            if auprc is not None:
                metrics["health_auprc"] = auprc
    if len(sex_true) > 0 and np.any(sex_pred_available > 0.5):
        valid_mask = (
            (sex_pred_available > 0.5)
            & (sex_true >= 0)
        )
        if np.any(valid_mask):
            metrics["sex_accuracy"] = float(
                accuracy_score(
                    sex_true[valid_mask],
                    sex_pred[valid_mask],
                )
            )
    if "shared_age_pred" in preds and preds["shared_age_pred"].size > 0:
        metrics["shared_age_std"] = float(
            np.std(preds["shared_age_pred"])
        )
    return metrics


def predict_age_only(
    model: nn.Module,
    dataset: AgingDataset,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    preds = _collect_predictions(
        model, dataset, batch_size=batch_size, device=device
    )
    return preds["age_pred"].astype(np.float32, copy=False)


def predict_dataset(
    model: nn.Module,
    dataset: AgingDataset,
    metadata: pd.DataFrame,
    *,
    batch_size: int,
    device: str,
    feature_matrix: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    preds = _collect_predictions(
        model, dataset, batch_size=batch_size, device=device
    )
    frame = metadata.copy().reset_index(drop=True)
    frame["age_pred"] = np.nan_to_num(preds["age_pred"], nan=0.0)
    if "age_base_pred" in preds:
        frame["age_pred_base"] = np.nan_to_num(
            preds["age_base_pred"], nan=0.0
        )
    if "age_core_pred" in preds:
        frame["age_pred_core"] = np.nan_to_num(
            preds["age_core_pred"], nan=0.0
        )
    if "age_residual_total_pred" in preds:
        frame["age_residual_pred"] = np.nan_to_num(
            preds["age_residual_total_pred"], nan=0.0
        )
    frame["shared_age_pred"] = np.nan_to_num(
        preds["shared_age_pred"], nan=0.0
    )
    frame["health_pred"] = preds.get(
        "health_pred", np.zeros(len(frame))
    )
    frame["sex_pred"] = preds.get(
        "sex_pred", np.full(len(frame), -1)
    )
    health_prob = preds.get("health_prob")
    if health_prob is not None and health_prob.ndim == 2:
        for idx in range(health_prob.shape[1]):
            frame[f"health_prob_{idx}"] = np.nan_to_num(
                health_prob[:, idx], nan=0.0
            )
    latent_frames: list[pd.DataFrame] = []
    for prefix in [
        "z_shared",
        "z_tissue",
        "z_disease",
        "z_nuisance",
        "z_raw_context",
    ]:
        block = preds.get(prefix)
        if block is None or block.size == 0:
            continue
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        columns = [
            f"{prefix}_{idx:02d}" for idx in range(block.shape[1])
        ]
        latent_frames.append(pd.DataFrame(block, columns=columns))
    extra_frames: list[pd.DataFrame] = []
    if feature_matrix is not None and feature_names:
        keep_indices = [
            idx
            for idx, name in enumerate(feature_names)
            if name.startswith("pathway::") or name.startswith("cell::")
        ]
        if keep_indices:
            renamed = [
                feature_names[idx]
                .replace("pathway::", "pathway_")
                .replace("cell::", "cell_")
                for idx in keep_indices
            ]
            extra_frames.append(
                pd.DataFrame(
                    feature_matrix[:, keep_indices],
                    columns=renamed,
                )
            )
    return pd.concat([frame] + latent_frames + extra_frames, axis=1)
