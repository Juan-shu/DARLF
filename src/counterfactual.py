from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import DisentangledAgingModel, ModelConfig
from .trainer import AgingDataset, predict_dataset


def _encode_to_latents(
    model: DisentangledAgingModel,
    features: Tensor,
    tissue_idx: Tensor,
) -> dict[str, Tensor]:
    hidden = model.encoder(features)
    return {
        "z_shared": model.shared_proj(hidden),
        "z_tissue": model.tissue_proj(hidden),
        "z_disease": model.disease_proj(hidden),
        "z_nuisance": model.nuisance_proj(hidden),
    }


def _predict_age_from_latents(
    model: DisentangledAgingModel,
    z_shared: Tensor,
    z_tissue: Tensor,
    tissue_idx: Tensor,
) -> Tensor:
    tissue_embed = model.tissue_embedding(tissue_idx)
    age_input = torch.cat([z_shared, z_tissue, tissue_embed], dim=1)
    base_age_pred = model.age_head(age_input).squeeze(1)
    if getattr(model, "n_age_experts", 1) > 1:
        gate_logits = model.age_expert_gate(age_input)
        gate_logits = gate_logits + model.age_expert_bias(tissue_idx)
        expert_weights = torch.softmax(gate_logits, dim=1)
        expert_preds = torch.cat(
            [expert(age_input) for expert in model.age_experts], dim=1
        )
        return base_age_pred + (
            expert_weights * expert_preds
        ).sum(dim=1)
    return base_age_pred


def run_shared_latent_perturbation(
    model: DisentangledAgingModel,
    features: np.ndarray,
    tissue_idx: np.ndarray,
    ages: np.ndarray,
    perturbation_scales: list[float] | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    if perturbation_scales is None:
        perturbation_scales = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    model = model.to(device)
    model.eval()
    feat_tensor = torch.as_tensor(features, dtype=torch.float32).to(device)
    tissue_tensor = torch.as_tensor(tissue_idx, dtype=torch.long).to(device)
    records: list[dict[str, float]] = []
    latents = _encode_to_latents(model, feat_tensor, tissue_tensor)
    z_shared = latents["z_shared"].detach().requires_grad_(True)
    z_tissue = latents["z_tissue"].detach()
    pred_age_ref = _predict_age_from_latents(
        model, z_shared, z_tissue, tissue_tensor
    )
    shared_direction = torch.autograd.grad(
        pred_age_ref.sum(), z_shared
    )[0]
    shared_direction = shared_direction / shared_direction.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-6)
    step_size = latents["z_shared"].std(dim=0).mean().clamp_min(1e-3)
    with torch.no_grad():
        base_shared = latents["z_shared"].detach()
        for scale in perturbation_scales:
            perturbed = base_shared + scale * step_size * shared_direction
            pred_age = _predict_age_from_latents(
                model, perturbed, z_tissue, tissue_tensor
            )
            for i in range(perturbed.shape[0]):
                records.append(
                    {
                        "sample_idx": float(i),
                        "true_age": float(ages[i]),
                        "perturbation_scale": scale,
                        "predicted_age": float(pred_age[i].cpu()),
                    }
                )
    return pd.DataFrame(records)


def run_tissue_latent_perturbation(
    model: DisentangledAgingModel,
    features: np.ndarray,
    tissue_idx: np.ndarray,
    ages: np.ndarray,
    target_tissue_idx: int,
    perturbation_strengths: list[float] | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    if perturbation_strengths is None:
        perturbation_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    model = model.to(device)
    model.eval()
    feat_tensor = torch.as_tensor(features, dtype=torch.float32).to(device)
    tissue_tensor = torch.as_tensor(tissue_idx, dtype=torch.long).to(device)
    with torch.no_grad():
        latents = _encode_to_latents(model, feat_tensor, tissue_tensor)
        target_mask = tissue_tensor.eq(int(target_tissue_idx))
        if target_mask.any():
            target_center = latents["z_tissue"][target_mask].mean(
                dim=0, keepdim=True
            )
        else:
            target_center = latents["z_tissue"].mean(dim=0, keepdim=True)
        target_embed = model.tissue_embedding(
            torch.full_like(tissue_tensor, target_tissue_idx)
        )
        results = []
        for strength in perturbation_strengths:
            perturbed_tissue = latents["z_tissue"] + strength * (
                target_center - latents["z_tissue"]
            )
            pred_age = _predict_age_from_latents(
                model,
                latents["z_shared"],
                perturbed_tissue,
                torch.full_like(tissue_tensor, target_tissue_idx),
            )
            for i in range(perturbed_tissue.shape[0]):
                results.append(
                    {
                        "sample_idx": float(i),
                        "true_age": float(ages[i]),
                        "target_tissue_idx": float(target_tissue_idx),
                        "perturbation_strength": float(strength),
                        "perturbed_predicted_age": float(pred_age[i].cpu()),
                    }
                )
    return pd.DataFrame(results)


def run_disease_latent_perturbation(
    model: DisentangledAgingModel,
    features: np.ndarray,
    tissue_idx: np.ndarray,
    health_label: np.ndarray,
    perturbation_strengths: list[float] | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    if perturbation_strengths is None:
        perturbation_strengths = [0.0, 1.0, 2.0, 3.0, 5.0]

    model = model.to(device)
    model.eval()
    feat_tensor = torch.as_tensor(features, dtype=torch.float32).to(device)
    tissue_tensor = torch.as_tensor(tissue_idx, dtype=torch.long).to(device)
    records: list[dict[str, float]] = []
    with torch.no_grad():
        latents = _encode_to_latents(model, feat_tensor, tissue_tensor)
        health_tensor = torch.as_tensor(health_label, dtype=torch.long).to(device)
        healthy_mask = health_tensor.eq(0)
        diseased_mask = health_tensor.eq(1)
        if not healthy_mask.any() or not diseased_mask.any():
            return pd.DataFrame()
        healthy_latent = latents["z_disease"][healthy_mask]
        disease_center = latents["z_disease"][diseased_mask].mean(
            dim=0, keepdim=True
        )
        healthy_indices = torch.nonzero(healthy_mask, as_tuple=False).squeeze(1)
        for strength in perturbation_strengths:
            perturbed = healthy_latent + strength * (
                disease_center - healthy_latent
            )
            health_logits = model.health_classifier(perturbed)
            health_probs = F.softmax(health_logits, dim=1)
            pred_labels = health_logits.argmax(dim=1)
            for i in range(perturbed.shape[0]):
                records.append(
                    {
                        "sample_idx": float(healthy_indices[i].item()),
                        "true_health_label": 0.0,
                        "perturbation_strength": float(strength),
                        "predicted_health_label": float(pred_labels[i].cpu()),
                        "disease_prob": float(health_probs[i, 1].cpu() if health_probs.shape[1] > 1 else 0.0),
                    }
                )
    return pd.DataFrame(records)


def run_batch_latent_swap(
    model: DisentangledAgingModel,
    features: np.ndarray,
    tissue_idx: np.ndarray,
    batch_idx: np.ndarray,
    ages: np.ndarray,
    target_batch_idx: int,
    device: str = "cpu",
) -> pd.DataFrame:
    model = model.to(device)
    model.eval()
    feat_tensor = torch.as_tensor(features, dtype=torch.float32).to(device)
    tissue_tensor = torch.as_tensor(tissue_idx, dtype=torch.long).to(device)
    records: list[dict[str, float]] = []
    with torch.no_grad():
        latents = _encode_to_latents(model, feat_tensor, tissue_tensor)
        reference = model.nuisance_batch_classifier(latents["z_nuisance"]).argmax(dim=1)
        nuisance_std = latents["z_nuisance"].std(dim=0, keepdim=True)
        perturbed_nuisance = latents["z_nuisance"] + 2.0 * nuisance_std
        shifted_pred = model.nuisance_batch_classifier(perturbed_nuisance).argmax(dim=1)
        num_switched = (reference != shifted_pred).sum().item()
        records.append(
            {
                "n_samples": float(features.shape[0]),
                "n_batch_switched": float(num_switched),
                "switch_fraction": (
                    float(num_switched / features.shape[0])
                    if features.shape[0] > 0
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(records)


def run_full_counterfactual_report(
    model: DisentangledAgingModel,
    dataset: AgingDataset,
    metadata: pd.DataFrame,
    output_dir: str | Path,
    target_tissue_idx: int | None = None,
    blood_target_tissue_idx: int | None = None,
    device: str = "cpu",
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, str] = {}

    features = dataset.features.numpy()
    ages = dataset.ages.numpy()
    tissue_idx = dataset.tissue_idx.numpy()
    health_label = dataset.health_label.numpy()
    batch_idx = dataset.batch_idx.numpy()

    shared_df = run_shared_latent_perturbation(
        model, features, tissue_idx, ages, device=device
    )
    if not shared_df.empty:
        path = output_dir / "counterfactual_shared_latent.csv"
        shared_df.to_csv(path, index=False)
        reports["shared_latent"] = str(path)

    unique_tissues = np.unique(tissue_idx)
    chosen_target_tissue_idx = target_tissue_idx
    if (
        chosen_target_tissue_idx is None
        and len(unique_tissues) >= 2
    ):
        chosen_target_tissue_idx = int(unique_tissues[1])
    if chosen_target_tissue_idx is not None and len(unique_tissues) >= 2:
        tissue_df = run_tissue_latent_perturbation(
            model,
            features,
            tissue_idx,
            ages,
            target_tissue_idx=int(chosen_target_tissue_idx),
            device=device,
        )
        if not tissue_df.empty:
            path = output_dir / "counterfactual_tissue_latent.csv"
            tissue_df.to_csv(path, index=False)
            reports["tissue_latent"] = str(path)

    disease_df = run_disease_latent_perturbation(
        model, features, tissue_idx, health_label, device=device
    )
    if not disease_df.empty:
        path = output_dir / "counterfactual_disease_latent.csv"
        disease_df.to_csv(path, index=False)
        reports["disease_latent"] = str(path)

    batch_df = run_batch_latent_swap(
        model, features, tissue_idx, batch_idx, ages,
        target_batch_idx=0, device=device,
    )
    if not batch_df.empty:
        path = output_dir / "counterfactual_batch_swap.csv"
        batch_df.to_csv(path, index=False)
        reports["batch_swap"] = str(path)

    blood_mask = metadata["tissue_family"].astype(str).str.lower().eq("blood")
    if bool(blood_mask.any()):
        blood_output_dir = output_dir / "blood_case"
        blood_output_dir.mkdir(parents=True, exist_ok=True)
        blood_features = features[blood_mask.to_numpy()]
        blood_ages = ages[blood_mask.to_numpy()]
        blood_tissue_idx = tissue_idx[blood_mask.to_numpy()]
        blood_health = health_label[blood_mask.to_numpy()]
        blood_shared = run_shared_latent_perturbation(
            model,
            blood_features,
            blood_tissue_idx,
            blood_ages,
            device=device,
        )
        blood_tissue = pd.DataFrame()
        unique_tissues = np.unique(tissue_idx)
        non_blood_targets = [
            idx
            for idx in unique_tissues
            if idx not in np.unique(blood_tissue_idx)
        ]
        chosen_blood_target_idx = blood_target_tissue_idx
        if (
            chosen_blood_target_idx is not None
            and chosen_blood_target_idx not in non_blood_targets
        ):
            chosen_blood_target_idx = None
        if chosen_blood_target_idx is None and non_blood_targets:
            chosen_blood_target_idx = int(non_blood_targets[0])
        if chosen_blood_target_idx is not None:
            blood_tissue = run_tissue_latent_perturbation(
                model,
                blood_features,
                blood_tissue_idx,
                blood_ages,
                target_tissue_idx=int(chosen_blood_target_idx),
                device=device,
            )
        blood_disease = run_disease_latent_perturbation(
            model,
            blood_features,
            blood_tissue_idx,
            blood_health,
            device=device,
        )
        if not blood_shared.empty:
            blood_shared.to_csv(
                blood_output_dir / "blood_shared_latent.csv",
                index=False,
            )
        if not blood_tissue.empty:
            blood_tissue.to_csv(
                blood_output_dir / "blood_tissue_latent.csv",
                index=False,
            )
        if not blood_disease.empty:
            blood_disease.to_csv(
                blood_output_dir / "blood_disease_latent.csv",
                index=False,
            )
        summary = summarize_blood_counterfactual_case(
            blood_shared, blood_tissue, blood_disease
        )
        if not summary.empty:
            summary = summary.copy()
            summary["source_tissue"] = "blood"
            summary["target_tissue_idx"] = (
                int(chosen_blood_target_idx)
                if chosen_blood_target_idx is not None
                else -1
            )
        summary.to_csv(
            blood_output_dir / "blood_counterfactual_summary.csv",
            index=False,
        )
        reports["blood_case"] = str(
            blood_output_dir / "blood_counterfactual_summary.csv"
        )

    return reports


def summarize_blood_counterfactual_case(
    shared_df: pd.DataFrame,
    tissue_df: pd.DataFrame,
    disease_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if not shared_df.empty:
        monotonic_by_sample = []
        spearman_by_sample = []
        for _, sub_df in shared_df.groupby("sample_idx"):
            ordered = sub_df.sort_values("perturbation_scale")
            diffs = np.diff(ordered["predicted_age"].to_numpy(dtype=float))
            monotonic_by_sample.append(float(np.mean(diffs >= -1e-6)))
            if ordered["predicted_age"].nunique() > 1:
                corr = ordered["perturbation_scale"].corr(
                    ordered["predicted_age"], method="spearman"
                )
                if pd.notna(corr):
                    spearman_by_sample.append(float(corr))
        rows.append(
            {
                "experiment": "shared_latent_monotonicity",
                "metric": "mean_non_decreasing_fraction",
                "value": float(np.mean(monotonic_by_sample)),
            }
        )
        rows.append(
            {
                "experiment": "shared_latent_monotonicity",
                "metric": "pass_rate_gt_0.5",
                "value": float(
                    np.mean(
                        np.asarray(monotonic_by_sample, dtype=float) > 0.5
                    )
                ),
            }
        )
        if spearman_by_sample:
            rows.append(
                {
                    "experiment": "shared_latent_monotonicity",
                    "metric": "mean_spearman_scale_vs_age",
                    "value": float(np.mean(spearman_by_sample)),
                }
            )
    if not tissue_df.empty:
        ordered = tissue_df.groupby("sample_idx")[
            "perturbed_predicted_age"
        ].agg(["min", "max"])
        rows.append(
            {
                "experiment": "tissue_latent_shift",
                "metric": "mean_age_shift_range",
                "value": float((ordered["max"] - ordered["min"]).mean()),
            }
        )
    if not disease_df.empty:
        disease_only = disease_df.loc[disease_df["true_health_label"].eq(0)]
        grouped = disease_only.groupby("perturbation_strength")[
            "disease_prob"
        ].mean()
        if len(grouped) >= 2:
            rows.append(
                {
                    "experiment": "disease_latent_shift",
                    "metric": "delta_disease_probability",
                    "value": float(grouped.iloc[-1] - grouped.iloc[0]),
                }
            )
    return pd.DataFrame(rows)
