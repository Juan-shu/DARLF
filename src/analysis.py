from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


LATENT_PREFIXES = (
    "z_shared_",
    "z_tissue_",
    "z_disease_",
    "z_nuisance_",
)


def _latent_columns(
    frame: pd.DataFrame, prefixes: tuple[str, ...]
) -> list[str]:
    columns: list[str] = []
    for prefix in prefixes:
        columns.extend(
            [col for col in frame.columns if col.startswith(prefix)]
        )
    return columns


def _fit_linear_slope(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot <= 1e-8 else 1.0 - ss_res / ss_tot
    return float(slope), float(r2)


def compute_healthy_manifold_deviation(
    train_predictions: pd.DataFrame,
    eval_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latent_cols = _latent_columns(
        train_predictions, ("z_shared_", "z_tissue_")
    )
    if not latent_cols:
        return pd.DataFrame(), pd.DataFrame()

    healthy_train = train_predictions.loc[
        train_predictions["health_label"].eq(0)
    ].copy()
    if healthy_train.empty:
        return pd.DataFrame(), pd.DataFrame()

    global_center = healthy_train[latent_cols].mean(axis=0)
    tissue_centers = healthy_train.groupby("tissue_family")[
        latent_cols
    ].mean()

    def deviation_for_row(row: pd.Series) -> float:
        tissue = row["tissue_family"]
        center = (
            tissue_centers.loc[tissue]
            if tissue in tissue_centers.index
            else global_center
        )
        return float(
            np.linalg.norm(
                row[latent_cols].to_numpy(dtype=float)
                - center.to_numpy(dtype=float)
            )
        )

    annotated = eval_predictions.copy()
    annotated["healthy_manifold_distance"] = annotated.apply(
        deviation_for_row, axis=1
    )
    summary = (
        annotated.groupby(
            ["health_status", "disease_group"], dropna=False
        )["healthy_manifold_distance"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values(
            ["mean", "count"], ascending=[False, False]
        )
    )

    healthy_distances = annotated.loc[
        annotated["health_status"].eq("Healthy"),
        "healthy_manifold_distance",
    ]
    p_value_map: dict[str, float | None] = {}
    for group in annotated["disease_group"].dropna().unique():
        if group == "Healthy":
            p_value_map[group] = None
            continue
        group_distances = annotated.loc[
            annotated["disease_group"].eq(group),
            "healthy_manifold_distance",
        ]
        if len(group_distances) < 3 or len(healthy_distances) < 3:
            p_value_map[group] = None
        else:
            _, p_val = stats.mannwhitneyu(
                group_distances,
                healthy_distances,
                alternative="two-sided",
            )
            p_value_map[group] = float(p_val)
    summary["p_vs_healthy"] = summary["disease_group"].map(p_value_map)

    return annotated, summary


def compute_tissue_pace_summary(
    predictions: pd.DataFrame,
    min_samples: int = 25,
) -> pd.DataFrame:
    records: list[dict[str, float | str | int]] = []
    for tissue, sub_df in predictions.groupby(
        "tissue_family", dropna=False
    ):
        if len(sub_df) < min_samples:
            continue
        ages = sub_df["age"].to_numpy(dtype=float)
        shared_age = sub_df["shared_age_pred"].to_numpy(dtype=float)
        age_pred = sub_df["age_pred"].to_numpy(dtype=float)
        shared_slope, shared_r2 = _fit_linear_slope(
            ages, shared_age
        )
        pred_slope, pred_r2 = _fit_linear_slope(ages, age_pred)
        if (
            len(sub_df) >= max(min_samples, 40)
            and np.std(ages) > 1e-8
            and np.std(shared_age) > 1e-8
        ):
            quad = np.polyfit(ages, shared_age, deg=2)
            acceleration = float(quad[0])
            switch_idx = -quad[1] / (2 * quad[0]) if abs(quad[0]) > 1e-10 else float("nan")
        else:
            acceleration = 0.0
            switch_idx = float("nan")
        records.append(
            {
                "tissue_family": tissue,
                "n_samples": int(len(sub_df)),
                "age_min": float(np.min(ages)),
                "age_max": float(np.max(ages)),
                "shared_age_slope": shared_slope,
                "shared_age_r2": shared_r2,
                "predicted_age_slope": pred_slope,
                "predicted_age_r2": pred_r2,
                "shared_age_acceleration": acceleration,
                "switch_point_candidate": switch_idx,
            }
        )
    if not records:
        return pd.DataFrame()
    summary = pd.DataFrame(records)
    slope_quantiles = (
        summary["shared_age_slope"]
        .quantile([0.33, 0.66])
        .to_dict()
    )

    def classify_pace(slope: float) -> str:
        if slope <= slope_quantiles.get(0.33, slope):
            return "slow"
        if slope <= slope_quantiles.get(0.66, slope):
            return "intermediate"
        return "fast"

    summary["pace_group"] = summary["shared_age_slope"].map(
        classify_pace
    )
    return summary.sort_values(
        ["shared_age_slope", "n_samples"],
        ascending=[False, False],
    ).reset_index(drop=True)


def compute_disease_deviation_ranking(
    eval_predictions: pd.DataFrame,
) -> pd.DataFrame:
    if "healthy_manifold_distance" not in eval_predictions.columns:
        return pd.DataFrame()
    ranking = (
        eval_predictions.groupby(
            ["disease_group", "health_status"], dropna=False
        )["healthy_manifold_distance"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    return ranking


def compute_shared_tissue_program_correlation(
    train_predictions: pd.DataFrame,
    pathway_names: list[str] | None = None,
) -> pd.DataFrame:
    shared_cols = [
        c for c in train_predictions.columns if c.startswith("z_shared_")
    ]
    if not shared_cols or "tissue_family" not in train_predictions.columns:
        return pd.DataFrame()
    tissue_means = train_predictions.groupby("tissue_family")[
        shared_cols
    ].mean()
    pc = tissue_means.apply(
        lambda row: float(
            np.dot(row.to_numpy(), row.to_numpy())
        ),
        axis=1,
    )
    pc.name = "shared_program_strength"
    result = tissue_means.copy()
    result["shared_program_strength"] = pc
    return result.sort_values(
        "shared_program_strength", ascending=False
    )


def compute_shared_aging_pathway_association(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    pathway_cols = [
        col for col in predictions.columns if col.startswith("pathway_")
    ]
    if not pathway_cols or "shared_age_pred" not in predictions.columns:
        return pd.DataFrame()
    rows: list[dict[str, float | str]] = []
    shared = predictions["shared_age_pred"].to_numpy(dtype=float)
    for column in pathway_cols:
        values = predictions[column].to_numpy(dtype=float)
        if np.std(values) < 1e-8 or np.std(shared) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(values, shared)[0, 1])
        rows.append(
            {
                "pathway": column.replace("pathway_", ""),
                "corr_with_shared_age": corr,
                "abs_corr": abs(corr),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "abs_corr", ascending=False
    ).reset_index(drop=True)


def compute_blood_brain_conservation(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    pathway_cols = [
        col for col in predictions.columns if col.startswith("pathway_")
    ]
    if not pathway_cols or "tissue_family" not in predictions.columns:
        return pd.DataFrame()
    subset = predictions.loc[
        predictions["tissue_family"].isin(["blood", "brain"])
    ].copy()
    if subset["tissue_family"].nunique() < 2:
        return pd.DataFrame()
    rows: list[dict[str, float | str]] = []
    for column in pathway_cols:
        by_tissue = (
            subset.groupby("tissue_family")[column]
            .mean()
            .to_dict()
        )
        if "blood" in by_tissue and "brain" in by_tissue:
            rows.append(
                {
                    "pathway": column.replace("pathway_", ""),
                    "blood_mean": float(by_tissue["blood"]),
                    "brain_mean": float(by_tissue["brain"]),
                    "abs_gap": float(
                        abs(by_tissue["blood"] - by_tissue["brain"])
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        "abs_gap", ascending=True
    ).reset_index(drop=True)


def compute_disease_pathway_activation(
    eval_predictions: pd.DataFrame,
) -> pd.DataFrame:
    pathway_cols = [
        col for col in eval_predictions.columns if col.startswith("pathway_")
    ]
    if (
        not pathway_cols
        or "healthy_manifold_distance" not in eval_predictions.columns
        or "disease_group" not in eval_predictions.columns
    ):
        return pd.DataFrame()
    rows: list[dict[str, float | str]] = []
    healthy = eval_predictions.loc[
        eval_predictions["health_status"].eq("Healthy")
    ].copy()
    if healthy.empty:
        return pd.DataFrame()
    healthy_means = healthy[pathway_cols].mean(axis=0)
    for disease_group, sub_df in eval_predictions.groupby(
        "disease_group", dropna=False
    ):
        if disease_group in {"Healthy", "Unknown"} or len(sub_df) < 5:
            continue
        for column in pathway_cols:
            delta = float(sub_df[column].mean() - healthy_means[column])
            if np.std(sub_df[column].to_numpy(dtype=float)) < 1e-8 or np.std(
                sub_df["healthy_manifold_distance"].to_numpy(dtype=float)
            ) < 1e-8:
                corr = 0.0
            else:
                corr = float(
                    np.corrcoef(
                        sub_df[column].to_numpy(dtype=float),
                        sub_df["healthy_manifold_distance"].to_numpy(
                            dtype=float
                        ),
                    )[0, 1]
                )
            rows.append(
                {
                    "disease_group": disease_group,
                    "pathway": column.replace("pathway_", ""),
                    "mean_delta_vs_healthy": delta,
                    "corr_with_deviation_distance": corr,
                    "score": abs(delta) + abs(corr),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["disease_group", "score"], ascending=[True, False]
    ).reset_index(drop=True)


def compute_sex_pathway_differences(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    pathway_cols = [
        col for col in predictions.columns if col.startswith("pathway_")
    ]
    if not pathway_cols or "sex" not in predictions.columns:
        return pd.DataFrame()
    subset = predictions.copy()
    subset["sex"] = subset["sex"].astype(str).str.lower()
    subset = subset.loc[subset["sex"].isin(["male", "female"])].copy()
    if "health_status" in subset.columns and subset[
        "health_status"
    ].eq("Healthy").any():
        subset = subset.loc[subset["health_status"].eq("Healthy")].copy()
    if subset["sex"].nunique() < 2:
        return pd.DataFrame()
    male = subset.loc[subset["sex"].eq("male")]
    female = subset.loc[subset["sex"].eq("female")]
    rows: list[dict[str, float | str | int]] = []
    for column in pathway_cols:
        male_values = male[column].to_numpy(dtype=float)
        female_values = female[column].to_numpy(dtype=float)
        if len(male_values) < 5 or len(female_values) < 5:
            continue
        try:
            _, p_val = stats.mannwhitneyu(
                male_values,
                female_values,
                alternative="two-sided",
            )
        except ValueError:
            p_val = np.nan
        rows.append(
            {
                "pathway": column.replace("pathway_", ""),
                "male_mean": float(np.mean(male_values)),
                "female_mean": float(np.mean(female_values)),
                "male_minus_female": float(
                    np.mean(male_values) - np.mean(female_values)
                ),
                "abs_delta": float(
                    abs(np.mean(male_values) - np.mean(female_values))
                ),
                "p_value": float(p_val)
                if np.isfinite(p_val)
                else np.nan,
                "n_male": int(len(male_values)),
                "n_female": int(len(female_values)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "abs_delta", ascending=False
    ).reset_index(drop=True)


def compute_sex_tissue_latent_summary(
    predictions: pd.DataFrame,
    min_samples_per_sex: int = 15,
) -> pd.DataFrame:
    required_cols = {
        "tissue_family",
        "sex",
        "age_pred",
        "shared_age_pred",
    }
    if not required_cols.issubset(predictions.columns):
        return pd.DataFrame()
    subset = predictions.copy()
    subset["sex"] = subset["sex"].astype(str).str.lower()
    subset = subset.loc[subset["sex"].isin(["male", "female"])].copy()
    if subset.empty:
        return pd.DataFrame()
    disease_latent_cols = [
        col for col in subset.columns if col.startswith("z_disease_")
    ]
    if disease_latent_cols:
        subset["z_disease_norm"] = np.linalg.norm(
            subset[disease_latent_cols].to_numpy(dtype=float), axis=1
        )
    else:
        subset["z_disease_norm"] = 0.0
    records: list[dict[str, float | str | int]] = []
    for tissue, tissue_df in subset.groupby("tissue_family", dropna=False):
        male = tissue_df.loc[tissue_df["sex"].eq("male")]
        female = tissue_df.loc[tissue_df["sex"].eq("female")]
        if (
            len(male) < int(min_samples_per_sex)
            or len(female) < int(min_samples_per_sex)
        ):
            continue
        for target_col in [
            "age_pred",
            "shared_age_pred",
            "z_disease_norm",
        ]:
            male_values = male[target_col].to_numpy(dtype=float)
            female_values = female[target_col].to_numpy(dtype=float)
            try:
                _, p_val = stats.mannwhitneyu(
                    male_values,
                    female_values,
                    alternative="two-sided",
                )
            except ValueError:
                p_val = np.nan
            records.append(
                {
                    "tissue_family": tissue,
                    "metric": target_col,
                    "male_mean": float(np.mean(male_values)),
                    "female_mean": float(np.mean(female_values)),
                    "male_minus_female": float(
                        np.mean(male_values)
                        - np.mean(female_values)
                    ),
                    "abs_delta": float(
                        abs(
                            np.mean(male_values)
                            - np.mean(female_values)
                        )
                    ),
                    "p_value": float(p_val)
                    if np.isfinite(p_val)
                    else np.nan,
                    "n_male": int(len(male)),
                    "n_female": int(len(female)),
                }
            )
    return pd.DataFrame(records).sort_values(
        ["metric", "abs_delta"], ascending=[True, False]
    ).reset_index(drop=True)


def run_posthoc_analyses(
    train_predictions: pd.DataFrame,
    eval_predictions: pd.DataFrame,
    output_dir: str | Path,
    pathway_names: list[str] | None = None,
) -> dict[str, int | float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_with_distance, deviation_summary = (
        compute_healthy_manifold_deviation(
            train_predictions, eval_predictions
        )
    )
    all_preds = pd.concat(
        [train_predictions, eval_predictions],
        axis=0,
        ignore_index=True,
    )
    tissue_pace_summary = compute_tissue_pace_summary(all_preds)
    disease_ranking = compute_disease_deviation_ranking(
        eval_with_distance
    )
    program_corr = compute_shared_tissue_program_correlation(
        train_predictions, pathway_names
    )
    shared_pathway_assoc = compute_shared_aging_pathway_association(
        train_predictions
    )
    blood_brain_conservation = compute_blood_brain_conservation(
        train_predictions
    )
    disease_pathway_activation = compute_disease_pathway_activation(
        eval_with_distance
    )
    sex_pathway_differences = compute_sex_pathway_differences(
        train_predictions
    )
    sex_tissue_summary = compute_sex_tissue_latent_summary(all_preds)

    if not eval_with_distance.empty:
        eval_with_distance.to_csv(
            output_dir / "eval_predictions_with_manifold_distance.csv",
            index=False,
        )
    if not deviation_summary.empty:
        deviation_summary.to_csv(
            output_dir / "healthy_manifold_deviation_summary.csv",
            index=False,
        )
    if not tissue_pace_summary.empty:
        tissue_pace_summary.to_csv(
            output_dir / "tissue_pace_summary.csv", index=False
        )
    if not disease_ranking.empty:
        disease_ranking.to_csv(
            output_dir / "disease_deviation_ranking.csv", index=False
        )
    if not program_corr.empty:
        program_corr.to_csv(
            output_dir / "shared_tissue_program_correlation.csv",
            index=True,
        )
    if not shared_pathway_assoc.empty:
        shared_pathway_assoc.to_csv(
            output_dir / "shared_aging_pathway_association.csv",
            index=False,
        )
    if not blood_brain_conservation.empty:
        blood_brain_conservation.to_csv(
            output_dir / "blood_brain_shared_program_conservation.csv",
            index=False,
        )
    if not disease_pathway_activation.empty:
        disease_pathway_activation.to_csv(
            output_dir / "disease_pathway_activation.csv",
            index=False,
        )
    if not sex_pathway_differences.empty:
        sex_pathway_differences.to_csv(
            output_dir / "sex_pathway_differences.csv",
            index=False,
        )
    if not sex_tissue_summary.empty:
        sex_tissue_summary.to_csv(
            output_dir / "sex_tissue_latent_summary.csv",
            index=False,
        )

    return {
        "n_eval_samples_for_deviation": int(
            len(eval_with_distance)
        ),
        "n_deviation_groups": int(len(deviation_summary)),
        "n_tissues_with_pace_summary": int(
            len(tissue_pace_summary)
        ),
        "n_disease_groups_ranked": int(len(disease_ranking)),
        "n_tissues_with_program_correlation": int(
            len(program_corr)
        ),
        "n_shared_pathways_ranked": int(len(shared_pathway_assoc)),
        "n_blood_brain_pathways": int(len(blood_brain_conservation)),
        "n_disease_pathway_rows": int(len(disease_pathway_activation)),
        "n_sex_pathway_rows": int(len(sex_pathway_differences)),
        "n_sex_tissue_rows": int(len(sex_tissue_summary)),
    }
