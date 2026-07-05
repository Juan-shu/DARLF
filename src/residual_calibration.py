from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ResidualCalibrationResult:
    selected_model: str
    raw_val_mae: float
    calibrated_val_mae: float
    raw_test_mae: float
    calibrated_test_mae: float
    selected_feature_count: int
    train_predictions: pd.DataFrame
    val_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    summary: dict[str, object]


def _feature_candidates(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [
        col
        for col in [
            "tissue_family",
            "disease_group",
            "health_status",
            "sex",
            "dataset",
            "age_bin",
        ]
        if col in frame.columns
    ]
    numeric = [
        col
        for col in frame.columns
        if col in {"age_pred", "shared_age_pred"}
        or col.startswith("pathway_")
        or col.startswith("cell_")
        or col.startswith("z_shared_")
        or col.startswith("z_tissue_")
        or col.startswith("z_disease_")
        or col.startswith("z_nuisance_")
        or col.startswith("z_raw_context_")
    ]
    return categorical, numeric


def _select_numeric_features(
    train_df: pd.DataFrame,
    numeric_columns: list[str],
    *,
    top_k: int,
) -> list[str]:
    residual = (
        train_df["age"].to_numpy(dtype=float)
        - train_df["age_pred"].to_numpy(dtype=float)
    )
    always_keep = [
        col
        for col in ["age_pred", "shared_age_pred"]
        if col in numeric_columns
    ]
    ranked: list[tuple[str, float]] = []
    for column in numeric_columns:
        if column in always_keep:
            continue
        values = pd.to_numeric(
            train_df[column], errors="coerce"
        ).to_numpy(dtype=float)
        if (
            np.isnan(values).all()
            or np.nanstd(values) < 1e-8
            or np.std(residual) < 1e-8
        ):
            corr = 0.0
        else:
            corr = float(
                np.corrcoef(
                    np.nan_to_num(values, nan=np.nanmedian(values)),
                    residual,
                )[0, 1]
            )
            if not np.isfinite(corr):
                corr = 0.0
        ranked.append((column, abs(corr)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    selected = always_keep + [
        column for column, _ in ranked[: max(0, int(top_k) - len(always_keep))]
    ]
    return selected


def _build_matrix(
    frame: pd.DataFrame,
    *,
    categorical_columns: list[str],
    numeric_columns: list[str],
    category_levels: dict[str, list[str]],
    numeric_fill: dict[str, float],
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    if categorical_columns:
        cat_frame = frame[categorical_columns].copy()
        for column in categorical_columns:
            cat_frame[column] = (
                cat_frame[column]
                .astype(str)
                .fillna("unknown")
            )
            categories = category_levels.get(column, [])
            if categories:
                cat_frame[column] = pd.Categorical(
                    cat_frame[column],
                    categories=categories,
                )
        cat_encoded = pd.get_dummies(
            cat_frame,
            columns=categorical_columns,
            dummy_na=False,
            dtype=np.float32,
        )
        blocks.append(cat_encoded)
    if numeric_columns:
        num_frame = frame[numeric_columns].copy()
        for column in numeric_columns:
            fill_value = float(numeric_fill.get(column, 0.0))
            num_frame[column] = pd.to_numeric(
                num_frame[column], errors="coerce"
            ).fillna(fill_value)
        blocks.append(num_frame.astype(np.float32))
    if not blocks:
        return pd.DataFrame(index=frame.index)
    matrix = pd.concat(blocks, axis=1)
    return matrix


def _regression_metrics(
    age_true: np.ndarray, age_pred: np.ndarray
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(age_true, age_pred)),
        "r2": float(r2_score(age_true, age_pred)),
    }


def _fit_and_predict(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_train_eval: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model_name == "ridge_alpha_1":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        model = Ridge(alpha=1.0, random_state=0)
    elif model_name == "ridge_alpha_0p3":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        model = Ridge(alpha=0.3, random_state=0)
    elif model_name == "ridge_alpha_3":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        model = Ridge(alpha=3.0, random_state=0)
    elif model_name == "ridge_alpha_10":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        model = Ridge(alpha=10.0, random_state=0)
    elif model_name == "huber":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        model = HuberRegressor(
            alpha=1e-4, epsilon=1.35, max_iter=1000
        )
    elif model_name == "gbr_depth2":
        model = GradientBoostingRegressor(
            n_estimators=240,
            learning_rate=0.03,
            max_depth=2,
            random_state=0,
            loss="squared_error",
        )
    elif model_name == "gbr_depth3":
        model = GradientBoostingRegressor(
            n_estimators=260,
            learning_rate=0.03,
            max_depth=3,
            random_state=0,
            loss="squared_error",
        )
    elif model_name == "gbr_depth4":
        model = GradientBoostingRegressor(
            n_estimators=320,
            learning_rate=0.025,
            max_depth=4,
            random_state=0,
            loss="squared_error",
        )
    elif model_name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=320,
            max_depth=10,
            min_samples_leaf=4,
            random_state=0,
            n_jobs=-1,
        )
    elif model_name == "extra_trees_deep":
        model = ExtraTreesRegressor(
            n_estimators=480,
            max_depth=None,
            min_samples_leaf=2,
            random_state=0,
            n_jobs=-1,
        )
    elif model_name == "extra_trees_ultra":
        model = ExtraTreesRegressor(
            n_estimators=720,
            max_depth=None,
            min_samples_leaf=1,
            random_state=0,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported calibration model: {model_name}")
    model.fit(x_train, y_train)
    return (
        model.predict(x_train_eval),
        model.predict(x_val),
        model.predict(x_test),
    )


def _search_best_model(
    *,
    candidate_models: list[str],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    base_val_age: np.ndarray,
    val_true: np.ndarray,
    base_test_age: np.ndarray,
    test_true: np.ndarray,
    current_best_val_mae: float,
    min_improvement: float,
    stage_name: str,
    enable_pairwise_blends: bool = False,
    blend_weights: list[float] | None = None,
) -> tuple[
    str,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, float | str]],
]:
    best_model = "none"
    best_val_mae = float(current_best_val_mae)
    best_train_residual = np.zeros(len(x_train), dtype=np.float32)
    best_val_residual = np.zeros(len(x_val), dtype=np.float32)
    best_test_residual = np.zeros(len(x_test), dtype=np.float32)
    candidate_rows: list[dict[str, float | str]] = []
    successful_predictions: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    x_train_np = x_train.to_numpy(dtype=np.float32, copy=False)
    x_val_np = x_val.to_numpy(dtype=np.float32, copy=False)
    x_test_np = x_test.to_numpy(dtype=np.float32, copy=False)
    y_train_np = y_train.astype(np.float32, copy=False)
    for model_name in candidate_models:
        try:
            (
                pred_train_residual,
                pred_val_residual,
                pred_test_residual,
            ) = _fit_and_predict(
                model_name,
                x_train_np,
                y_train_np,
                x_train_np,
                x_val_np,
                x_test_np,
            )
        except Exception as exc:  # pragma: no cover - defensive
            candidate_rows.append(
                {
                    "stage": stage_name,
                    "model": model_name,
                    "val_mae": float("nan"),
                    "test_mae": float("nan"),
                    "error": str(exc),
                }
            )
            continue
        calibrated_val_age = base_val_age + pred_val_residual
        calibrated_test_age = base_test_age + pred_test_residual
        val_mae = float(mean_absolute_error(val_true, calibrated_val_age))
        test_mae = float(
            mean_absolute_error(test_true, calibrated_test_age)
        )
        candidate_rows.append(
            {
                "stage": stage_name,
                "model": model_name,
                "val_mae": val_mae,
                "test_mae": test_mae,
            }
        )
        successful_predictions[model_name] = (
            pred_train_residual.astype(np.float32),
            pred_val_residual.astype(np.float32),
            pred_test_residual.astype(np.float32),
        )
        if val_mae < best_val_mae - min_improvement:
            best_model = model_name
            best_val_mae = val_mae
            best_train_residual = pred_train_residual.astype(np.float32)
            best_val_residual = pred_val_residual.astype(np.float32)
            best_test_residual = pred_test_residual.astype(np.float32)
    if enable_pairwise_blends and len(successful_predictions) >= 2:
        blend_weights = (
            [0.5] if not blend_weights else list(blend_weights)
        )
        for model_a, model_b in itertools.combinations(
            successful_predictions.keys(), 2
        ):
            pred_a = successful_predictions[model_a]
            pred_b = successful_predictions[model_b]
            for alpha in blend_weights:
                alpha = float(alpha)
                beta = 1.0 - alpha
                pred_train_residual = (
                    alpha * pred_a[0] + beta * pred_b[0]
                )
                pred_val_residual = (
                    alpha * pred_a[1] + beta * pred_b[1]
                )
                pred_test_residual = (
                    alpha * pred_a[2] + beta * pred_b[2]
                )
                model_name = (
                    f"blend:{alpha:.2f}*{model_a}+{beta:.2f}*{model_b}"
                )
                calibrated_val_age = base_val_age + pred_val_residual
                calibrated_test_age = base_test_age + pred_test_residual
                val_mae = float(
                    mean_absolute_error(val_true, calibrated_val_age)
                )
                test_mae = float(
                    mean_absolute_error(test_true, calibrated_test_age)
                )
                candidate_rows.append(
                    {
                        "stage": stage_name,
                        "model": model_name,
                        "val_mae": val_mae,
                        "test_mae": test_mae,
                    }
                )
                if val_mae < best_val_mae - min_improvement:
                    best_model = model_name
                    best_val_mae = val_mae
                    best_train_residual = pred_train_residual.astype(
                        np.float32
                    )
                    best_val_residual = pred_val_residual.astype(
                        np.float32
                    )
                    best_test_residual = pred_test_residual.astype(
                        np.float32
                    )
    return (
        best_model,
        best_val_mae,
        best_train_residual,
        best_val_residual,
        best_test_residual,
        candidate_rows,
    )


def run_residual_calibration(
    train_predictions: pd.DataFrame,
    val_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    *,
    output_dir: str | Path,
    config: dict | None = None,
) -> ResidualCalibrationResult | None:
    config = config or {}
    if not config.get("enabled", False):
        return None

    output_dir = Path(output_dir)
    age_bin_edges = list(
        config.get("age_bin_edges", [0, 20, 40, 60, 80, 200])
    )
    age_bin_labels = list(
        config.get(
            "age_bin_labels",
            ["<=20", "20-40", "40-60", "60-80", "80+"],
        )
    )
    for frame in (train_predictions, val_predictions, test_predictions):
        if "age_bin" not in frame.columns:
            frame["age_bin"] = pd.cut(
                frame["age"],
                bins=age_bin_edges,
                labels=age_bin_labels,
                include_lowest=True,
                right=True,
            ).astype(str)
    top_k = int(config.get("top_numeric_features", 96))
    min_improvement = float(config.get("min_improvement", 0.01))
    candidate_models = [
        str(name)
        for name in config.get(
            "candidate_models",
            [
                "ridge_alpha_1",
                "ridge_alpha_0p3",
                "ridge_alpha_3",
                "ridge_alpha_10",
                "huber",
                "gbr_depth2",
                "gbr_depth3",
                "gbr_depth4",
                "extra_trees",
                "extra_trees_deep",
                "extra_trees_ultra",
            ],
        )
    ]
    stage2_enabled = bool(config.get("stage2_enabled", True))
    stage2_candidate_models = [
        str(name)
        for name in config.get(
            "stage2_candidate_models",
            candidate_models,
        )
    ]
    stage2_min_improvement = float(
        config.get("stage2_min_improvement", 0.0)
    )
    enable_pairwise_blends = bool(
        config.get("enable_pairwise_blends", False)
    )
    blend_weights = [
        float(value)
        for value in config.get("blend_weights", [0.5])
    ]
    targeted_group_rules: list[dict[str, object]] = list(
        config.get(
            "targeted_group_rules",
            [
                {
                    "column": "tissue_family",
                    "values": ["eye", "synovial biopsy", "liver"],
                },
                {
                    "column": "disease_group",
                    "values": [
                        "Ocular degeneration",
                        "Autoimmune/Inflammatory",
                    ],
                },
            ],
        )
    )
    min_group_train = int(config.get("min_group_train_samples", 30))
    min_group_val = int(config.get("min_group_val_samples", 8))
    min_group_improvement = float(
        config.get("min_group_improvement", 0.02)
    )

    categorical_columns, numeric_columns = _feature_candidates(
        train_predictions
    )
    selected_numeric = _select_numeric_features(
        train_predictions, numeric_columns, top_k=top_k
    )
    category_levels = {
        column: sorted(
            train_predictions[column]
            .astype(str)
            .fillna("unknown")
            .unique()
            .tolist()
        )
        for column in categorical_columns
    }
    numeric_fill = {
        column: float(
            pd.to_numeric(
                train_predictions[column], errors="coerce"
            ).median()
        )
        for column in selected_numeric
    }

    x_train = _build_matrix(
        train_predictions,
        categorical_columns=categorical_columns,
        numeric_columns=selected_numeric,
        category_levels=category_levels,
        numeric_fill=numeric_fill,
    )
    x_val = _build_matrix(
        val_predictions,
        categorical_columns=categorical_columns,
        numeric_columns=selected_numeric,
        category_levels=category_levels,
        numeric_fill=numeric_fill,
    )
    x_test = _build_matrix(
        test_predictions,
        categorical_columns=categorical_columns,
        numeric_columns=selected_numeric,
        category_levels=category_levels,
        numeric_fill=numeric_fill,
    )
    x_val = x_val.reindex(columns=x_train.columns, fill_value=0.0)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)

    y_train = (
        train_predictions["age"].to_numpy(dtype=float)
        - train_predictions["age_pred"].to_numpy(dtype=float)
    )
    raw_val_age = val_predictions["age_pred"].to_numpy(dtype=float)
    raw_test_age = test_predictions["age_pred"].to_numpy(dtype=float)
    val_true = val_predictions["age"].to_numpy(dtype=float)
    test_true = test_predictions["age"].to_numpy(dtype=float)
    raw_val_mae = float(mean_absolute_error(val_true, raw_val_age))
    raw_test_mae = float(mean_absolute_error(test_true, raw_test_age))

    (
        best_model,
        best_val_mae,
        best_train_residual,
        best_val_residual,
        best_test_residual,
        candidate_rows,
    ) = _search_best_model(
        candidate_models=candidate_models,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        x_test=x_test,
        base_val_age=raw_val_age,
        val_true=val_true,
        base_test_age=raw_test_age,
        test_true=test_true,
        current_best_val_mae=raw_val_mae,
        min_improvement=min_improvement,
        stage_name="global_stage1",
        enable_pairwise_blends=enable_pairwise_blends,
        blend_weights=blend_weights,
    )

    val_calibrated = val_predictions.copy()
    test_calibrated = test_predictions.copy()
    train_calibrated = train_predictions.copy()
    train_calibrated["age_pred_raw"] = train_calibrated["age_pred"]
    train_calibrated["residual_pred_calibrated"] = best_train_residual
    train_calibrated["age_pred_calibrated"] = (
        train_calibrated["age_pred"] + best_train_residual
    )
    train_calibrated["abs_err_raw"] = (
        train_calibrated["age_pred"] - train_calibrated["age"]
    ).abs()
    train_calibrated["abs_err_calibrated"] = (
        train_calibrated["age_pred_calibrated"]
        - train_calibrated["age"]
    ).abs()

    val_calibrated["age_pred_raw"] = val_calibrated["age_pred"]
    val_calibrated["residual_pred_calibrated"] = best_val_residual
    val_calibrated["age_pred_calibrated"] = (
        val_calibrated["age_pred_raw"] + best_val_residual
    )
    val_calibrated["abs_err_raw"] = (
        val_calibrated["age_pred_raw"] - val_calibrated["age"]
    ).abs()
    val_calibrated["abs_err_calibrated"] = (
        val_calibrated["age_pred_calibrated"] - val_calibrated["age"]
    ).abs()

    test_calibrated["age_pred_raw"] = test_calibrated["age_pred"]
    test_calibrated["residual_pred_calibrated"] = best_test_residual
    test_calibrated["age_pred_calibrated"] = (
        test_calibrated["age_pred_raw"] + best_test_residual
    )
    test_calibrated["abs_err_raw"] = (
        test_calibrated["age_pred_raw"] - test_calibrated["age"]
    ).abs()
    test_calibrated["abs_err_calibrated"] = (
        test_calibrated["age_pred_calibrated"] - test_calibrated["age"]
    ).abs()

    second_stage_model = "none"
    if stage2_enabled:
        y_train_stage2 = (
            train_predictions["age"].to_numpy(dtype=float)
            - train_calibrated["age_pred_calibrated"].to_numpy(dtype=float)
        )
        (
            second_stage_model,
            second_stage_val_mae,
            stage2_train_residual,
            stage2_val_residual,
            stage2_test_residual,
            stage2_rows,
        ) = _search_best_model(
            candidate_models=stage2_candidate_models,
            x_train=x_train,
            y_train=y_train_stage2,
            x_val=x_val,
            x_test=x_test,
            base_val_age=val_calibrated["age_pred_calibrated"].to_numpy(
                dtype=float
            ),
            val_true=val_true,
            base_test_age=test_calibrated["age_pred_calibrated"].to_numpy(
                dtype=float
            ),
            test_true=test_true,
            current_best_val_mae=best_val_mae,
            min_improvement=stage2_min_improvement,
            stage_name="global_stage2",
            enable_pairwise_blends=enable_pairwise_blends,
            blend_weights=blend_weights,
        )
        candidate_rows.extend(stage2_rows)
        if second_stage_model != "none":
            best_val_mae = second_stage_val_mae
            train_calibrated["residual_pred_calibrated"] = (
                train_calibrated["residual_pred_calibrated"].to_numpy(
                    dtype=np.float32
                )
                + stage2_train_residual
            )
            val_calibrated["residual_pred_calibrated"] = (
                val_calibrated["residual_pred_calibrated"].to_numpy(
                    dtype=np.float32
                )
                + stage2_val_residual
            )
            test_calibrated["residual_pred_calibrated"] = (
                test_calibrated["residual_pred_calibrated"].to_numpy(
                    dtype=np.float32
                )
                + stage2_test_residual
            )
            for frame in (
                train_calibrated,
                val_calibrated,
                test_calibrated,
            ):
                frame["age_pred_calibrated"] = (
                    frame["age_pred_raw"]
                    + frame["residual_pred_calibrated"]
                )
                frame["abs_err_calibrated"] = (
                    frame["age_pred_calibrated"] - frame["age"]
                ).abs()

    group_override_rows: list[dict[str, object]] = []
    for rule in targeted_group_rules:
        column = str(rule.get("column", "")).strip()
        values = [str(value) for value in rule.get("values", [])]
        if (
            not column
            or column not in train_predictions.columns
            or column not in val_predictions.columns
            or column not in test_predictions.columns
        ):
            continue
        for value in values:
            train_mask = (
                train_predictions[column].astype(str).eq(value).to_numpy()
            )
            val_mask = (
                val_predictions[column].astype(str).eq(value).to_numpy()
            )
            test_mask = (
                test_predictions[column].astype(str).eq(value).to_numpy()
            )
            if (
                int(train_mask.sum()) < min_group_train
                or int(val_mask.sum()) < min_group_val
            ):
                continue
            group_best_model = "none"
            current_group_val_mae = float(
                mean_absolute_error(
                    val_predictions.loc[val_mask, "age"].to_numpy(
                        dtype=float
                    ),
                    val_calibrated.loc[
                        val_mask, "age_pred_calibrated"
                    ].to_numpy(dtype=float),
                )
            )
            group_best_val_mae = current_group_val_mae
            group_train_pred = train_calibrated.loc[
                train_mask, "residual_pred_calibrated"
            ].to_numpy(dtype=np.float32, copy=True)
            group_val_pred = val_calibrated.loc[
                val_mask, "residual_pred_calibrated"
            ].to_numpy(dtype=np.float32, copy=True)
            group_test_pred = test_calibrated.loc[
                test_mask, "residual_pred_calibrated"
            ].to_numpy(dtype=np.float32, copy=True)
            for model_name in candidate_models:
                try:
                    (
                        candidate_train_residual,
                        candidate_val_residual,
                        candidate_test_residual,
                    ) = _fit_and_predict(
                        model_name,
                        x_train.loc[train_mask].to_numpy(
                            dtype=np.float32, copy=False
                        ),
                        (
                            train_predictions.loc[train_mask, "age"].to_numpy(
                                dtype=float
                            )
                            - train_calibrated.loc[
                                train_mask, "age_pred_calibrated"
                            ].to_numpy(dtype=float)
                        ).astype(
                            np.float32, copy=False
                        ),
                        x_train.loc[train_mask].to_numpy(
                            dtype=np.float32, copy=False
                        ),
                        x_val.loc[val_mask].to_numpy(
                            dtype=np.float32, copy=False
                        ),
                        x_test.loc[test_mask].to_numpy(
                            dtype=np.float32, copy=False
                        ),
                    )
                except Exception:
                    continue
                group_val_mae = float(
                    mean_absolute_error(
                        val_predictions.loc[val_mask, "age"].to_numpy(
                            dtype=float
                        ),
                        val_calibrated.loc[
                            val_mask, "age_pred_calibrated"
                        ].to_numpy(dtype=float)
                        + candidate_val_residual,
                    )
                )
                if (
                    group_val_mae
                    < group_best_val_mae - min_group_improvement
                ):
                    group_best_model = model_name
                    group_best_val_mae = group_val_mae
                    group_train_pred = candidate_train_residual.astype(
                        np.float32
                    )
                    group_val_pred = candidate_val_residual.astype(
                        np.float32
                    )
                    group_test_pred = candidate_test_residual.astype(
                        np.float32
                    )
            if group_best_model != "none":
                train_calibrated.loc[
                    train_mask, "residual_pred_calibrated"
                ] = group_train_pred
                val_calibrated.loc[
                    val_mask, "residual_pred_calibrated"
                ] = group_val_pred
                test_calibrated.loc[
                    test_mask, "residual_pred_calibrated"
                ] = group_test_pred
                train_calibrated.loc[
                    train_mask, "age_pred_calibrated"
                ] = (
                    train_calibrated.loc[train_mask, "age_pred_raw"]
                    + group_train_pred
                )
                val_calibrated.loc[
                    val_mask, "age_pred_calibrated"
                ] = (
                    val_calibrated.loc[val_mask, "age_pred_raw"]
                    + group_val_pred
                )
                test_calibrated.loc[
                    test_mask, "age_pred_calibrated"
                ] = (
                    test_calibrated.loc[test_mask, "age_pred_raw"]
                    + group_test_pred
                )
                train_calibrated.loc[
                    train_mask, "abs_err_calibrated"
                ] = (
                    train_calibrated.loc[
                        train_mask, "age_pred_calibrated"
                    ]
                    - train_calibrated.loc[train_mask, "age"]
                ).abs()
                val_calibrated.loc[
                    val_mask, "abs_err_calibrated"
                ] = (
                    val_calibrated.loc[
                        val_mask, "age_pred_calibrated"
                    ]
                    - val_calibrated.loc[val_mask, "age"]
                ).abs()
                test_calibrated.loc[
                    test_mask, "abs_err_calibrated"
                ] = (
                    test_calibrated.loc[
                        test_mask, "age_pred_calibrated"
                    ]
                    - test_calibrated.loc[test_mask, "age"]
                ).abs()
                group_override_rows.append(
                    {
                        "column": column,
                        "value": value,
                        "selected_model": group_best_model,
                        "n_train": int(train_mask.sum()),
                        "n_val": int(val_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "val_mae_before": current_group_val_mae,
                        "val_mae_after": group_best_val_mae,
                    }
                )

    val_metrics = _regression_metrics(
        val_true,
        val_calibrated["age_pred_calibrated"].to_numpy(dtype=float),
    )
    test_metrics = _regression_metrics(
        test_true,
        test_calibrated["age_pred_calibrated"].to_numpy(dtype=float),
    )

    summary: dict[str, object] = {
        "enabled": True,
        "selected_model": best_model,
        "second_stage_model": second_stage_model,
        "raw_validation_mae": raw_val_mae,
        "calibrated_validation_mae": float(val_metrics["mae"]),
        "raw_test_mae": raw_test_mae,
        "calibrated_test_mae": float(test_metrics["mae"]),
        "selected_feature_count": int(len(x_train.columns)),
        "selected_numeric_features": selected_numeric,
        "categorical_features": categorical_columns,
        "candidate_results": candidate_rows,
        "group_overrides": group_override_rows,
    }
    pd.DataFrame(candidate_rows).to_csv(
        output_dir / "residual_calibration_candidates.csv",
        index=False,
    )
    train_calibrated.to_csv(
        output_dir / "train_predictions_calibrated.csv", index=False
    )
    val_calibrated.to_csv(
        output_dir / "val_predictions_calibrated.csv", index=False
    )
    test_calibrated.to_csv(
        output_dir / "test_predictions_calibrated.csv", index=False
    )
    return ResidualCalibrationResult(
        selected_model=best_model,
        raw_val_mae=raw_val_mae,
        calibrated_val_mae=float(val_metrics["mae"]),
        raw_test_mae=raw_test_mae,
        calibrated_test_mae=float(test_metrics["mae"]),
        selected_feature_count=int(len(x_train.columns)),
        train_predictions=train_calibrated,
        val_predictions=val_calibrated,
        test_predictions=test_calibrated,
        summary=summary,
    )
