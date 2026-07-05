from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .trainer import AgingDataset, set_seed


def train_elasticnet_baseline(
    train_features: np.ndarray,
    train_ages: np.ndarray,
    test_features: np.ndarray,
    test_ages: np.ndarray,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)
    y_train = train_ages
    y_test = test_ages
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=cv_folds,
        random_state=seed,
        max_iter=5000,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "alpha": float(model.alpha_),
        "l1_ratio": float(model.l1_ratio_),
    }


def train_mlp_baseline(
    train_features: np.ndarray,
    train_ages: np.ndarray,
    test_features: np.ndarray,
    test_ages: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (256, 128, 64),
    seed: int = 42,
) -> dict[str, float]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)
    y_train = train_ages
    y_test = test_ages
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_iter": int(model.n_iter_),
    }


def train_maple_like_baseline(
    train_features: np.ndarray,
    train_ages: np.ndarray,
    test_features: np.ndarray,
    test_ages: np.ndarray,
    n_pairs: int = 5000,
    n_anchors: int = 64,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)
    y_train = train_ages
    y_test = test_ages
    n_samples = X_train.shape[0]
    n_pairs = min(n_pairs, n_samples * (n_samples - 1) // 2)
    pairs: list[tuple[int, int]] = []
    for _ in range(n_pairs):
        i, j = int(rng.integers(0, n_samples)), int(
            rng.integers(0, n_samples)
        )
        if i != j and abs(y_train[i] - y_train[j]) >= 5:
            pairs.append((i, j))
    if not pairs:
        return {
            "mae": float("nan"),
            "r2": float("nan"),
            "n_pairs": 0,
        }
    pair_features = []
    pair_labels = []
    for i, j in pairs:
        pair_features.append(X_train[i] - X_train[j])
        pair_labels.append(y_train[i] - y_train[j])
    pair_features = np.array(pair_features, dtype=np.float32)
    pair_labels = np.array(pair_labels, dtype=np.float32)
    pairwise_model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=3,
        random_state=seed,
        max_iter=3000,
    )
    pairwise_model.fit(pair_features, pair_labels)
    anchor_count = min(int(n_anchors), X_train.shape[0])
    anchor_indices = np.linspace(
        0, X_train.shape[0] - 1, num=anchor_count, dtype=int
    )
    anchor_features = X_train[anchor_indices]
    anchor_ages = y_train[anchor_indices]

    absolute_preds = []
    for x in X_test:
        diffs = anchor_features - x
        predicted_deltas = pairwise_model.predict(diffs)
        anchor_based_ages = anchor_ages - predicted_deltas
        absolute_preds.append(float(np.median(anchor_based_ages)))
    absolute_preds = np.asarray(absolute_preds, dtype=np.float32)
    return {
        "mae": float(
            mean_absolute_error(y_test, absolute_preds)
        ),
        "r2": float(r2_score(y_test, absolute_preds)),
        "n_pairs": len(pairs),
        "n_anchors": anchor_count,
    }


def run_all_baselines(
    train_dataset: AgingDataset,
    test_dataset: AgingDataset,
    output_dir: str | Path,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    train_features = train_dataset.features.numpy()
    train_ages = train_dataset.ages.numpy()
    test_features = test_dataset.features.numpy()
    test_ages = test_dataset.ages.numpy()

    results: dict[str, dict[str, float]] = {}

    elasticnet_result = train_elasticnet_baseline(
        train_features, train_ages, test_features, test_ages, seed=seed
    )
    results["ElasticNet"] = elasticnet_result

    mlp_result = train_mlp_baseline(
        train_features, train_ages, test_features, test_ages, seed=seed
    )
    results["MLP"] = mlp_result

    maple_result = train_maple_like_baseline(
        train_features, train_ages, test_features, test_ages, seed=seed
    )
    results["MAPLE-like"] = maple_result

    summary_df = pd.DataFrame(results).T
    summary_df.to_csv(output_dir / "baseline_summary.csv")
    with (output_dir / "baseline_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results
