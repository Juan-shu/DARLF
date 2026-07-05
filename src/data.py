from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA

from .priors import (
    AGING_PATHWAYS,
    CELL_MARKERS,
    NOISY_GENE_PREFIXES,
    NOISY_GENE_SYMBOLS,
)


def clean_sample_id(text: str) -> str:
    value = str(text).strip().replace("\\", "/")
    value = value.split("/")[-1]
    return value.lstrip("./")


def clean_gene_name(text: str) -> str:
    return str(text).strip().upper()


def derive_age_group(age: float) -> str:
    if age < 20:
        return "young"
    if age < 45:
        return "adult"
    if age < 65:
        return "midlife"
    return "old"


def _normalize_string(series: pd.Series, default: str = "unknown") -> pd.Series:
    return series.replace("", np.nan).fillna(default).astype(str).str.strip()


def _standardize_tissue(text: str) -> tuple[str, str]:
    value = str(text).strip()
    value_lower = value.lower()
    if "blood" in value_lower or "pbmc" in value_lower:
        return "blood", value
    if "brain" in value_lower or "cortex" in value_lower or "hippoc" in value_lower or "cerebell" in value_lower:
        return "brain", value
    if "retina" in value_lower or "macula" in value_lower or "rpe" in value_lower:
        return "eye", value
    if "heart" in value_lower or "atrium" in value_lower or "ventricle" in value_lower or "myocard" in value_lower:
        return "heart", value
    if "skin" in value_lower or "dermis" in value_lower or "epidermis" in value_lower or "fibroblast" in value_lower:
        return "skin", value
    if "liver" in value_lower or "hepat" in value_lower:
        return "liver", value
    if "adipose" in value_lower or "fat" in value_lower:
        return "adipose", value
    if "ovary" in value_lower:
        return "ovary", value
    if "lung" in value_lower:
        return "lung", value
    if "breast" in value_lower or "mammary" in value_lower:
        return "breast", value
    if "bone" in value_lower and "marrow" in value_lower:
        return "bone marrow", value
    if "synov" in value_lower:
        return "synovial biopsy", value
    if "pancrea" in value_lower and "islet" in value_lower:
        return "pancreatic islet", value
    if "nsclc" in value_lower or "lung cancer" in value_lower:
        return "nsclc", value
    if "meningioma" in value_lower:
        return "meningioma", value
    if "muscle" in value_lower or "skeletal" in value_lower:
        return "muscle", value
    if "kidney" in value_lower or "renal" in value_lower:
        return "kidney", value
    return value_lower.split(";")[0].strip() or "other", value


def _standardize_condition(text: str) -> tuple[str, str]:
    value = str(text).strip()
    value_lower = value.lower()
    if value_lower in {"healthy", "control", "normal", "wt", "wild type", "wild-type"}:
        return "Healthy", "Healthy"
    if any(term in value_lower for term in ("cancer", "tumor", "carcinoma", "nsclc", "glioma", "meningioma", "neoplas", "malignan")):
        return "Disease", "Cancer"
    if any(term in value_lower for term in ("alzheimer", "parkinson", "dementia", "als", "huntington", "neurodeg")):
        return "Disease", "Neurodegeneration"
    if any(term in value_lower for term in ("amd", "retin", "macular", "ocular", "glaucoma", "cataract")):
        return "Disease", "Ocular degeneration"
    if any(term in value_lower for term in ("dcm", "hcm", "cardio", "heart failure", "ppcm", "myocard", "hypertens")):
        return "Disease", "Cardiometabolic"
    if any(term in value_lower for term in ("schiz", "bipolar", "depression", "mdd", "psychi", "autism", "adhd")):
        return "Disease", "Psychiatric"
    if any(term in value_lower for term in ("arthritis", "lupus", "immune", "inflam", "crohn", "colitis", "scleroderma")):
        return "Disease", "Autoimmune/Inflammatory"
    if any(term in value_lower for term in ("diabetes", "obes", "metabolic", "insulin")):
        return "Disease", "Metabolic"
    if value_lower in {"unknown", "nan", "none", "", "na"}:
        return "Unknown", "Unknown"
    return "Disease", "Other disease"


def _is_noisy_gene(gene: str) -> bool:
    gene = clean_gene_name(gene)
    if gene in NOISY_GENE_SYMBOLS:
        return True
    return any(gene.startswith(prefix) for prefix in NOISY_GENE_PREFIXES)


def _collect_available_genes(
    mapping: dict[str, list[str]], available_genes: set[str]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for key, genes in mapping.items():
        present = [gene for gene in genes if gene in available_genes]
        if len(present) >= 2:
            result[key] = present
    return result


def summarize_expression_header(file_path: Path) -> tuple[str, dict[str, str]]:
    header = pd.read_csv(file_path, sep="\t", nrows=0)
    columns = [str(col) for col in header.columns]
    gene_col = columns[0]
    sample_map = {clean_sample_id(col): col for col in columns[1:]}
    return gene_col, sample_map


def read_expression_subset(
    file_path: Path, requested_samples: set[str]
) -> pd.DataFrame:
    gene_col, sample_map = summarize_expression_header(file_path)
    selected_cols = [
        sample_map[sid] for sid in sorted(requested_samples) if sid in sample_map
    ]
    usecols = [gene_col] + selected_cols
    expr = pd.read_csv(file_path, sep="\t", usecols=usecols)
    expr = expr.rename(columns={gene_col: "gene"})
    expr["gene"] = expr["gene"].map(clean_gene_name)
    expr = expr[
        ~expr["gene"].isin({"", "N_UNMAPPED", "__NO_FEATURE", "__AMBIGUOUS"})
    ].copy()
    expr = expr.set_index("gene")
    expr.columns = [clean_sample_id(col) for col in expr.columns]
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    expr = expr.groupby(level=0).sum()
    return expr


def compute_size_factors(expr: pd.DataFrame) -> pd.Series:
    expr = expr.astype(np.float32)
    positive_mask = expr > 0
    sufficient_genes = positive_mask.sum(axis=1) >= max(
        3, int(expr.shape[1] * 0.05)
    )
    work_expr = expr.loc[sufficient_genes].copy()
    if work_expr.empty:
        library_sizes = expr.sum(axis=0).replace(0, np.nan)
        return (
            (library_sizes / float(np.nanmedian(library_sizes)))
            .fillna(1.0)
            .astype(np.float32)
        )

    log_expr = np.log(work_expr.where(work_expr > 0))
    gene_geo_means = np.exp(log_expr.mean(axis=1, skipna=True))
    valid = np.isfinite(gene_geo_means.to_numpy(dtype=float)) & (
        gene_geo_means.to_numpy(dtype=float) > 0
    )
    work_expr = work_expr.loc[valid]
    gene_geo_means = gene_geo_means.loc[work_expr.index]
    ratios = work_expr.div(gene_geo_means, axis=0).replace(
        [np.inf, -np.inf], np.nan
    )
    size_factors = ratios.median(axis=0, skipna=True)
    if (
        not np.isfinite(size_factors.to_numpy(dtype=float)).all()
    ) or float(size_factors.median()) <= 0:
        upper_quartile = expr.replace(0, np.nan).quantile(
            0.75, axis=0, interpolation="linear"
        )
        size_factors = upper_quartile / float(np.nanmedian(upper_quartile))
    else:
        size_factors = size_factors / float(np.nanmedian(size_factors))
    return size_factors.replace(0, np.nan).fillna(1.0).astype(np.float32)


def normalize_counts(
    expr: pd.DataFrame, method: str = "median_ratio_log1p"
) -> pd.DataFrame:
    method = str(method).strip().lower()
    expr = expr.astype(np.float32)
    if method == "log_cpm":
        library_sizes = expr.sum(axis=0).replace(0, np.nan)
        cpm = expr.div(library_sizes, axis=1) * 1_000_000.0
        return (
            np.log1p(cpm.replace([np.inf, -np.inf], np.nan).fillna(0.0))
            .astype(np.float32)
        )
    size_factors = compute_size_factors(expr)
    normalized = np.log1p(expr.div(size_factors, axis=1))
    return (
        normalized.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )


@dataclass
class FeaturePreprocessor:
    top_hvgs: int = 2500
    clip_zscore: float = 5.0
    exclude_noisy_genes: bool = True
    add_rank_features: bool = True
    add_pathway_scores: bool = True
    add_cell_scores: bool = True
    feature_selection_mode: str = "variance_age"
    age_corr_weight: float = 2.0
    within_tissue_age_weight: float = 2.5
    tissue_specificity_penalty: float = 1.5
    cell_marker_penalty: float = 1.0
    exclude_cell_marker_genes: bool = False
    min_gene_detection_rate: float = 0.05
    min_tissue_samples_for_corr: int = 25
    regress_out_cell_scores: bool = False
    cell_residual_ridge: float = 1e-4
    n_gene_pcs: int = 0
    selected_genes_: list[str] = field(default_factory=list)
    gene_means_: pd.Series | None = None
    gene_stds_: pd.Series | None = None
    feature_names_: list[str] = field(default_factory=list)
    pathway_gene_map_: dict[str, list[str]] = field(default_factory=dict)
    cell_gene_map_: dict[str, list[str]] = field(default_factory=dict)
    cell_score_means_: pd.Series | None = None
    cell_score_stds_: pd.Series | None = None
    selected_gene_cell_regression_coef_: pd.DataFrame | None = None
    gene_pca_: PCA | None = None

    def _compute_cell_score_frame(
        self, expr: pd.DataFrame
    ) -> pd.DataFrame:
        if not self.cell_gene_map_:
            return pd.DataFrame(index=expr.columns)
        score_dict = {}
        for name, genes in self.cell_gene_map_.items():
            score_dict[name] = (
                expr.reindex(genes)
                .mean(axis=0)
                .to_numpy(dtype=np.float32)
            )
        return pd.DataFrame(score_dict, index=expr.columns)

    def _residualize_by_cell_scores(
        self, expr: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cell_score_frame = self._compute_cell_score_frame(expr)
        if cell_score_frame.empty:
            return expr, cell_score_frame
        if self.cell_score_means_ is None or self.cell_score_stds_ is None:
            self.cell_score_means_ = cell_score_frame.mean(axis=0)
            self.cell_score_stds_ = cell_score_frame.std(axis=0).replace(
                0, 1.0
            )
        standardized_scores = (
            cell_score_frame - self.cell_score_means_
        ) / self.cell_score_stds_
        design = np.column_stack(
            [
                np.ones(len(standardized_scores), dtype=np.float32),
                standardized_scores.to_numpy(
                    dtype=np.float32, copy=True
                ),
            ]
        )
        penalty = np.eye(design.shape[1], dtype=np.float32) * float(
            self.cell_residual_ridge
        )
        penalty[0, 0] = 0.0
        expr_matrix = expr.T.to_numpy(dtype=np.float32, copy=True)
        gram = design.T @ design + penalty
        coef = np.linalg.solve(gram, design.T @ expr_matrix)
        residual = expr_matrix - design @ coef
        coef_df = pd.DataFrame(
            coef,
            index=["intercept", *standardized_scores.columns.tolist()],
            columns=expr.index,
        )
        return (
            pd.DataFrame(
                residual.T,
                index=expr.index,
                columns=expr.columns,
            ),
            coef_df,
        )

    def fit(
        self,
        expr: pd.DataFrame,
        ages: pd.Series,
        tissue_labels: pd.Series | None = None,
    ) -> "FeaturePreprocessor":
        expr = expr.copy()
        expr.index = expr.index.astype(str)
        if self.exclude_noisy_genes:
            expr = expr.loc[
                [not _is_noisy_gene(gene) for gene in expr.index]
            ]
        detection_rate = (expr > 0).mean(axis=1)
        expr = expr.loc[
            detection_rate >= max(self.min_gene_detection_rate, 0.0)
        ]
        cell_marker_genes = {
            gene
            for genes in CELL_MARKERS.values()
            for gene in genes
        }
        if self.exclude_cell_marker_genes:
            expr = expr.loc[
                [gene not in cell_marker_genes for gene in expr.index]
            ]
        available_genes = set(expr.index.tolist())
        self.pathway_gene_map_ = _collect_available_genes(
            AGING_PATHWAYS, available_genes
        )
        self.cell_gene_map_ = _collect_available_genes(
            CELL_MARKERS, available_genes
        )
        cell_coef_df: pd.DataFrame | None = None
        if self.regress_out_cell_scores and self.cell_gene_map_:
            expr, cell_coef_df = self._residualize_by_cell_scores(
                expr
            )

        variances = expr.var(axis=1)
        score = np.log1p(variances.clip(lower=1e-6))
        if len(expr.columns) >= 5:
            corrs = []
            age_array = (
                pd.Series(ages, index=expr.columns)
                .reindex(expr.columns)
                .to_numpy(dtype=float)
            )
            for gene in expr.index:
                gene_array = expr.loc[gene].to_numpy(dtype=float)
                if np.std(gene_array) < 1e-8 or np.std(age_array) < 1e-8:
                    corrs.append(0.0)
                else:
                    corrs.append(
                        abs(np.corrcoef(gene_array, age_array)[0, 1])
                    )
            score = score + pd.Series(corrs, index=expr.index) * float(
                self.age_corr_weight
            )
        tissue_labels = (
            pd.Series(tissue_labels, index=expr.columns)
            .reindex(expr.columns)
            .fillna("unknown")
            .astype(str)
            if tissue_labels is not None
            else None
        )
        if tissue_labels is not None and tissue_labels.nunique() >= 2:
            if self.tissue_specificity_penalty > 0:
                tissue_means = []
                for tissue_name, sample_index in tissue_labels.groupby(
                    tissue_labels
                ).groups.items():
                    if len(sample_index) < 2:
                        continue
                    tissue_means.append(
                        expr.loc[:, list(sample_index)]
                        .mean(axis=1)
                        .rename(str(tissue_name))
                    )
                if tissue_means:
                    tissue_mean_df = pd.concat(
                        tissue_means, axis=1
                    ).fillna(0.0)
                    tissue_specificity = tissue_mean_df.var(
                        axis=1
                    ) / variances.clip(lower=1e-6)
                    score = score - np.log1p(
                        tissue_specificity.clip(lower=0.0)
                    ) * float(self.tissue_specificity_penalty)
            if self.within_tissue_age_weight > 0:
                within_corrs: list[np.ndarray] = []
                ages_series = (
                    pd.Series(ages, index=expr.columns)
                    .reindex(expr.columns)
                    .astype(float)
                )
                expr_values = expr.to_numpy(dtype=np.float32)
                for _, sample_index in tissue_labels.groupby(
                    tissue_labels
                ).groups.items():
                    if len(sample_index) < int(
                        self.min_tissue_samples_for_corr
                    ):
                        continue
                    column_idx = expr.columns.get_indexer(
                        list(sample_index)
                    )
                    age_sub = ages_series.iloc[column_idx].to_numpy(
                        dtype=np.float32
                    )
                    age_centered = age_sub - age_sub.mean()
                    age_denom = float(
                        np.sqrt(
                            np.sum(age_centered * age_centered)
                        )
                    )
                    if age_denom < 1e-8:
                        continue
                    block = expr_values[:, column_idx]
                    block_centered = block - block.mean(
                        axis=1, keepdims=True
                    )
                    denom = np.sqrt(
                        np.sum(block_centered * block_centered, axis=1)
                    ) * age_denom
                    corr = np.divide(
                        block_centered @ age_centered,
                        denom,
                        out=np.zeros(
                            expr_values.shape[0], dtype=np.float32
                        ),
                        where=denom > 1e-8,
                    )
                    within_corrs.append(np.nan_to_num(corr))
                if within_corrs:
                    within_corr_matrix = np.column_stack(
                        within_corrs
                    )
                    mean_abs_within = np.mean(
                        np.abs(within_corr_matrix), axis=1
                    )
                    sign_consistency = np.abs(
                        np.mean(
                            np.sign(within_corr_matrix), axis=1
                        )
                    )
                    score = score + pd.Series(
                        mean_abs_within
                        + 0.35 * sign_consistency,
                        index=expr.index,
                    ) * float(self.within_tissue_age_weight)
        if self.cell_marker_penalty > 0 and not self.exclude_cell_marker_genes:
            marker_mask = pd.Series(
                [gene in cell_marker_genes for gene in expr.index],
                index=expr.index,
            )
            score = score - marker_mask.astype(float) * float(
                self.cell_marker_penalty
            )
        selected = (
            score.sort_values(ascending=False)
            .head(min(self.top_hvgs, len(score)))
            .index.tolist()
        )
        self.selected_genes_ = selected
        selected_expr = expr.loc[selected]
        self.gene_means_ = selected_expr.mean(axis=1)
        self.gene_stds_ = selected_expr.std(axis=1).replace(0, 1.0)
        if cell_coef_df is not None:
            self.selected_gene_cell_regression_coef_ = (
                cell_coef_df.loc[:, self.selected_genes_].copy()
            )
        z_selected = (
            (selected_expr.T - self.gene_means_) / self.gene_stds_
        ).clip(lower=-self.clip_zscore, upper=self.clip_zscore)
        feature_names: list[str] = []
        if self.n_gene_pcs > 0 and len(self.selected_genes_) >= 2:
            n_components = min(
                int(self.n_gene_pcs),
                z_selected.shape[0],
                z_selected.shape[1],
            )
            if n_components >= 2:
                self.gene_pca_ = PCA(
                    n_components=n_components,
                    svd_solver="auto",
                    random_state=0,
                )
                self.gene_pca_.fit(
                    z_selected.to_numpy(
                        dtype=np.float32, copy=True
                    )
                )
                feature_names.extend(
                    [f"pc::{i+1}" for i in range(n_components)]
                )
            else:
                self.gene_pca_ = None
                feature_names.extend(
                    [f"z::{gene}" for gene in self.selected_genes_]
                )
        else:
            self.gene_pca_ = None
            feature_names.extend(
                [f"z::{gene}" for gene in self.selected_genes_]
            )
        if self.add_rank_features:
            feature_names.extend(
                [f"rank::{gene}" for gene in self.selected_genes_]
            )
        if self.add_pathway_scores:
            feature_names.extend(
                [f"pathway::{name}" for name in self.pathway_gene_map_]
            )
        if self.add_cell_scores:
            feature_names.extend(
                [f"cell::{name}" for name in self.cell_gene_map_]
            )
        self.feature_names_ = feature_names
        return self

    def transform(self, expr: pd.DataFrame) -> np.ndarray:
        if self.gene_means_ is None or self.gene_stds_ is None:
            raise RuntimeError(
                "FeaturePreprocessor must be fit before transform."
            )
        full_x = expr.T.apply(pd.to_numeric, errors="coerce")
        x = expr.reindex(self.selected_genes_).T
        x = (
            x.apply(pd.to_numeric, errors="coerce")
            .fillna(self.gene_means_.to_dict())
        )
        if (
            self.regress_out_cell_scores
            and self.selected_gene_cell_regression_coef_ is not None
            and self.cell_score_means_ is not None
            and self.cell_score_stds_ is not None
        ):
            cell_score_frame = self._compute_cell_score_frame(expr)
            standardized_scores = (
                cell_score_frame.reindex(
                    columns=self.cell_score_means_.index
                ).fillna(0.0)
                - self.cell_score_means_
            ) / self.cell_score_stds_
            design = np.column_stack(
                [
                    np.ones(len(standardized_scores), dtype=np.float32),
                    standardized_scores.to_numpy(
                        dtype=np.float32, copy=True
                    ),
                ]
            )
            coef = self.selected_gene_cell_regression_coef_.to_numpy(
                dtype=np.float32, copy=True
            ).T
            fitted = design @ coef
            x = pd.DataFrame(
                x.to_numpy(dtype=np.float32, copy=True) - fitted,
                index=x.index,
                columns=x.columns,
            )
        z = (x - self.gene_means_) / self.gene_stds_
        z = z.clip(lower=-self.clip_zscore, upper=self.clip_zscore)
        if self.gene_pca_ is not None:
            z_block = self.gene_pca_.transform(
                z.to_numpy(dtype=np.float32, copy=True)
            ).astype(np.float32, copy=False)
        else:
            z_block = z.to_numpy(dtype=np.float32, copy=True)
        blocks: list[np.ndarray] = [z_block]

        rank_x = None
        if self.add_rank_features:
            rank_x = x.rank(axis=1, method="average", pct=True)
            rank_x = (rank_x - 0.5) * 2.0
            blocks.append(rank_x.to_numpy(dtype=np.float32, copy=True))

        if self.add_pathway_scores and self.pathway_gene_map_:
            pathways = [
                full_x.reindex(columns=genes)
                .mean(axis=1)
                .to_numpy(dtype=np.float32)
                for genes in self.pathway_gene_map_.values()
            ]
            blocks.append(np.column_stack(pathways))
        if self.add_cell_scores and self.cell_gene_map_:
            cells = [
                full_x.reindex(columns=genes)
                .mean(axis=1)
                .to_numpy(dtype=np.float32)
                for genes in self.cell_gene_map_.values()
            ]
            blocks.append(np.column_stack(cells))
        combined = np.concatenate(blocks, axis=1).astype(
            np.float32, copy=False
        )
        return np.nan_to_num(
            combined,
            nan=0.0,
            posinf=self.clip_zscore,
            neginf=-self.clip_zscore,
        )

    def fit_transform(
        self,
        expr: pd.DataFrame,
        ages: pd.Series,
        tissue_labels: pd.Series | None = None,
    ) -> np.ndarray:
        return self.fit(
            expr,
            ages,
            tissue_labels=tissue_labels,
        ).transform(expr)

    def state_dict(self) -> dict:
        return {
            "top_hvgs": self.top_hvgs,
            "clip_zscore": self.clip_zscore,
            "exclude_noisy_genes": self.exclude_noisy_genes,
            "add_rank_features": self.add_rank_features,
            "add_pathway_scores": self.add_pathway_scores,
            "add_cell_scores": self.add_cell_scores,
            "feature_selection_mode": self.feature_selection_mode,
            "age_corr_weight": self.age_corr_weight,
            "within_tissue_age_weight": self.within_tissue_age_weight,
            "tissue_specificity_penalty": self.tissue_specificity_penalty,
            "cell_marker_penalty": self.cell_marker_penalty,
            "exclude_cell_marker_genes": self.exclude_cell_marker_genes,
            "min_gene_detection_rate": self.min_gene_detection_rate,
            "min_tissue_samples_for_corr": self.min_tissue_samples_for_corr,
            "regress_out_cell_scores": self.regress_out_cell_scores,
            "cell_residual_ridge": self.cell_residual_ridge,
            "n_gene_pcs": self.n_gene_pcs,
            "selected_genes_": list(self.selected_genes_),
            "gene_means_": self.gene_means_.to_dict()
            if self.gene_means_ is not None
            else None,
            "gene_stds_": self.gene_stds_.to_dict()
            if self.gene_stds_ is not None
            else None,
            "feature_names_": list(self.feature_names_),
            "pathway_gene_map_": self.pathway_gene_map_,
            "cell_gene_map_": self.cell_gene_map_,
            "cell_score_means_": self.cell_score_means_.to_dict()
            if self.cell_score_means_ is not None
            else None,
            "cell_score_stds_": self.cell_score_stds_.to_dict()
            if self.cell_score_stds_ is not None
            else None,
            "gene_pca_n_components_": int(self.gene_pca_.n_components_)
            if self.gene_pca_ is not None
            else 0,
            "gene_pca_explained_variance_ratio_": self.gene_pca_.explained_variance_ratio_.tolist()
            if self.gene_pca_ is not None
            else [],
        }


@dataclass
class DiscoveryDataBundle:
    metadata: pd.DataFrame
    expression: pd.DataFrame
    features: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    preprocessor: FeaturePreprocessor
    pathway_names: list[str]
    cell_score_names: list[str]


def load_metadata(data_root: Path) -> pd.DataFrame:
    allmeta = pd.read_csv(
        data_root / "Allcounts" / "Allmeta.txt", sep="\t"
    )
    rawmeta = pd.read_csv(
        data_root / "RawTables" / "meta_filtered.txt", sep="\t"
    )

    allmeta = allmeta.rename(
        columns={
            "GEO #": "geo_accession",
            "SRR ID": "sample_id",
            "Age": "age",
            "Gender": "sex",
            "Condition": "condition",
            "Tissue": "tissue",
            "Batch": "batch",
        }
    )
    rawmeta = rawmeta.rename(
        columns={
            "GEO..": "geo_accession",
            "SRR.ID": "sample_id",
            "Age": "age",
            "Gender": "sex",
            "Condition": "condition",
            "Tissue": "tissue",
            "Batch": "batch",
            "Healthy": "healthy_raw",
        }
    )

    allmeta["sample_id"] = allmeta["sample_id"].map(clean_sample_id)
    rawmeta["sample_id"] = rawmeta["sample_id"].map(clean_sample_id)
    allmeta["healthy"] = (
        allmeta["condition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("healthy")
    )
    rawmeta["healthy"] = (
        rawmeta["healthy_raw"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    keep_cols = [
        "sample_id",
        "geo_accession",
        "age",
        "sex",
        "condition",
        "tissue",
        "batch",
        "healthy",
    ]
    allmeta = allmeta[keep_cols].copy()
    rawmeta = rawmeta[keep_cols].copy()

    merged = allmeta.set_index("sample_id")
    merged.update(rawmeta.set_index("sample_id"))
    meta = merged.reset_index()
    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    meta = meta.dropna(subset=["age", "tissue"]).copy()
    meta["sex"] = _normalize_string(meta["sex"], default="unknown").str.lower()
    meta["condition"] = _normalize_string(
        meta["condition"], default="unknown"
    )
    meta["batch"] = _normalize_string(meta["batch"], default="unknown")
    meta["geo_accession"] = _normalize_string(
        meta["geo_accession"], default="unknown"
    )
    meta["dataset"] = meta["batch"]
    meta["sample"] = meta["sample_id"]
    meta["age_group"] = meta["age"].map(derive_age_group)
    meta[["tissue_family", "tissue_fine"]] = meta["tissue"].apply(
        lambda x: pd.Series(_standardize_tissue(x))
    )
    meta[["health_status", "disease_group"]] = meta["condition"].apply(
        lambda x: pd.Series(_standardize_condition(x))
    )
    meta["is_disease"] = meta["health_status"].eq("Disease").astype(int)
    meta = meta.drop_duplicates(subset=["sample_id"], keep="first")
    return meta


def load_expression(
    data_root: Path, sample_ids: set[str]
) -> pd.DataFrame:
    all_expr = read_expression_subset(
        data_root / "Allcounts" / "Allcounts.txt", sample_ids
    )
    missing = sample_ids - set(all_expr.columns)
    if not missing:
        return all_expr
    raw_expr = read_expression_subset(
        data_root / "RawTables" / "raw_filtered.txt", missing
    )
    return all_expr.combine_first(raw_expr).astype(np.float32)


def split_by_dataset(
    metadata: pd.DataFrame, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    groups = metadata["dataset"].fillna("unknown").astype(str)
    if groups.nunique() < 3:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(metadata))
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_fraction)))
        test_idx = indices[:n_test]
        mask_test = np.zeros(len(metadata), dtype=bool)
        mask_test[test_idx] = True
        return ~mask_test, mask_test

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=seed
    )
    train_idx, test_idx = next(
        splitter.split(metadata, groups=groups)
    )
    train_mask = np.zeros(len(metadata), dtype=bool)
    test_mask = np.zeros(len(metadata), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask


def split_random(
    metadata: pd.DataFrame, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(metadata))
    rng.shuffle(indices)
    n_test = max(1, int(round(len(indices) * test_fraction)))
    test_idx = indices[:n_test]
    mask_test = np.zeros(len(metadata), dtype=bool)
    mask_test[test_idx] = True
    return ~mask_test, mask_test


def load_discovery_bundle(
    data_root: str | Path,
    min_samples_per_tissue: int = 25,
    normalization: str = "median_ratio_log1p",
    test_fraction: float = 0.2,
    seed: int = 42,
    split_mode: str = "group_by_dataset",
    top_hvgs: int = 2500,
    add_rank_features: bool = True,
    add_pathway_scores: bool = True,
    add_cell_scores: bool = True,
    age_corr_weight: float = 2.0,
    within_tissue_age_weight: float = 2.5,
    tissue_specificity_penalty: float = 1.5,
    cell_marker_penalty: float = 1.0,
    exclude_cell_marker_genes: bool = False,
    min_gene_detection_rate: float = 0.05,
    min_tissue_samples_for_corr: int = 25,
    regress_out_cell_scores: bool = False,
    cell_residual_ridge: float = 1e-4,
    n_gene_pcs: int = 0,
    include_tissues: list[str] | tuple[str, ...] | None = None,
) -> DiscoveryDataBundle:
    data_root = Path(data_root)
    metadata = load_metadata(data_root)
    requested_tissues = [
        str(value).strip()
        for value in (include_tissues or [])
        if str(value).strip()
    ]
    if requested_tissues:
        metadata = metadata[
            metadata["tissue_family"].isin(requested_tissues)
        ].copy()
    tissue_counts = metadata["tissue_family"].value_counts()
    keep_tissues = tissue_counts[
        tissue_counts >= int(min_samples_per_tissue)
    ].index
    metadata = metadata[
        metadata["tissue_family"].isin(keep_tissues)
    ].copy()
    sample_ids = set(metadata["sample_id"])
    expression = load_expression(data_root, sample_ids)
    common = metadata["sample_id"].isin(expression.columns)
    metadata = metadata.loc[common].copy().reset_index(drop=True)
    expression = expression.loc[
        :, metadata["sample_id"].tolist()
    ].copy()
    expression = normalize_counts(expression, method=normalization)

    preprocessor = FeaturePreprocessor(
        top_hvgs=int(top_hvgs),
        add_rank_features=bool(add_rank_features),
        add_pathway_scores=bool(add_pathway_scores),
        add_cell_scores=bool(add_cell_scores),
        age_corr_weight=float(age_corr_weight),
        within_tissue_age_weight=float(within_tissue_age_weight),
        tissue_specificity_penalty=float(
            tissue_specificity_penalty
        ),
        cell_marker_penalty=float(cell_marker_penalty),
        exclude_cell_marker_genes=bool(
            exclude_cell_marker_genes
        ),
        min_gene_detection_rate=float(
            min_gene_detection_rate
        ),
        min_tissue_samples_for_corr=int(
            min_tissue_samples_for_corr
        ),
        regress_out_cell_scores=bool(
            regress_out_cell_scores
        ),
        cell_residual_ridge=float(cell_residual_ridge),
        n_gene_pcs=int(n_gene_pcs),
    )
    features = preprocessor.fit_transform(
        expression,
        metadata.set_index("sample_id")["age"],
        tissue_labels=metadata.set_index("sample_id")[
            "tissue_family"
        ],
    )
    split_mode = str(split_mode).strip().lower()
    if split_mode == "random":
        train_mask, test_mask = split_random(
            metadata,
            test_fraction=test_fraction,
            seed=seed,
        )
    else:
        train_mask, test_mask = split_by_dataset(
            metadata,
            test_fraction=test_fraction,
            seed=seed,
        )

    return DiscoveryDataBundle(
        metadata=metadata,
        expression=expression,
        features=features,
        train_mask=train_mask,
        test_mask=test_mask,
        preprocessor=preprocessor,
        pathway_names=list(preprocessor.pathway_gene_map_.keys()),
        cell_score_names=list(preprocessor.cell_gene_map_.keys()),
    )
