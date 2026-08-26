#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Observed embedding–PPI concordance analysis for main and repeated TCM-ES models.

The script retains the observed herb–herb, disease–disease, disease–herb, and
protein–protein analyses, repeat consensus, local-neighbour analysis, and
leave-one-repeat-out stability analysis. Permutation-based comparisons are not
performed in this version.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr, rankdata, t

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HERB_LIST_FILE = PROJECT_ROOT / "core/standard_TCM_entities/herb_list.pkl"
PROTEIN_LIST_FILE = PROJECT_ROOT / "data/herb_targets/protein_list.pkl"
HERB_TARGETS_FILE = PROJECT_ROOT / "data/herb_targets/herb_targets.pkl"

DISEASE_TABLE_FILE = PROJECT_ROOT / "data/disease/Disease_class_gene(science2015).xlsx"
DISEASE_GENES_FILE = PROJECT_ROOT / "data/disease/disease_genes_science2015.pkl"
DISEASE_MMSYM_FILE = PROJECT_ROOT / "data/disease/disease_mmsym.pkl"

PPI_GENES_FILE = PROJECT_ROOT / "data/PPI/ppi_genes.pkl"
PPI_DISTANCE_FILE = PROJECT_ROOT / "data/PPI/ppi_genes_distance.npy"

MAIN_HERB_EMBEDDING_FILE = PROJECT_ROOT / "results/embeddings/TCM_embeddings/individual_herb_embeddings.pkl"
MAIN_PROTEIN_EMBEDDING_FILE = PROJECT_ROOT / "results/protein_alignment_embeddings/main/target_embeddings.pkl"
MAIN_DISEASE_EMBEDDING_FILE = PROJECT_ROOT / "results/embeddings/disease_embeddings/main/disease_embedding.pkl"

REPEAT_HERB_ROOT = PROJECT_ROOT / "results/embeddings/TCM_embeddings_repeated"
REPEAT_PROTEIN_ROOT = PROJECT_ROOT / "results/protein_alignment_embeddings"
REPEAT_DISEASE_ROOT = PROJECT_ROOT / "results/embeddings/disease_embeddings"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/PPI_concordance_analysis_no_permutation"
DEFAULT_RANDOM_SEED = 2026

MODULE_ANALYSES = ("herb_herb", "disease_disease", "disease_herb")
ALL_ANALYSES = ("herb_herb", "disease_disease", "disease_herb", "protein_protein")

ANALYSIS_LABELS = {
    "herb_herb": "Herb–herb",
    "disease_disease": "Disease–disease",
    "disease_herb": "Disease–herb",
    "protein_protein": "Protein–protein",
}

MODULE_Y_LABEL = "Standardized PPI separation"
MODULE_X_LABEL = "Within-model distance percentile"
PROTEIN_Y_LABEL = "Embedding distance percentile"
PROTEIN_X_LABEL = "PPI shortest-path group"
PROTEIN_Y_MIN = 0.30
PROTEIN_Y_MAX = 0.65

GLOBAL_DISPLAY_BINS = 20
LOCAL_K_VALUES = (5, 10, 20, 50)
LOCAL_BOOTSTRAP_ITERATIONS = 10000
LOCAL_PLOT_ORDER = ("disease_herb", "disease_disease", "herb_herb")
LEADING_BINS_TO_MERGE = {"disease_disease": 2}

PROTEIN_GROUP_LABELS = ["1", "2", "3", "4", ">4"]
PROTEIN_GROUP_COLORS = ["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#B279A2"]
OBSERVED_COLORS = {
    "herb_herb": "#ff8a65",
    "disease_disease": "#d8a12a",
    "disease_herb": "#54b992",
    "protein_protein": "#4C78A8",
}
MAIN_REFERENCE_COLOR = "#3B73C5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze observed main, repeat-consensus, local-neighbour, and "
            "leave-one-repeat-out embedding-PPI concordance."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for bootstrap resampling.")
    parser.add_argument(
        "--consensus-method",
        choices=("mean", "median"),
        default="mean",
        help="How to aggregate the same pair's percentile ranks across repeat models.",
    )
    parser.add_argument("--min-module-size", type=int, default=1, help="Minimum mapped target/gene count for herb/disease modules.")
    parser.add_argument("--min-disease-symptoms", type=int, default=1, help="Minimum modern-medicine symptoms for disease rows.")
    parser.add_argument(
        "--repeat-models",
        nargs="*",
        default=None,
        help="Optional repeat model names to include. By default all complete repeat_* models are used.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--figure-dpi", type=int, default=450, help="Figure DPI.")
    parser.add_argument("--panel-width", type=float, default=2.3, help="Width of each grid panel.")
    parser.add_argument("--panel-height", type=float, default=3.0, help="Height of each grid panel.")
    parser.add_argument("--skip-figure", action="store_true", help="Write data tables without drawing figures.")
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)

def save_json(data: Dict[str, Any], path: Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): convert(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    path.write_text(
        json.dumps(convert(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))

def canonical_gene(value: Any) -> str:
    return str(value).strip()

def normalize_entity_terms(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [item for item in value if pd.notna(item)]
    if pd.isna(value):
        return []
    return [value]

def model_sort_key(model_name: str) -> Tuple[int, int]:
    if model_name == "main":
        return (-3, 0)
    if model_name in {
        "repeat_consensus",
        "repeat_mean_rank_consensus",
        "repeat_median_rank_consensus",
    }:
        return (-2, 0)
    if model_name.startswith("loo_without_"):
        tail = model_name.rsplit("_", 1)[-1]
        try:
            return (1, int(tail))
        except ValueError:
            return (1, 999)
    if model_name.startswith("repeat_"):
        try:
            return (0, int(model_name.split("_", 1)[1]))
        except Exception:
            return (0, 999)
    return (2, 999)

def discover_embedding_files(
    requested_models: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Path]]:
    model_files: Dict[str, Dict[str, Path]] = {
        "main": {
            "herb": MAIN_HERB_EMBEDDING_FILE,
            "protein": MAIN_PROTEIN_EMBEDDING_FILE,
            "disease": MAIN_DISEASE_EMBEDDING_FILE,
        }
    }

    herb_dirs = {
        path.name: path
        for path in REPEAT_HERB_ROOT.glob("repeat_*")
        if path.is_dir()
    }
    disease_dirs = {
        path.name: path
        for path in REPEAT_DISEASE_ROOT.glob("repeat_*")
        if path.is_dir()
    }
    protein_dirs = {
        path.name: path
        for path in REPEAT_PROTEIN_ROOT.glob("repeat_*")
        if path.is_dir()
    }

    complete = sorted(
        set(herb_dirs) & set(disease_dirs) & set(protein_dirs),
        key=model_sort_key,
    )
    if requested_models is not None:
        requested = set(requested_models)
        complete = [name for name in complete if name in requested]

    for model_name in complete:
        model_files[model_name] = {
            "herb": herb_dirs[model_name] / "individual_herb_embeddings.pkl",
            "protein": protein_dirs[model_name] / "target_embeddings.pkl",
            "disease": disease_dirs[model_name] / "disease_embedding.pkl",
        }

    require_files([path for files in model_files.values() for path in files.values()])
    if len(model_files) <= 1:
        raise FileNotFoundError(
            "No complete repeat embeddings were discovered across herb, disease, "
            "and protein repeat roots."
        )
    return model_files

def load_embedding_array(path: Path, expected_rows: int, label: str) -> np.ndarray:
    array = np.asarray(load_pickle(path), dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{label} must be 2D, got shape {array.shape}.")
    if array.shape[0] != expected_rows:
        raise ValueError(
            f"{label} row count {array.shape[0]} does not match expected "
            f"{expected_rows}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values.")
    return array

def build_membership_matrix(
    gene_lists: Sequence[Sequence[Any]],
    gene_to_basis: Dict[str, int],
) -> sparse.csr_matrix:
    rows: List[int] = []
    cols: List[int] = []
    for entity_id, genes in enumerate(gene_lists):
        mapped = {
            gene_to_basis[canonical_gene(gene)]
            for gene in genes
            if canonical_gene(gene) in gene_to_basis
        }
        for gene_id in mapped:
            rows.append(entity_id)
            cols.append(gene_id)
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(gene_lists), len(gene_to_basis)),
        dtype=np.float64,
    )

def replace_disconnected(distance_matrix: np.ndarray, disconnected_distance: float) -> np.ndarray:
    clean = np.asarray(distance_matrix, dtype=np.float64).copy()
    clean[clean < 0] = disconnected_distance
    np.fill_diagonal(clean, 0.0)
    return clean

def _intersection_self_distance_correction(
    dense_a: np.ndarray,
    dense_b: np.ndarray,
    distance_matrix: np.ndarray,
    cross_sum: np.ndarray,
    symmetric: bool,
) -> None:
    overlap_count = dense_a @ dense_b.T
    if symmetric:
        pairs = np.argwhere(np.triu(overlap_count >= 2, k=1))
        for row_id, col_id in pairs:
            overlap = np.flatnonzero((dense_a[row_id] > 0) & (dense_b[col_id] > 0))
            correction = float(distance_matrix[np.ix_(overlap, overlap)].sum())
            cross_sum[row_id, col_id] += correction
            cross_sum[col_id, row_id] += correction
    else:
        pairs = np.argwhere(overlap_count >= 2)
        for row_id, col_id in pairs:
            overlap = np.flatnonzero((dense_a[row_id] > 0) & (dense_b[col_id] > 0))
            correction = float(distance_matrix[np.ix_(overlap, overlap)].sum())
            cross_sum[row_id, col_id] += correction

def symmetric_module_metrics(
    membership: sparse.csr_matrix,
    distance_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    sizes = np.asarray(membership.sum(axis=1)).ravel().astype(np.float64)
    if np.any(sizes <= 0):
        raise ValueError("Symmetric module metric received an empty module.")

    dense = membership.toarray().astype(np.float64, copy=False)
    projected = np.asarray(membership @ distance_matrix, dtype=np.float64)
    total_cross = projected @ dense.T
    weighted_overlap = dense * projected
    cross_sum = (
        total_cross
        - dense @ weighted_overlap.T
        - weighted_overlap @ dense.T
    )
    _intersection_self_distance_correction(
        dense,
        dense,
        distance_matrix,
        cross_sum,
        symmetric=True,
    )
    cross_mean = cross_sum / np.outer(sizes, sizes)
    within_mean = np.diag(total_cross) / (sizes * sizes)
    separation = cross_mean - 0.5 * (within_mean[:, None] + within_mean[None, :])
    return separation, cross_mean

def cross_module_metrics(
    membership_a: sparse.csr_matrix,
    membership_b: sparse.csr_matrix,
    distance_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    sizes_a = np.asarray(membership_a.sum(axis=1)).ravel().astype(np.float64)
    sizes_b = np.asarray(membership_b.sum(axis=1)).ravel().astype(np.float64)
    if np.any(sizes_a <= 0) or np.any(sizes_b <= 0):
        raise ValueError("Cross module metric received an empty module.")

    dense_a = membership_a.toarray().astype(np.float64, copy=False)
    dense_b = membership_b.toarray().astype(np.float64, copy=False)

    projected_a = np.asarray(membership_a @ distance_matrix, dtype=np.float64)
    projected_b = np.asarray(membership_b @ distance_matrix, dtype=np.float64)

    total_cross = projected_a @ dense_b.T
    weighted_a = dense_a * projected_a
    weighted_b = dense_b * projected_b
    cross_sum = (
        total_cross
        - dense_a @ weighted_b.T
        - weighted_a @ dense_b.T
    )
    _intersection_self_distance_correction(
        dense_a,
        dense_b,
        distance_matrix,
        cross_sum,
        symmetric=False,
    )

    cross_mean = cross_sum / np.outer(sizes_a, sizes_b)
    within_a = np.diag(projected_a @ dense_a.T) / (sizes_a * sizes_a)
    within_b = np.diag(projected_b @ dense_b.T) / (sizes_b * sizes_b)
    separation = cross_mean - 0.5 * (within_a[:, None] + within_b[None, :])
    return separation, cross_mean

def upper_triangle_indices(n_entities: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n_entities, k=1)

def correlation_test(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "pearson"
) -> Tuple[float, float, int]:

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]
    n = len(x)

    if n < 3:
        return np.nan, np.nan, n
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan, n
    if method == "spearman":
        result = spearmanr(x, y)
    else:
        result = pearsonr(x, y)

    return float(result.statistic), float(result.pvalue), n

def correlation_value(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "pearson",
) -> float:
    """Return the correlation coefficient for finite paired values."""
    statistic, _, _ = correlation_test(x, y, method=method)
    return statistic

def row_standardize_symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    standardized = np.full_like(matrix, np.nan, dtype=np.float64)
    for row_id in range(n):
        mask = np.arange(n) != row_id
        values = matrix[row_id, mask]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        if std <= 0 or not np.isfinite(std):
            continue
        standardized[row_id, mask] = (values - mean) / std
    return standardized

def row_standardize_cross(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True, ddof=0)
    if np.any(std <= 0):
        raise ValueError("Cross matrix contains a zero-variance row.")
    return (matrix - mean) / std

def percentile_rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return output
    if n_valid == 1:
        output[valid] = 0.0
        return output
    ranks = rankdata(values[valid], method="average")
    output[valid] = (ranks - 1.0) / (n_valid - 1.0)
    return output

def row_percentile_distance_matrix(
    distance_matrix: np.ndarray,
    symmetric: bool,
) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    percentile = np.full_like(distance_matrix, np.nan, dtype=np.float64)

    for row_id in range(distance_matrix.shape[0]):
        values = distance_matrix[row_id].copy()
        if symmetric:
            values[row_id] = np.nan
        percentile[row_id] = percentile_rank_values(values)

    if symmetric:
        percentile = 0.5 * (percentile + percentile.T)
        np.fill_diagonal(percentile, np.nan)
    return percentile

def pair_percentile_vector(values: np.ndarray) -> np.ndarray:
    return percentile_rank_values(np.asarray(values, dtype=np.float64))

def cross_modal_centered_distance(
    disease_embedding: np.ndarray,
    herb_embedding: np.ndarray,
) -> np.ndarray:
    disease_centered = disease_embedding - disease_embedding.mean(axis=0, keepdims=True)
    herb_centered = herb_embedding - herb_embedding.mean(axis=0, keepdims=True)
    return cdist(disease_centered, herb_centered, metric="euclidean")

def module_pair_vectors(
    distance_matrix: np.ndarray,
    genetic_matrix: np.ndarray,
    symmetric: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if symmetric:
        row_id, col_id = upper_triangle_indices(distance_matrix.shape[0])
        x_values = distance_matrix[row_id, col_id]
        y_values = 0.5 * (genetic_matrix[row_id, col_id] + genetic_matrix[col_id, row_id])
    else:
        x_values = distance_matrix.ravel()
        y_values = genetic_matrix.ravel()
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[valid], y_values[valid]

def equal_frequency_bin_plan(x_values: np.ndarray, n_bins: int) -> List[np.ndarray]:
    x_values = np.asarray(x_values, dtype=np.float64)
    valid_order = np.argsort(x_values, kind="mergesort")
    return [
        chunk
        for chunk in np.array_split(valid_order, n_bins)
        if len(chunk) > 0
    ]

def display_bin_plan(
    x_values: np.ndarray,
    n_bins: int,
    analysis: str,
) -> List[np.ndarray]:
    """Create the fixed display bin plan for one analysis.

    Disease-disease starts from 20 equal-frequency bins, then combines the
    first two bins into one closest-decile bin. Other analyses retain all 20
    bins. This operation is applied before calculating plotted means, SEMs,
    fitted lines, and r_bin.
    """
    plan = equal_frequency_bin_plan(x_values, n_bins)
    n_merge = int(LEADING_BINS_TO_MERGE.get(analysis, 1))
    if n_merge <= 1:
        return plan
    if len(plan) < n_merge:
        raise ValueError(
            f"Cannot merge {n_merge} leading bins for {analysis}; "
            f"only {len(plan)} bins are available."
        )
    merged_first = np.concatenate(plan[:n_merge])
    return [merged_first] + plan[n_merge:]

def binned_summary_from_plan(
    x_values: np.ndarray,
    y_values: np.ndarray,
    plan: Sequence[np.ndarray],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for bin_id, indices in enumerate(plan):
        y_bin = y_values[indices]
        rows.append({
            "bin_id": int(bin_id),
            "x_mean": float(np.mean(x_values[indices])),
            "y_mean": float(np.mean(y_bin)),
            "y_sem": float(np.std(y_bin, ddof=0) / np.sqrt(len(y_bin))) if len(y_bin) else np.nan,
            "count": int(len(indices)),
        })
    return pd.DataFrame(rows)

def binned_correlation(
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_bins: int,
    analysis: str,
) -> float:
    plan = display_bin_plan(x_values, n_bins, analysis)
    if len(plan) < 3:
        return np.nan
    summary = binned_summary_from_plan(x_values, y_values, plan)
    return correlation_value(
        summary["x_mean"].to_numpy(dtype=float),
        summary["y_mean"].to_numpy(dtype=float),
        method="pearson",
    )

def protein_group_masks(path_values: np.ndarray) -> List[np.ndarray]:
    path_values = np.asarray(path_values, dtype=np.float64)
    return [
        path_values == 1,
        path_values == 2,
        path_values == 3,
        path_values == 4,
        path_values > 4,
    ]

def compute_protein_group_summary(
    embedding_values: np.ndarray,
    path_values: np.ndarray,
) -> pd.DataFrame:
    embedding_values = np.asarray(embedding_values, dtype=np.float64)
    path_values = np.asarray(path_values, dtype=np.float64)
    valid = np.isfinite(embedding_values) & np.isfinite(path_values)
    embedding_values = embedding_values[valid]
    path_values = path_values[valid]

    rows: List[Dict[str, Any]] = []
    for group_id, (label, mask) in enumerate(zip(PROTEIN_GROUP_LABELS, protein_group_masks(path_values))):
        selected = embedding_values[mask]
        rows.append({
            "group_id": int(group_id),
            "ppi_group": label,
            "count": int(len(selected)),
            "mean": float(np.mean(selected)) if len(selected) else np.nan,
            "sem": (
                float(np.std(selected, ddof=0) / np.sqrt(len(selected)))
                if len(selected)
                else np.nan
            ),
        })
    return pd.DataFrame(rows)

def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)

def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return np.nan, np.nan
    return tuple(np.polyfit(x, y, deg=1))

def fit_line_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    confidence: float = 0.95,
    n_grid: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ordinary least-squares fit on binned means, plus confidence band
    for the fitted mean trend.

    Returns
    -------
    x_grid, y_hat, lower, upper
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        # too few points for a stable CI
        x_grid = np.linspace(float(np.min(x)), float(np.max(x)), n_grid)
        slope, intercept = fit_line(x, y)
        y_hat = slope * x_grid + intercept
        return x_grid, y_hat, y_hat, y_hat

    n = len(x)
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    sxx = np.sum((x - x_bar) ** 2)
    sxy = np.sum((x - x_bar) * (y - y_bar))

    slope = sxy / sxx
    intercept = y_bar - slope * x_bar

    y_fitted = intercept + slope * x
    resid = y - y_fitted

    dof = n - 2
    mse = np.sum(resid ** 2) / dof
    s = np.sqrt(mse)

    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), n_grid)
    y_hat = intercept + slope * x_grid

    tcrit = t.ppf(0.5 + confidence / 2.0, dof)

    se_mean = s * np.sqrt(
        1.0 / n + ((x_grid - x_bar) ** 2) / sxx
    )

    lower = y_hat - tcrit * se_mean
    upper = y_hat + tcrit * se_mean

    return x_grid, y_hat, lower, upper

def aggregate_consensus(
    matrices: Sequence[np.ndarray],
    method: str,
) -> np.ndarray:
    if not matrices:
        raise ValueError("At least one repeat matrix is required for consensus.")

    stacked = np.stack(matrices, axis=0)
    with np.errstate(invalid="ignore"):
        if method == "mean":
            return np.nanmean(stacked, axis=0)
        if method == "median":
            return np.nanmedian(stacked, axis=0)

    raise ValueError(
        f"Unknown consensus method {method!r}; expected 'mean' or 'median'."
    )

def append_module_panel_rows(
    rows: List[Dict[str, Any]],
    metric_rows: List[Dict[str, Any]],
    model_name: str,
    analysis: str,
    distance_matrix: np.ndarray,
    genetic_matrix: np.ndarray,
    symmetric: bool,
    n_bins: int = GLOBAL_DISPLAY_BINS,
) -> None:
    x_values, y_values = module_pair_vectors(distance_matrix, genetic_matrix, symmetric)
    plan = display_bin_plan(x_values, n_bins, analysis)
    summary = binned_summary_from_plan(x_values, y_values, plan)
    for row in summary.to_dict("records"):
        rows.append({
            "model": model_name,
            "analysis": analysis,
            **row,
        })
    pearson_r, pearson_p, n_pairs = correlation_test(
        x_values,
        y_values,
        "pearson"
    )

    spearman_r, spearman_p, _ = correlation_test(
        x_values,
        y_values,
        "spearman"
    )

    metric_rows.append({
        "model": model_name,
        "analysis": analysis,

        "n_pairs": n_pairs,

        "pairwise_r": pearson_r,
        "pairwise_p": pearson_p,

        "rank_r": spearman_r,
        "rank_p": spearman_p,

        "r_bin": binned_correlation(
            x_values,
            y_values,
            n_bins,
            analysis,
        ),
    })

def append_protein_rows(
    rows: List[Dict[str, Any]],
    model_name: str,
    embedding_pair_values: np.ndarray,
    path_values: np.ndarray,
) -> None:
    summary = compute_protein_group_summary(embedding_pair_values, path_values)
    for row in summary.to_dict("records"):
        rows.append({
            "model": model_name,
            "analysis": "protein_protein",
            **row,
        })

def triangular_local_structure(
    embedding_distance: np.ndarray,
    k_values: Sequence[int],
    symmetric: bool,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Define top-k neighbours and adaptive triangular-kernel weights.

    The (k+1)-th embedding neighbour supplies the bandwidth. All top-k
    neighbours therefore receive positive weights, while more distant
    candidates receive zero local weight. PPI information is not used to
    define neighbours or weights.
    """
    ranked_distance = np.asarray(
        embedding_distance,
        dtype=np.float64,
    ).copy()
    if symmetric:
        np.fill_diagonal(ranked_distance, np.inf)

    order = np.argsort(ranked_distance, axis=1, kind="mergesort")
    structures: Dict[int, Dict[str, np.ndarray]] = {}

    for k in k_values:
        n_available = ranked_distance.shape[1] - int(symmetric)
        if k < 1 or k + 1 > n_available:
            raise ValueError(
                f"Invalid local-neighbour k={k}; "
                f"only {n_available} candidates are available."
            )

        top_ids = order[:, :k]
        boundary_ids = order[:, k:k + 1]
        bandwidth = np.take_along_axis(
            ranked_distance,
            boundary_ids,
            axis=1,
        )
        top_distance = np.take_along_axis(
            ranked_distance,
            top_ids,
            axis=1,
        )
        weights = np.maximum(
            0.0,
            1.0 - top_distance / bandwidth,
        )
        row_sums = weights.sum(axis=1, keepdims=True)
        if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0):
            raise ValueError(
                f"Non-positive triangular-kernel weight sum for k={k}."
            )
        weights /= row_sums

        structures[int(k)] = {
            "top_ids": top_ids,
            "weights": weights,
        }

    return structures

def local_entity_scores(
    standardized_separation: np.ndarray,
    structures: Dict[int, Dict[str, np.ndarray]],
    symmetric: bool,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Return one paired local-versus-remaining effect per focal entity."""
    standardized_separation = np.asarray(
        standardized_separation,
        dtype=np.float64,
    )
    n_rows, n_columns = standardized_separation.shape
    n_candidates = n_columns - 1 if symmetric else n_columns
    row_ids = np.arange(n_rows)[:, None]
    output: Dict[int, Dict[str, np.ndarray]] = {}

    for k, structure in structures.items():
        top_ids = structure["top_ids"]
        weights = structure["weights"]
        top_values = standardized_separation[row_ids, top_ids]

        if np.any(~np.isfinite(top_values)):
            raise ValueError(
                f"Non-finite standardized separation among top-{k} neighbours."
            )

        local_score = np.sum(weights * top_values, axis=1)

        # Each standardized row has mean zero over its candidate set.
        # Removing the unweighted top-k values therefore gives the exact
        # unweighted mean among all remaining candidates.
        remaining_score = -np.sum(top_values, axis=1) / (
            n_candidates - k
        )
        difference = local_score - remaining_score

        output[int(k)] = {
            "local_score": local_score,
            "remaining_score": remaining_score,
            "difference": difference,
        }

    return output

def bootstrap_mean_interval(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Percentile bootstrap interval over focal-entity differences."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or n_bootstrap <= 0:
        return np.nan, np.nan

    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)
    for bootstrap_id in range(n_bootstrap):
        indices = rng.integers(
            0,
            len(values),
            size=len(values),
        )
        bootstrap_means[bootstrap_id] = float(
            np.mean(values[indices])
        )

    low, high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )
    return float(low), float(high)

def plot_module_observed_panel(
    ax: plt.Axes,
    bins: pd.DataFrame,
    metric: pd.Series,
    analysis: str,
    show_main_reference: Optional[pd.DataFrame] = None,
) -> None:
    bins = bins.sort_values("bin_id")
    x = bins["x_mean"].to_numpy(dtype=float)
    y = bins["y_mean"].to_numpy(dtype=float)
    sem = bins["y_sem"].to_numpy(dtype=float)

    ax.errorbar(
        x,
        y,
        yerr=sem,
        fmt="o",
        markersize=3.4,
        capsize=1.7,
        linewidth=0.75,
        color=OBSERVED_COLORS[analysis],
        ecolor=OBSERVED_COLORS[analysis],
        zorder=3,
    )
    if len(x) >= 3:
        x_line, y_line, y_low, y_high = fit_line_with_ci(x, y, confidence=0.95)

        ax.fill_between(
            x_line,
            y_low,
            y_high,
            color=OBSERVED_COLORS[analysis],
            alpha=0.16,
            linewidth=0,
            zorder=2,
        )

        ax.plot(
            x_line,
            y_line,
            color=OBSERVED_COLORS[analysis],
            linewidth=1.2,
            zorder=3,
        )

    elif len(x) >= 2:
        slope, intercept = fit_line(x, y)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color=OBSERVED_COLORS[analysis],
            linewidth=1.2,
            zorder=3,
        )

    if show_main_reference is not None and not show_main_reference.empty:
        mb = show_main_reference.sort_values("bin_id")
        mx = mb["x_mean"].to_numpy(dtype=float)
        my = mb["y_mean"].to_numpy(dtype=float)
        if len(mx) >= 2:
            slope, intercept = fit_line(mx, my)
            x_line = np.linspace(float(np.min(mx)), float(np.max(mx)), 200)
            ax.plot(
                x_line,
                slope * x_line + intercept,
                color=MAIN_REFERENCE_COLOR,
                linewidth=0.9,
                linestyle="--",
                zorder=1,
                label="Main trend",
            )

    ax.text(
        0.96,
        0.06,
        (
            rf"$r_{{bin}}={float(metric['r_bin']):.2f}$"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
    )
    style_axis(ax)

def plot_protein_observed_panel(
    ax: plt.Axes,
    protein_groups: pd.DataFrame,
    model_name: str,
    main_reference: Optional[pd.DataFrame] = None,
) -> None:
    rows = protein_groups[
        (protein_groups["model"] == model_name)
        & (protein_groups["analysis"] == "protein_protein")
    ].sort_values("group_id")
    x = np.arange(len(PROTEIN_GROUP_LABELS), dtype=float)
    means = rows["mean"].to_numpy(dtype=float)
    sem = rows["sem"].to_numpy(dtype=float)
    ax.bar(
        x,
        means,
        yerr=sem,
        width=0.64,
        color=PROTEIN_GROUP_COLORS,
        edgecolor="0.25",
        linewidth=0.55,
        error_kw={"elinewidth": 0.7, "capsize": 1.8, "ecolor": "0.2"},
        zorder=3,
    )

    if main_reference is not None and model_name != "main":
        main = main_reference.sort_values("group_id")
        ax.plot(
            x,
            main["mean"].to_numpy(dtype=float),
            color=MAIN_REFERENCE_COLOR,
            linestyle="--",
            marker="o",
            markerfacecolor="white",
            markersize=3.0,
            linewidth=0.9,
            label="Main protein",
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(PROTEIN_GROUP_LABELS)
    ax.set_ylim(PROTEIN_Y_MIN, PROTEIN_Y_MAX)
    style_axis(ax)

def plot_leave_one_out_grid(
    loo_names: Sequence[str],
    module_bins: pd.DataFrame,
    module_metrics: pd.DataFrame,
    protein_groups: pd.DataFrame,
    consensus_method: str,
    output_png: Path,
    output_pdf: Path,
    dpi: int,
    panel_width: float,
    panel_height: float,
) -> None:
    loo_names = sorted(loo_names, key=model_sort_key)
    if not loo_names:
        return

    fig, axes = plt.subplots(
        len(loo_names),
        len(ALL_ANALYSES),
        figsize=(panel_width * len(ALL_ANALYSES), panel_height * len(loo_names)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    main_bins = {
        analysis: module_bins[
            (module_bins["model"] == "main")
            & (module_bins["analysis"] == analysis)
        ]
        for analysis in MODULE_ANALYSES
    }
    main_protein = protein_groups[
        (protein_groups["model"] == "main")
        & (protein_groups["analysis"] == "protein_protein")
    ].copy()

    for col_id, analysis in enumerate(ALL_ANALYSES):
        axes[0, col_id].set_title(ANALYSIS_LABELS[analysis], fontsize=9.0, pad=6)

    for row_id, model_name in enumerate(loo_names):
        label = model_name.replace("loo_without_", "LOO-")
        for col_id, analysis in enumerate(ALL_ANALYSES):
            ax = axes[row_id, col_id]
            if analysis == "protein_protein":
                plot_protein_observed_panel(
                    ax,
                    protein_groups,
                    model_name=model_name,
                    main_reference=main_protein,
                )
                if row_id == len(loo_names) - 1:
                    ax.set_xlabel(PROTEIN_X_LABEL, fontsize=7.0)
                ax.set_ylabel(PROTEIN_Y_LABEL, fontsize=6.8)
            else:
                bins = module_bins[
                    (module_bins["model"] == model_name)
                    & (module_bins["analysis"] == analysis)
                ]
                metric = module_metrics[
                    (module_metrics["model"] == model_name)
                    & (module_metrics["analysis"] == analysis)
                ].iloc[0]
                plot_module_observed_panel(
                    ax,
                    bins,
                    metric,
                    analysis,
                    show_main_reference=main_bins[analysis],
                )
                if row_id == len(loo_names) - 1:
                    ax.set_xlabel(MODULE_X_LABEL, fontsize=7.0)
                ax.set_ylabel(MODULE_Y_LABEL, fontsize=6.8)

            if col_id == 0:
                ax.annotate(
                    label,
                    xy=(-0.40, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=7.7,
                    fontweight="bold",
                )
            ax.tick_params(axis="both", labelsize=6.0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.005),
            frameon=False,
            ncol=2,
            fontsize=7.2,
        )

    fig.suptitle(
        (
            "Leave-one-repeat-out "
            f"{consensus_method}-rank consensus: embedding–PPI concordance"
        ),
        y=1.015,
        fontsize=11.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.990), h_pad=0.70, w_pad=0.72)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

def summarize_local_observed(
    main_entity_scores: pd.DataFrame,
    bootstrap_rng: np.random.Generator,
) -> pd.DataFrame:
    """Summarize observed local-neighbour effects with bootstrap intervals."""
    rows: List[Dict[str, Any]] = []
    for analysis in MODULE_ANALYSES:
        for k in LOCAL_K_VALUES:
            observed_values = main_entity_scores[
                (main_entity_scores["analysis"] == analysis)
                & (main_entity_scores["k"] == k)
            ]["difference"].to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_mean_interval(
                observed_values,
                LOCAL_BOOTSTRAP_ITERATIONS,
                bootstrap_rng,
            )
            rows.append({
                "analysis": analysis,
                "analysis_label": ANALYSIS_LABELS[analysis],
                "k": int(k),
                "n_entities": int(len(observed_values)),
                "mean_difference": float(np.mean(observed_values)),
                "median_difference": float(np.median(observed_values)),
                "fraction_local_lower": float(np.mean(observed_values < 0)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            })
    return pd.DataFrame(rows)


def plot_consensus_global(
    module_bins: pd.DataFrame,
    module_metrics: pd.DataFrame,
    protein_groups: pd.DataFrame,
    consensus_model_name: str,
    output_png: Path,
    output_pdf: Path,
    dpi: int,
    panel_width: float,
    panel_height: float,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(ALL_ANALYSES),
        figsize=(panel_width * len(ALL_ANALYSES), panel_height),
        squeeze=False,
    )
    for col_id, analysis in enumerate(ALL_ANALYSES):
        ax = axes[0, col_id]
        if analysis == "protein_protein":
            plot_protein_observed_panel(ax, protein_groups, consensus_model_name)
            ax.set_xlabel(PROTEIN_X_LABEL, fontsize=7.0)
            ax.set_ylabel(PROTEIN_Y_LABEL, fontsize=6.8)
        else:
            bins = module_bins[
                (module_bins["model"] == consensus_model_name)
                & (module_bins["analysis"] == analysis)
            ]
            metric = module_metrics[
                (module_metrics["model"] == consensus_model_name)
                & (module_metrics["analysis"] == analysis)
            ].iloc[0]
            plot_module_observed_panel(ax, bins, metric, analysis)
            ax.set_xlabel(MODULE_X_LABEL, fontsize=7.0)
            ax.set_ylabel(MODULE_Y_LABEL, fontsize=6.8)
        ax.set_title(ANALYSIS_LABELS[analysis], fontsize=9.0, pad=6)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_local_observed(
    summary_df: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(LOCAL_PLOT_ORDER),
        figsize=(3.2 * len(LOCAL_PLOT_ORDER), 3.4),
        squeeze=False,
    )
    for col_id, analysis in enumerate(LOCAL_PLOT_ORDER):
        ax = axes[0, col_id]
        subset = (
            summary_df[summary_df["analysis"] == analysis]
            .set_index("k")
            .reindex(LOCAL_K_VALUES)
            .reset_index()
        )
        y = np.arange(len(LOCAL_K_VALUES), dtype=float)
        mean = subset["mean_difference"].to_numpy(dtype=float)
        low = subset["bootstrap_ci_low"].to_numpy(dtype=float)
        high = subset["bootstrap_ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            mean,
            y,
            xerr=[mean - low, high - mean],
            fmt="o",
            markersize=5.4,
            capsize=3.2,
            linewidth=1.35,
            color=OBSERVED_COLORS[analysis],
            ecolor=OBSERVED_COLORS[analysis],
            markerfacecolor="white",
            markeredgecolor=OBSERVED_COLORS[analysis],
        )
        ax.axvline(0.0, linestyle="--", linewidth=1.0, color=MAIN_REFERENCE_COLOR)
        ax.set_yticks(y)
        ax.set_yticklabels([f"k = {k}" for k in LOCAL_K_VALUES])
        ax.invert_yaxis()
        ax.set_xlabel("Mean difference: top-k local score − remaining score", fontsize=8.0)
        ax.set_title(ANALYSIS_LABELS[analysis], fontsize=9.4, pad=7)
        style_axis(ax)
    axes[0, 0].set_ylabel("Neighbourhood size", fontsize=8.0)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    start_time = time.time()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require_files([
        HERB_LIST_FILE,
        PROTEIN_LIST_FILE,
        HERB_TARGETS_FILE,
        DISEASE_TABLE_FILE,
        DISEASE_GENES_FILE,
        DISEASE_MMSYM_FILE,
        PPI_GENES_FILE,
        PPI_DISTANCE_FILE,
        MAIN_HERB_EMBEDDING_FILE,
        MAIN_PROTEIN_EMBEDDING_FILE,
        MAIN_DISEASE_EMBEDDING_FILE,
    ])

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {output_dir}")
    print(f"Consensus method: {args.consensus_method}")

    # ------------------------------------------------------------------
    # Load entities, annotations and PPI distances
    # ------------------------------------------------------------------
    herb_list = list(load_pickle(HERB_LIST_FILE))
    protein_list = list(load_pickle(PROTEIN_LIST_FILE))
    herb_targets_raw = load_pickle(HERB_TARGETS_FILE)
    if not isinstance(herb_targets_raw, dict):
        raise TypeError("herb_targets.pkl must contain a herb -> target-list dict.")

    disease_table = pd.read_excel(DISEASE_TABLE_FILE)
    disease_table.columns = disease_table.columns.astype(str).str.strip()
    if "Disease" not in disease_table.columns:
        raise ValueError("Disease table does not contain a 'Disease' column.")
    disease_names = disease_table["Disease"].astype(str).tolist()

    disease_genes_raw = load_pickle(DISEASE_GENES_FILE)
    if isinstance(disease_genes_raw, dict):
        disease_genes = [
            normalize_entity_terms(disease_genes_raw.get(disease, []))
            for disease in disease_names
        ]
    else:
        disease_genes = [
            normalize_entity_terms(value)
            for value in list(disease_genes_raw)[:len(disease_names)]
        ]
    if len(disease_genes) != len(disease_names):
        raise ValueError("Disease names and disease-gene rows are not aligned.")

    disease_mmsym = load_pickle(DISEASE_MMSYM_FILE)
    disease_symptom_counts = np.asarray([
        len(normalize_entity_terms(disease_mmsym.get(disease, [])))
        for disease in disease_names
    ], dtype=int)

    ppi_genes = list(load_pickle(PPI_GENES_FILE))
    ppi_distance = np.load(PPI_DISTANCE_FILE, mmap_mode="r")
    if ppi_distance.shape != (len(ppi_genes), len(ppi_genes)):
        raise ValueError("PPI distance matrix shape does not match ppi_genes.")

    ppi_gene_to_id = {canonical_gene(gene): index for index, gene in enumerate(ppi_genes)}
    finite_ppi_values = np.asarray(ppi_distance[ppi_distance >= 0])
    if finite_ppi_values.size == 0:
        raise ValueError("PPI distance matrix contains no finite distances.")
    network_max_distance = float(np.max(finite_ppi_values))

    herb_gene_sets = [
        normalize_entity_terms(herb_targets_raw.get(herb, []))
        for herb in herb_list
    ]
    target_gene_set = {
        canonical_gene(gene)
        for genes in herb_gene_sets
        for gene in genes
        if canonical_gene(gene) in ppi_gene_to_id
    }
    disease_gene_set = {
        canonical_gene(gene)
        for genes in disease_genes
        for gene in genes
        if canonical_gene(gene) in ppi_gene_to_id
    }
    relevant_genes = sorted(target_gene_set | disease_gene_set)
    gene_to_basis = {gene: index for index, gene in enumerate(relevant_genes)}
    basis_ppi_ids = np.asarray([ppi_gene_to_id[gene] for gene in relevant_genes], dtype=np.int64)
    relevant_distance = replace_disconnected(
        ppi_distance[np.ix_(basis_ppi_ids, basis_ppi_ids)],
        disconnected_distance=network_max_distance,
    )

    herb_membership_all = build_membership_matrix(herb_gene_sets, gene_to_basis)
    disease_membership_all = build_membership_matrix(disease_genes, gene_to_basis)

    herb_sizes = np.asarray(herb_membership_all.sum(axis=1)).ravel().astype(int)
    disease_sizes = np.asarray(disease_membership_all.sum(axis=1)).ravel().astype(int)

    herb_analysis_ids = np.where(herb_sizes >= args.min_module_size)[0]
    disease_analysis_ids = np.where(
        (disease_sizes >= args.min_module_size)
        & (disease_symptom_counts >= args.min_disease_symptoms)
    )[0]

    herb_membership = herb_membership_all[herb_analysis_ids]
    disease_membership = disease_membership_all[disease_analysis_ids]

    protein_embedding_ids: List[int] = []
    protein_ppi_ids: List[int] = []
    for protein_id, raw_gene in enumerate(protein_list):
        gene = canonical_gene(raw_gene)
        if gene in ppi_gene_to_id:
            protein_embedding_ids.append(protein_id)
            protein_ppi_ids.append(ppi_gene_to_id[gene])
    protein_embedding_ids_array = np.asarray(protein_embedding_ids, dtype=int)
    protein_ppi_ids_array = np.asarray(protein_ppi_ids, dtype=np.int64)

    print(f"Herbs analyzed: {len(herb_analysis_ids)}")
    print(f"Diseases analyzed: {len(disease_analysis_ids)}")
    print(f"Mapped proteins analyzed: {len(protein_embedding_ids_array)}")

    # ------------------------------------------------------------------
    # Observed PPI module separation and standardized genetic matrices
    # ------------------------------------------------------------------
    herb_separation, _ = symmetric_module_metrics(herb_membership, relevant_distance)
    disease_separation, _ = symmetric_module_metrics(disease_membership, relevant_distance)
    disease_herb_separation, _ = cross_module_metrics(
        disease_membership,
        herb_membership,
        relevant_distance,
    )

    standardized_genetic = {
        "herb_herb": row_standardize_symmetric(herb_separation),
        "disease_disease": row_standardize_symmetric(disease_separation),
        "disease_herb": row_standardize_cross(disease_herb_separation),
    }
    symmetric_lookup = {
        "herb_herb": True,
        "disease_disease": True,
        "disease_herb": False,
    }

    # Protein PPI paths for the observed gene labels.
    protein_pair_i, protein_pair_j = upper_triangle_indices(len(protein_embedding_ids_array))
    observed_protein_ppi = np.asarray(
        ppi_distance[
            protein_ppi_ids_array[protein_pair_i],
            protein_ppi_ids_array[protein_pair_j],
        ],
        dtype=np.float64,
    )
    observed_protein_connected = observed_protein_ppi >= 0

    # ------------------------------------------------------------------
    # Load embeddings and convert distances to percentile ranks
    # ------------------------------------------------------------------
    model_files = discover_embedding_files(args.repeat_models)
    repeat_names = sorted([name for name in model_files if name != "main"], key=model_sort_key)
    print("Complete repeat models: " + ", ".join(repeat_names))

    module_percentiles: Dict[str, Dict[str, np.ndarray]] = {}
    protein_pair_percentiles: Dict[str, np.ndarray] = {}
    main_embedding_distances: Dict[str, np.ndarray] = {}

    for model_name, files in model_files.items():
        herb_embedding = load_embedding_array(
            files["herb"],
            expected_rows=len(herb_list),
            label=f"{model_name} herb embedding",
        )[herb_analysis_ids]
        disease_embedding = load_embedding_array(
            files["disease"],
            expected_rows=len(disease_names),
            label=f"{model_name} disease embedding",
        )[disease_analysis_ids]
        protein_embedding = load_embedding_array(
            files["protein"],
            expected_rows=len(protein_list),
            label=f"{model_name} protein embedding",
        )[protein_embedding_ids_array]

        distances = {
            "herb_herb": cdist(herb_embedding, herb_embedding, metric="euclidean"),
            "disease_disease": cdist(disease_embedding, disease_embedding, metric="euclidean"),
            "disease_herb": cross_modal_centered_distance(disease_embedding, herb_embedding),
        }
        module_percentiles[model_name] = {
            analysis: row_percentile_distance_matrix(
                distances[analysis],
                symmetric=symmetric_lookup[analysis],
            )
            for analysis in MODULE_ANALYSES
        }

        if model_name == "main":
            main_embedding_distances = {
                analysis: np.asarray(
                    distances[analysis],
                    dtype=np.float64,
                ).copy()
                for analysis in MODULE_ANALYSES
            }

        protein_distance = cdist(protein_embedding, protein_embedding, metric="euclidean")
        protein_pair_values = protein_distance[protein_pair_i, protein_pair_j]
        protein_pair_percentiles[model_name] = pair_percentile_vector(protein_pair_values)

    # Repeat consensus and leave-one-repeat-out consensus.
    consensus_method = args.consensus_method
    consensus_model_name = f"repeat_{consensus_method}_rank_consensus"

    module_percentiles[consensus_model_name] = {}
    for analysis in MODULE_ANALYSES:
        module_percentiles[consensus_model_name][analysis] = aggregate_consensus(
            [module_percentiles[name][analysis] for name in repeat_names],
            method=consensus_method,
        )

    protein_pair_percentiles[consensus_model_name] = aggregate_consensus(
        [protein_pair_percentiles[name] for name in repeat_names],
        method=consensus_method,
    )

    loo_names: List[str] = []
    for excluded in repeat_names:
        loo_name = "loo_without_" + excluded
        loo_names.append(loo_name)
        included_names = [
            name for name in repeat_names if name != excluded
        ]

        module_percentiles[loo_name] = {}
        for analysis in MODULE_ANALYSES:
            module_percentiles[loo_name][analysis] = aggregate_consensus(
                [
                    module_percentiles[name][analysis]
                    for name in included_names
                ],
                method=consensus_method,
            )

        protein_pair_percentiles[loo_name] = aggregate_consensus(
            [protein_pair_percentiles[name] for name in included_names],
            method=consensus_method,
        )

    # ------------------------------------------------------------------
    # Observed module and protein panel data
    # ------------------------------------------------------------------
    module_bin_rows: List[Dict[str, Any]] = []
    module_metric_rows: List[Dict[str, Any]] = []
    protein_group_rows: List[Dict[str, Any]] = []

    output_model_names = ["main", consensus_model_name] + loo_names
    for model_name in output_model_names:
        for analysis in MODULE_ANALYSES:
            append_module_panel_rows(
                rows=module_bin_rows,
                metric_rows=module_metric_rows,
                model_name=model_name,
                analysis=analysis,
                distance_matrix=module_percentiles[model_name][analysis],
                genetic_matrix=standardized_genetic[analysis],
                symmetric=symmetric_lookup[analysis],
                n_bins=GLOBAL_DISPLAY_BINS,
            )

        append_protein_rows(
            rows=protein_group_rows,
            model_name=model_name,
            embedding_pair_values=protein_pair_percentiles[model_name][observed_protein_connected],
            path_values=observed_protein_ppi[observed_protein_connected],
        )

    module_bins_df = pd.DataFrame(module_bin_rows)
    module_metrics_df = pd.DataFrame(module_metric_rows)
    protein_groups_df = pd.DataFrame(protein_group_rows)

    module_bins_df.to_csv(output_dir / "observed_module_percentile_binned_data.csv", index=False, encoding="utf_8_sig")
    module_metrics_df.to_csv(output_dir / "observed_module_percentile_metrics.csv", index=False, encoding="utf_8_sig")
    protein_groups_df.to_csv(output_dir / "observed_protein_percentile_group_data.csv", index=False, encoding="utf_8_sig")

    if set(main_embedding_distances) != set(MODULE_ANALYSES):
        raise RuntimeError(
            "Main-model raw embedding distances were not retained for all "
            "module analyses."
        )

    local_entity_names = {
        "herb_herb": [
            str(herb_list[index])
            for index in herb_analysis_ids
        ],
        "disease_disease": [
            str(disease_names[index])
            for index in disease_analysis_ids
        ],
        "disease_herb": [
            str(disease_names[index])
            for index in disease_analysis_ids
        ],
    }
    main_local_entity_rows: List[Dict[str, Any]] = []

    for analysis in MODULE_ANALYSES:
        local_structures = triangular_local_structure(
            main_embedding_distances[analysis],
            LOCAL_K_VALUES,
            symmetric=symmetric_lookup[analysis],
        )
        local_scores = local_entity_scores(
            standardized_genetic[analysis],
            local_structures,
            symmetric=symmetric_lookup[analysis],
        )

        for k in LOCAL_K_VALUES:
            scores = local_scores[k]
            for entity_id, entity_name in enumerate(
                local_entity_names[analysis]
            ):
                main_local_entity_rows.append({
                    "model": "main",
                    "analysis": analysis,
                    "entity": entity_name,
                    "k": int(k),
                    "local_score": float(
                        scores["local_score"][entity_id]
                    ),
                    "remaining_score": float(
                        scores["remaining_score"][entity_id]
                    ),
                    "difference": float(
                        scores["difference"][entity_id]
                    ),
                })

    main_local_entity_df = pd.DataFrame(
        main_local_entity_rows
    )
    main_local_entity_df.to_csv(
        output_dir / "main_local_neighbour_entity_scores.csv",
        index=False,
        encoding="utf_8_sig",
    )

    # ------------------------------------------------------------------

    local_summary_df = summarize_local_observed(
        main_entity_scores=main_local_entity_df,
        bootstrap_rng=np.random.default_rng(args.seed + 100003),
    )
    local_summary_df.to_csv(
        output_dir / "local_neighbour_observed_summary.csv",
        index=False,
        encoding="utf_8_sig",
    )

    if not args.skip_figure:
        plot_consensus_global(
            module_bins=module_bins_df,
            module_metrics=module_metrics_df,
            protein_groups=protein_groups_df,
            consensus_model_name=consensus_model_name,
            output_png=output_dir / "global_embedding_ppi_concordance.png",
            output_pdf=output_dir / "global_embedding_ppi_concordance.pdf",
            dpi=args.figure_dpi,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
        )
        plot_local_observed(
            summary_df=local_summary_df,
            output_png=output_dir / "local_embedding_ppi_concordance.png",
            output_pdf=output_dir / "local_embedding_ppi_concordance.pdf",
            dpi=args.figure_dpi,
        )
        plot_leave_one_out_grid(
            loo_names=loo_names,
            module_bins=module_bins_df,
            module_metrics=module_metrics_df,
            protein_groups=protein_groups_df,
            consensus_method=consensus_method,
            output_png=output_dir / "training_seed_stability.png",
            output_pdf=output_dir / "training_seed_stability.pdf",
            dpi=args.figure_dpi,
            panel_width=args.panel_width,
            panel_height=args.panel_height * 0.72,
        )

    report = {
        "script": Path(__file__).name,
        "project_root": PROJECT_ROOT,
        "output_dir": output_dir,
        "settings": {
            "seed": args.seed,
            "min_module_size": args.min_module_size,
            "min_disease_symptoms": args.min_disease_symptoms,
            "global_display_bins": GLOBAL_DISPLAY_BINS,
            "display_bin_merging": {
                "disease_disease": "first two equal-frequency bins merged into one closest-decile bin"
            },
            "consensus_method": consensus_method,
            "repeat_consensus": (
                f"{consensus_method} of repeat-specific entity-relative percentile ranks; "
                "main is not used as the alignment target"
            ),
            "leave_one_out": (
                f"{consensus_method} percentile-rank consensus after excluding one repeat model"
            ),
            "disease_herb_cross_modal_centering": True,
            "protein_y_limits": [PROTEIN_Y_MIN, PROTEIN_Y_MAX],
            "local_k_values": list(LOCAL_K_VALUES),
            "local_weighting": (
                "adaptive triangular kernel with bandwidth equal to the (k+1)-th "
                "raw embedding-neighbour distance"
            ),
            "local_background": (
                "unweighted mean standardized PPI separation among all remaining candidates"
            ),
            "local_bootstrap_iterations": LOCAL_BOOTSTRAP_ITERATIONS,
            "repeat_models": repeat_names,
        },
        "entity_counts": {
            "herbs_total": len(herb_list),
            "herbs_analyzed": len(herb_analysis_ids),
            "diseases_total": len(disease_names),
            "diseases_analyzed": len(disease_analysis_ids),
            "proteins_total": len(protein_list),
            "proteins_analyzed": len(protein_embedding_ids_array),
            "relevant_ppi_genes": len(relevant_genes),
        },
        "elapsed_seconds": time.time() - start_time,
    }
    save_json(report, output_dir / "analysis_report.json")

    print("\nSaved:")
    for filename in [
        "observed_module_percentile_binned_data.csv",
        "observed_module_percentile_metrics.csv",
        "observed_protein_percentile_group_data.csv",
        "main_local_neighbour_entity_scores.csv",
        "local_neighbour_observed_summary.csv",
        "global_embedding_ppi_concordance.png",
        "global_embedding_ppi_concordance.pdf",
        "local_embedding_ppi_concordance.png",
        "local_embedding_ppi_concordance.pdf",
        "training_seed_stability.png",
        "training_seed_stability.pdf",
        "analysis_report.json",
    ]:
        path = output_dir / filename
        if path.exists():
            print(path)


if __name__ == "__main__":
    main()
