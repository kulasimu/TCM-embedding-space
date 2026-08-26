#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clinical distance-based analyses for the general TCM clinical cases.

This script implements the manuscript analyses based on the model-eligible
clinical-case dataframe and row-aligned embeddings:

1. Formula-to-individual-symptom relationships
   - prescribed formula to alleviated versus unalleviated initial symptoms;
   - Euclidean distance in TCM-ES;
   - case-level paired comparison;
   - paired two-sided t-test and 95% CI for the paired mean difference.

2. Bidirectional complete-set retrieval
   - initial symptom-pattern -> prescribed formula retrieval;
   - prescribed formula -> alleviated symptom-pattern retrieval;
   - TCM-ES complete-set encoder embeddings versus co-occurrence SVD/graph
     pooled embeddings;
   - 50 candidates, 100 repeated samplings by default;
   - Hit@1, Hit@5, Hit@10; error bars are SEM across repeated samplings.

3. Optional formula-symptom ranking diagnostics
   - overall case-balanced positive-negative pairwise AUC;
   - frequency-matched same-case negative controls with caliper sensitivity;
   - individual-symptom AUC difference between TCM-ES and co-occurrence SVD
     across symptom-frequency octiles.

All analyses are retrospective association/ranking analyses. No supervised
clinical-outcome model is trained in this script.
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SOURCE_ORDER = ["Original", "SVD baseline", "Graph baseline"]
MODEL_ORDER = SOURCE_ORDER.copy()
MODEL_LABELS = {
    "Original": "TCM-ES",
    "SVD baseline": "Co-occurrence SVD",
    "Graph baseline": "Co-occurrence graph",
}
MODEL_COLORS = {
    "Original": "#5B7FA6",
    "SVD baseline": "#F3A13B",
    "Graph baseline": "#7DB56F",
}

SET_SOURCE_ORDER = ["Original_encoder_set", "SVD baseline", "Graph baseline"]
SET_SOURCE_LABELS = {
    "Original_encoder_set": "TCM-ES",
    "SVD baseline": "Co-occurrence SVD",
    "Graph baseline": "Co-occurrence graph",
}
SET_SOURCE_COLORS = {
    "Original_encoder_set": MODEL_COLORS["Original"],
    "SVD baseline": MODEL_COLORS["SVD baseline"],
    "Graph baseline": MODEL_COLORS["Graph baseline"],
}

# Paired-distance colours follow the main-text style: alleviated = orange,
# comparison = light grey.
ALLEVIATED_DISTANCE_COLOR = "#F39C12"
UNALLEVIATED_DISTANCE_COLOR = "#D9D9D9"


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_pickle(path: str):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def unique_list(values: Iterable) -> List:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig, out_path: str) -> None:
    out_path = str(out_path)
    stem = out_path[:-4] if out_path.lower().endswith(".png") else out_path
    fig.savefig(stem + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".svg", bbox_inches="tight")


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def pair_correct(pos_score: float, neg_score: float) -> float:
    if pos_score > neg_score:
        return 1.0
    if pos_score < neg_score:
        return 0.0
    return 0.5


def auc_rank(y_true, score) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(s).rank(method="average").to_numpy()
    sum_pos = ranks[y == 1].sum()
    return float(
        (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def zscore_within_case(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(values, dtype=float)
    return (values - np.nanmean(values)) / sd


def format_p_value(p: float) -> str:
    if not np.isfinite(p):
        return "P = NA"
    if p < 1e-4:
        return "P < 0.0001"
    if p < 0.001:
        return f"P = {p:.2e}"
    return f"P = {p:.3f}"


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# -----------------------------------------------------------------------------
# Loading and row-alignment checks
# -----------------------------------------------------------------------------

def load_clinical_cases(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path).reset_index(drop=True)
    required = ["Cases_id", "Initial_symptoms", "Formula", "Alleviated_symptoms"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Clinical dataframe missing required columns: {missing}")
    return df


def load_standard_terms(symptom_list_path: str):
    symptom_list = load_pickle(symptom_list_path)
    symptom2id = {symptom: i for i, symptom in enumerate(symptom_list)}
    return symptom_list, symptom2id


def load_symptom_frequency(
    path: str,
    symptom_list: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    freq = np.zeros(len(symptom_list), dtype=float)
    ids = df["local_entity_id"].astype(int).to_numpy()
    freq[ids] = df["frequency_in_train"].astype(float).to_numpy()
    return freq, np.log1p(freq)


def _load_required_array(path: str, name: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")
    return np.asarray(load_pickle(path), dtype=np.float32)


def _check_case_id_alignment(
    clinical_df: pd.DataFrame,
    case_id_path: str,
    source_name: str,
) -> None:
    if not os.path.exists(case_id_path):
        raise FileNotFoundError(
            f"Missing {source_name} case_ids.pkl required for row-alignment check: "
            f"{case_id_path}"
        )
    expected = clinical_df["Cases_id"].tolist()
    observed = list(load_pickle(case_id_path))
    if len(expected) != len(observed):
        raise ValueError(
            f"{source_name} case_ids rows {len(observed)} != clinical rows {len(expected)}"
        )
    mismatch = [i for i, (a, b) in enumerate(zip(expected, observed)) if a != b]
    if mismatch:
        i = mismatch[0]
        raise ValueError(
            f"{source_name} case-ID order mismatch at row {i}: "
            f"clinical={expected[i]!r}, embeddings={observed[i]!r}"
        )


def load_embedding_sources(args, clinical_df: pd.DataFrame) -> Dict[str, Dict]:
    n_cases = len(clinical_df)

    _check_case_id_alignment(
        clinical_df,
        os.path.join(args.tcm_clinical_emb_dir, "case_ids.pkl"),
        "TCM-ES",
    )
    _check_case_id_alignment(
        clinical_df,
        os.path.join(
            args.cooc_svd_emb_dir,
            args.clinical_case_subdir,
            "case_ids.pkl",
        ),
        "Co-occurrence SVD",
    )
    _check_case_id_alignment(
        clinical_df,
        os.path.join(
            args.cooc_graph_emb_dir,
            args.clinical_case_subdir,
            "case_ids.pkl",
        ),
        "Co-occurrence graph",
    )

    sources = {
        "Original": {
            "formula_embeddings": _load_required_array(
                os.path.join(args.tcm_clinical_emb_dir, "case_herb_embeddings.pkl"),
                "TCM-ES case_herb_embeddings.pkl",
            ),
            "case_symptom_embeddings": _load_required_array(
                os.path.join(args.tcm_clinical_emb_dir, "case_symptom_embeddings.pkl"),
                "TCM-ES case_symptom_embeddings.pkl",
            ),
            "symptom_embeddings": _load_required_array(
                os.path.join(args.tcm_emb_dir, "individual_symptom_embeddings.pkl"),
                "TCM-ES individual_symptom_embeddings.pkl",
            ),
            "improved_symptom_set_embeddings": _load_required_array(
                os.path.join(
                    args.tcm_clinical_emb_dir,
                    "case_improved_symptom_embeddings.pkl",
                ),
                "TCM-ES case_improved_symptom_embeddings.pkl",
            ),
        },
        "SVD baseline": {
            "formula_embeddings": _load_required_array(
                os.path.join(
                    args.cooc_svd_emb_dir,
                    args.clinical_case_subdir,
                    "case_herb_embeddings.pkl",
                ),
                "SVD case_herb_embeddings.pkl",
            ),
            "case_symptom_embeddings": _load_required_array(
                os.path.join(
                    args.cooc_svd_emb_dir,
                    args.clinical_case_subdir,
                    "case_symptom_embeddings.pkl",
                ),
                "SVD case_symptom_embeddings.pkl",
            ),
            "symptom_embeddings": _load_required_array(
                os.path.join(args.cooc_svd_emb_dir, "individual_symptom_embeddings.pkl"),
                "SVD individual_symptom_embeddings.pkl",
            ),
            "improved_symptom_set_embeddings": None,
        },
        "Graph baseline": {
            "formula_embeddings": _load_required_array(
                os.path.join(
                    args.cooc_graph_emb_dir,
                    args.clinical_case_subdir,
                    "case_herb_embeddings.pkl",
                ),
                "Graph case_herb_embeddings.pkl",
            ),
            "case_symptom_embeddings": _load_required_array(
                os.path.join(
                    args.cooc_graph_emb_dir,
                    args.clinical_case_subdir,
                    "case_symptom_embeddings.pkl",
                ),
                "Graph case_symptom_embeddings.pkl",
            ),
            "symptom_embeddings": _load_required_array(
                os.path.join(args.cooc_graph_emb_dir, "individual_symptom_embeddings.pkl"),
                "Graph individual_symptom_embeddings.pkl",
            ),
            "improved_symptom_set_embeddings": None,
        },
    }

    for source_name, source in sources.items():
        for key in ["formula_embeddings", "case_symptom_embeddings"]:
            if source[key].shape[0] != n_cases:
                raise ValueError(
                    f"{source_name} {key} rows {source[key].shape[0]} "
                    f"!= clinical rows {n_cases}"
                )
        if source["formula_embeddings"].shape[1] != source["symptom_embeddings"].shape[1]:
            raise ValueError(
                f"{source_name}: formula and individual-symptom dimensions differ."
            )

    if sources["Original"]["improved_symptom_set_embeddings"].shape[0] != n_cases:
        raise ValueError(
            "TCM-ES improved-symptom set embeddings are not row-aligned with clinical data."
        )

    return sources


# -----------------------------------------------------------------------------
# Distance / similarity scoring
# -----------------------------------------------------------------------------

def score_formula_to_symptoms(
    formula_emb: np.ndarray,
    symptom_embs: np.ndarray,
    metric: str,
) -> np.ndarray:
    formula_emb = np.asarray(formula_emb, dtype=np.float32)
    symptom_embs = np.asarray(symptom_embs, dtype=np.float32)
    if metric == "euclidean":
        return -np.linalg.norm(symptom_embs - formula_emb[None, :], axis=1)
    if metric == "cosine":
        formula_norm = formula_emb / max(np.linalg.norm(formula_emb), 1e-12)
        return normalize_rows(symptom_embs) @ formula_norm
    raise ValueError(f"Unknown metric: {metric}")


def score_query_to_candidates(
    query_embs: np.ndarray,
    candidate_embs: np.ndarray,
    metric: str,
) -> np.ndarray:
    query_embs = np.asarray(query_embs, dtype=np.float32)
    candidate_embs = np.asarray(candidate_embs, dtype=np.float32)
    if metric == "cosine":
        return normalize_rows(query_embs) @ normalize_rows(candidate_embs).T
    if metric == "euclidean":
        out = np.empty((query_embs.shape[0], candidate_embs.shape[0]), dtype=np.float32)
        for i in range(query_embs.shape[0]):
            out[i] = -np.linalg.norm(candidate_embs - query_embs[i][None, :], axis=1)
        return out
    raise ValueError(f"Unknown retrieval metric: {metric}")


def mean_pool_symptom_set_embeddings(
    symptom_id_sets: List[List[int]],
    symptom_embeddings: np.ndarray,
) -> np.ndarray:
    dim = symptom_embeddings.shape[1]
    out = np.zeros((len(symptom_id_sets), dim), dtype=np.float32)
    for i, ids in enumerate(symptom_id_sets):
        ids = sorted(set(int(x) for x in ids))
        if ids:
            out[i] = symptom_embeddings[ids].mean(axis=0)
    return out


def make_formula_key(formula) -> str:
    if isinstance(formula, (list, tuple, set)):
        herbs = [str(x) for x in formula]
    else:
        herbs = [str(formula)]
    return "||".join(sorted(set(herbs)))


# -----------------------------------------------------------------------------
# Formula-to-individual-symptom relationships: paired alleviated vs unalleviated distances
# -----------------------------------------------------------------------------

def build_paired_alleviated_distance_table(
    clinical_df: pd.DataFrame,
    symptom2id: Dict[str, int],
    formula_embeddings: np.ndarray,
    symptom_embeddings: np.ndarray,
) -> pd.DataFrame:
    """
    Build case-level paired distances for the general TCM clinical analysis.

    Alleviated symptoms are the initial-visit symptoms also documented as
    alleviated at follow-up. Unalleviated symptoms are the remaining mapped
    initial-visit symptoms. Only cases containing at least one symptom in each
    group contribute to the paired comparison.
    """
    if formula_embeddings.shape[0] != len(clinical_df):
        raise ValueError("Formula embeddings are not row-aligned with clinical data.")

    rows = []
    for case_row_idx, row in clinical_df.iterrows():
        initial = [
            symptom
            for symptom in unique_list(row["Initial_symptoms"])
            if symptom in symptom2id
        ]
        alleviated_set = set(row["Alleviated_symptoms"])
        alleviated = [symptom for symptom in initial if symptom in alleviated_set]
        unalleviated = [symptom for symptom in initial if symptom not in alleviated_set]

        if len(alleviated) == 0 or len(unalleviated) == 0:
            continue

        formula = formula_embeddings[int(case_row_idx)]
        allev_ids = [symptom2id[s] for s in alleviated]
        unallev_ids = [symptom2id[s] for s in unalleviated]

        allev_dist = np.linalg.norm(
            symptom_embeddings[allev_ids] - formula[None, :],
            axis=1,
        )
        unallev_dist = np.linalg.norm(
            symptom_embeddings[unallev_ids] - formula[None, :],
            axis=1,
        )

        allev_mean = float(np.mean(allev_dist))
        unallev_mean = float(np.mean(unallev_dist))

        rows.append({
            "case_id": row["Cases_id"],
            "case_row_idx": int(case_row_idx),
            "n_initial_symptoms_mapped": int(len(initial)),
            "n_alleviated_symptoms": int(len(alleviated)),
            "n_unalleviated_symptoms": int(len(unalleviated)),
            "alleviated_symptoms": ";".join(alleviated),
            "unalleviated_symptoms": ";".join(unalleviated),
            "alleviated_mean_distance": allev_mean,
            "unalleviated_mean_distance": unallev_mean,
            "paired_difference_unalleviated_minus_alleviated": (
                unallev_mean - allev_mean
            ),
        })

    return pd.DataFrame(rows)


def summarize_paired_distances(paired_df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "alleviated_mean_distance",
        "unalleviated_mean_distance",
        "paired_difference_unalleviated_minus_alleviated",
    ]
    missing = [col for col in required if col not in paired_df.columns]
    if missing:
        raise ValueError(f"Paired-distance table missing columns: {missing}")

    n = len(paired_df)
    allev = paired_df["alleviated_mean_distance"].astype(float).to_numpy()
    unallev = paired_df["unalleviated_mean_distance"].astype(float).to_numpy()
    diff = paired_df[
        "paired_difference_unalleviated_minus_alleviated"
    ].astype(float).to_numpy()

    if n >= 2:
        t_stat, p_value = stats.ttest_rel(unallev, allev, nan_policy="omit")
        diff_mean = float(np.mean(diff))
        diff_sd = float(np.std(diff, ddof=1))
        diff_sem = diff_sd / np.sqrt(n)
        crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_low = diff_mean - crit * diff_sem
        ci_high = diff_mean + crit * diff_sem
        cohens_dz = diff_mean / diff_sd if diff_sd > 0 else np.nan
    elif n == 1:
        t_stat = p_value = diff_sd = diff_sem = ci_low = ci_high = cohens_dz = np.nan
        diff_mean = float(diff[0])
    else:
        t_stat = p_value = diff_sd = diff_sem = ci_low = ci_high = cohens_dz = np.nan
        diff_mean = np.nan

    row = {
        "n_paired_cases": int(n),
        "n_alleviated_symptoms_total": int(
            paired_df["n_alleviated_symptoms"].sum()
        ) if "n_alleviated_symptoms" in paired_df else np.nan,
        "n_unalleviated_symptoms_total": int(
            paired_df["n_unalleviated_symptoms"].sum()
        ) if "n_unalleviated_symptoms" in paired_df else np.nan,
        "mean_alleviated_distance": float(np.mean(allev)) if n else np.nan,
        "sd_alleviated_distance": float(np.std(allev, ddof=1)) if n >= 2 else np.nan,
        "mean_unalleviated_distance": float(np.mean(unallev)) if n else np.nan,
        "sd_unalleviated_distance": float(np.std(unallev, ddof=1)) if n >= 2 else np.nan,
        "paired_mean_difference": diff_mean,
        "paired_difference_definition": "unalleviated - alleviated",
        "paired_difference_sd": diff_sd,
        "paired_difference_sem": diff_sem,
        "paired_difference_ci95_low": ci_low,
        "paired_difference_ci95_high": ci_high,
        "paired_t_statistic": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "paired_t_df": int(n - 1) if n >= 2 else np.nan,
        "paired_t_p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "cohens_dz": cohens_dz,
    }
    return pd.DataFrame([row])


def plot_paired_alleviated_distances(
    paired_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: str,
) -> None:
    """Nature-style paired-distance plot for the general TCM clinical analysis."""
    allev = paired_df["alleviated_mean_distance"].astype(float).to_numpy()
    unallev = paired_df["unalleviated_mean_distance"].astype(float).to_numpy()
    summary = summary_df.iloc[0]

    fig, ax = plt.subplots(figsize=(2.35, 3.15))

    violins = ax.violinplot(
        [allev, unallev],
        positions=[0, 1],
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method="scott",
    )
    for body, color in zip(
        violins["bodies"],
        [ALLEVIATED_DISTANCE_COLOR, UNALLEVIATED_DISTANCE_COLOR],
    ):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.9)
        body.set_alpha(1.0)

    box = ax.boxplot(
        [allev, unallev],
        positions=[0, 1],
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.9},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_zorder(4)

    data_min = float(np.nanmin(np.concatenate([allev, unallev])))
    data_max = float(np.nanmax(np.concatenate([allev, unallev])))
    data_range = max(data_max - data_min, 0.1)
    bracket_y = data_max + 0.10 * data_range
    bracket_h = 0.025 * data_range

    ax.plot(
        [0, 0, 1, 1],
        [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
        color="black",
        linewidth=0.9,
        clip_on=False,
    )
    ax.text(
        0.5,
        bracket_y + bracket_h + 0.01 * data_range,
        p_to_stars(float(summary["paired_t_p_value"])),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    annotation = (
        f"n = {int(summary['n_paired_cases'])}\n"
        f"Δ(U−A) = {summary['paired_mean_difference']:.3f}\n"
        f"95% CI [{summary['paired_difference_ci95_low']:.3f}, "
        f"{summary['paired_difference_ci95_high']:.3f}]\n"
        f"{format_p_value(float(summary['paired_t_p_value']))}"
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A", "U"], fontsize=8)
    ax.set_ylabel("Embedding distance")
    ax.set_xlabel("")
    ax.set_title("(general)", pad=5, fontsize=9)
    ax.set_xlim(-0.55, 1.55)
    lower = 0.0 if data_min >= 0 else data_min - 0.05 * data_range
    upper = bracket_y + bracket_h + 0.16 * data_range
    ax.set_ylim(lower, upper)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(False)

    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.29, top=0.92)
    fig.text(
        0.62,
        0.025,
        annotation,
        ha="center",
        va="bottom",
        fontsize=6.4,
        linespacing=1.15,
    )
    save_figure(fig, out_path)
    plt.close(fig)


def run_formula_to_individual_symptom_relationships(
    clinical_df: pd.DataFrame,
    symptom2id: Dict[str, int],
    sources: Dict[str, Dict],
    out_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    original = sources["Original"]
    paired = build_paired_alleviated_distance_table(
        clinical_df,
        symptom2id,
        original["formula_embeddings"],
        original["symptom_embeddings"],
    )
    if len(paired) < 2:
        raise ValueError("Fewer than two paired clinical cases are available for the paired-distance analysis.")

    summary = summarize_paired_distances(paired)
    paired.to_csv(
        os.path.join(out_dir, "general_tcm_paired_case_distances.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    summary.to_csv(
        os.path.join(out_dir, "general_tcm_paired_distance_summary.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    plot_paired_alleviated_distances(
        paired,
        summary,
        os.path.join(out_dir, "general_tcm_paired_distance.png"),
    )
    return paired, summary


# -----------------------------------------------------------------------------
# Optional formula-symptom ranking diagnostics
# -----------------------------------------------------------------------------

def build_candidate_symptom_table(
    clinical_df: pd.DataFrame,
    symptom2id: Dict[str, int],
    symptom_freq: np.ndarray,
    symptom_logfreq: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for case_row_idx, row in clinical_df.iterrows():
        initial = [
            symptom
            for symptom in unique_list(row["Initial_symptoms"])
            if symptom in symptom2id
        ]
        alleviated_set = set(row["Alleviated_symptoms"])
        positive = [symptom for symptom in initial if symptom in alleviated_set]
        negative = [symptom for symptom in initial if symptom not in alleviated_set]
        if len(positive) == 0 or len(negative) == 0:
            continue

        for symptom in initial:
            sid = int(symptom2id[symptom])
            rows.append({
                "case_row_idx": int(case_row_idx),
                "case_id": row["Cases_id"],
                "symptom": symptom,
                "symptom_id": sid,
                "label": int(symptom in alleviated_set),
                "train_freq": float(symptom_freq[sid]),
                "log1p_train_freq": float(symptom_logfreq[sid]),
                "n_initial_symptoms": int(len(initial)),
                "n_alleviated_symptoms": int(len(positive)),
                "n_unalleviated_symptoms": int(len(negative)),
            })
    return pd.DataFrame(rows)


def compute_symptom_formula_scores(
    candidate_df: pd.DataFrame,
    sources: Dict[str, Dict],
    metric: str,
) -> pd.DataFrame:
    scored = candidate_df.copy()
    for source_name in SOURCE_ORDER:
        source = sources[source_name]
        scores = np.full(len(scored), np.nan, dtype=float)
        for case_row_idx, idx in scored.groupby("case_row_idx").groups.items():
            rows = scored.loc[idx]
            symptom_ids = rows["symptom_id"].to_numpy(dtype=int)
            formula = source["formula_embeddings"][int(case_row_idx)]
            symptoms = source["symptom_embeddings"][symptom_ids]
            scores[scored.index.get_indexer(idx)] = score_formula_to_symptoms(
                formula,
                symptoms,
                metric,
            )
        scored[f"{source_name}_score"] = scores
    return scored


def build_pos_neg_pair_table(scored_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, cdf in scored_df.groupby("case_id", sort=False):
        pos_df = cdf[cdf["label"] == 1]
        neg_df = cdf[cdf["label"] == 0]
        for _, pos in pos_df.iterrows():
            for _, neg in neg_df.iterrows():
                rec = {
                    "case_id": case_id,
                    "case_row_idx": int(pos["case_row_idx"]),
                    "pos_symptom_id": int(pos["symptom_id"]),
                    "neg_symptom_id": int(neg["symptom_id"]),
                    "pos_symptom": pos["symptom"],
                    "neg_symptom": neg["symptom"],
                    "pos_freq": float(pos["train_freq"]),
                    "neg_freq": float(neg["train_freq"]),
                    "pos_logfreq": float(pos["log1p_train_freq"]),
                    "neg_logfreq": float(neg["log1p_train_freq"]),
                }
                rec["abs_logfreq_diff"] = abs(rec["pos_logfreq"] - rec["neg_logfreq"])
                for model in MODEL_ORDER:
                    rec[f"{model}_correct"] = pair_correct(
                        float(pos[f"{model}_score"]),
                        float(neg[f"{model}_score"]),
                    )
                rows.append(rec)
    return pd.DataFrame(rows)


def summarize_overall_case_balanced_pairwise_auc(
    pair_table: pd.DataFrame,
    out_dir: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    ensure_dir(out_dir)
    case_rows = []
    summary_rows = []
    for model in MODEL_ORDER:
        case_auc = (
            pair_table.groupby("case_id")[f"{model}_correct"]
            .mean()
            .reset_index(name="case_pairwise_auc")
        )
        case_auc["Model"] = model
        case_rows.append(case_auc)
        mean, lo, hi = bootstrap_mean_ci(
            case_auc["case_pairwise_auc"].to_numpy(),
            n_boot,
            seed,
        )
        summary_rows.append({
            "Model": model,
            "N cases": int(case_auc["case_id"].nunique()),
            "Pairwise AUC": mean,
            "Pairwise AUC 95% CI low": lo,
            "Pairwise AUC 95% CI high": hi,
        })

    case_metrics = pd.concat(case_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    case_metrics.to_csv(
        os.path.join(out_dir, "overall_case_balanced_pairwise_auc_case_metrics.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    summary.to_csv(
        os.path.join(out_dir, "overall_case_balanced_pairwise_auc_summary.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    plot_overall_case_balanced_pairwise_auc(summary, os.path.join(out_dir, "overall_case_balanced_pairwise_auc.png"))
    return summary


def plot_overall_case_balanced_pairwise_auc(summary: pd.DataFrame, out_path: str) -> None:
    summary = summary.copy()
    summary["Model"] = pd.Categorical(summary["Model"], MODEL_ORDER, ordered=True)
    summary = summary.sort_values("Model")
    x = np.arange(len(summary))
    y = summary["Pairwise AUC"].to_numpy(dtype=float)
    lo = summary["Pairwise AUC 95% CI low"].to_numpy(dtype=float)
    hi = summary["Pairwise AUC 95% CI high"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.bar(
        x,
        y,
        width=0.58,
        yerr=np.vstack([y - lo, hi - y]),
        capsize=2.5,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
        color=[MODEL_COLORS[str(model)] for model in summary["Model"]],
        edgecolor="none",
    )
    ax.axhline(0.5, linestyle="--", linewidth=0.8, color="0.45")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[str(model)] for model in summary["Model"]],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Case-wise pairwise AUC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def build_frequency_matched_pairs(scored_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, cdf in scored_df.groupby("case_id", sort=False):
        pos_df = cdf[cdf["label"] == 1]
        neg_df = cdf[cdf["label"] == 0]
        if len(pos_df) == 0 or len(neg_df) == 0:
            continue
        neg_logfreq = neg_df["log1p_train_freq"].to_numpy(dtype=float)
        for match_i, (_, pos) in enumerate(pos_df.iterrows()):
            best_j = int(
                np.argmin(np.abs(neg_logfreq - float(pos["log1p_train_freq"])))
            )
            neg = neg_df.iloc[best_j]
            rec = {
                "case_id": case_id,
                "case_row_idx": int(pos["case_row_idx"]),
                "match_id": f"{case_id}_{match_i}",
                "pos_symptom": pos["symptom"],
                "neg_symptom": neg["symptom"],
                "pos_freq": float(pos["train_freq"]),
                "neg_freq": float(neg["train_freq"]),
                "pos_logfreq": float(pos["log1p_train_freq"]),
                "neg_logfreq": float(neg["log1p_train_freq"]),
            }
            rec["abs_logfreq_diff"] = abs(rec["pos_logfreq"] - rec["neg_logfreq"])
            for model in MODEL_ORDER:
                rec[f"{model}_correct"] = pair_correct(
                    float(pos[f"{model}_score"]),
                    float(neg[f"{model}_score"]),
                )
            rows.append(rec)
    return pd.DataFrame(rows)


def parse_calipers(value: str) -> List[Optional[float]]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if token.lower() in {"none", "na", ""}:
            out.append(None)
        else:
            out.append(float(token))
    return out


def run_frequency_matched_pairwise_ranking(
    matched_pairs: pd.DataFrame,
    out_dir: str,
    calipers: List[Optional[float]],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    ensure_dir(out_dir)
    matched_pairs.to_csv(
        os.path.join(out_dir, "frequency_matched_pairs.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    rows = []
    for caliper in calipers:
        if caliper is None:
            sub = matched_pairs
            label = "none"
        else:
            sub = matched_pairs[
                matched_pairs["abs_logfreq_diff"] <= float(caliper)
            ]
            label = f"≤{caliper:g}"
        if len(sub) == 0:
            continue

        for model in MODEL_ORDER:
            case_values = sub.groupby("case_id")[f"{model}_correct"].mean()
            mean, lo, hi = bootstrap_mean_ci(case_values.to_numpy(), n_boot, seed)
            rows.append({
                "Caliper": label,
                "Model": model,
                "N matched pairs": int(len(sub)),
                "N cases": int(sub["case_id"].nunique()),
                "Case mean matched-pair accuracy": mean,
                "Accuracy 95% CI low": lo,
                "Accuracy 95% CI high": hi,
                "Mean abs log-frequency diff": float(sub["abs_logfreq_diff"].mean()),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(
        os.path.join(out_dir, "frequency_matched_pairwise_ranking_summary.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    plot_frequency_matched_pairwise_ranking(summary, os.path.join(out_dir, "frequency_matched_pairwise_ranking.png"))
    return summary


def plot_frequency_matched_pairwise_ranking(summary: pd.DataFrame, out_path: str) -> None:
    caliper_order = summary["Caliper"].drop_duplicates().tolist()
    x = np.arange(len(caliper_order))
    width = 0.18
    offsets = (np.arange(len(MODEL_ORDER)) - 1) * width
    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    for offset, model in zip(offsets, MODEL_ORDER):
        sub = summary[summary["Model"] == model].set_index("Caliper").reindex(caliper_order)
        y = sub["Case mean matched-pair accuracy"].to_numpy(dtype=float)
        lo = sub["Accuracy 95% CI low"].to_numpy(dtype=float)
        hi = sub["Accuracy 95% CI high"].to_numpy(dtype=float)
        ax.bar(
            x + offset,
            y,
            width=width,
            yerr=np.vstack([y - lo, hi - y]),
            capsize=2.0,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            edgecolor="none",
        )

    ax.axhline(0.5, linestyle="--", linewidth=0.8, color="0.45")
    ax.set_xticks(x)
    ax.set_xticklabels(caliper_order)
    ax.set_ylabel("Case mean pair accuracy")
    ax.set_xlabel("Frequency caliper")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.32, top=0.96)
    save_figure(fig, out_path)
    plt.close(fig)


def run_individual_symptom_auc_by_frequency(
    scored_df: pd.DataFrame,
    out_dir: str,
    min_pos: int,
    min_neg: int,
) -> pd.DataFrame:
    ensure_dir(out_dir)
    df = scored_df.copy()

    # Relative proximity is standardized within each clinical case before the
    # same symptom is compared across cases.
    for model in ["Original", "SVD baseline"]:
        relative_col = f"{model}_relative_proximity"
        df[relative_col] = np.nan
        for _, idx in df.groupby("case_id").groups.items():
            df.loc[idx, relative_col] = zscore_within_case(
                df.loc[idx, f"{model}_score"].to_numpy(dtype=float)
            )

    rows = []
    for symptom_id, sdf in df.groupby("symptom_id"):
        n_pos = int(sdf["label"].sum())
        n_neg = int((1 - sdf["label"]).sum())
        eligible = n_pos >= min_pos and n_neg >= min_neg
        rec = {
            "symptom_id": int(symptom_id),
            "symptom": sdf["symptom"].iloc[0],
            "train_freq": float(sdf["train_freq"].iloc[0]),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "eligible": bool(eligible),
            "TCM-ES_AUC": auc_rank(
                sdf["label"].to_numpy(),
                sdf["Original_relative_proximity"].to_numpy(),
            ),
            "SVD_AUC": auc_rank(
                sdf["label"].to_numpy(),
                sdf["SVD baseline_relative_proximity"].to_numpy(),
            ),
        }
        rec["TCM-ES_minus_SVD_AUC"] = rec["TCM-ES_AUC"] - rec["SVD_AUC"]
        rows.append(rec)

    auc_df = pd.DataFrame(rows)
    eligible = auc_df[auc_df["eligible"]].copy()
    if len(eligible) > 0:
        labels = [f"B{i}" for i in range(1, 9)]
        eligible["frequency_octile"] = pd.qcut(
            eligible["train_freq"].rank(method="first"),
            8,
            labels=labels,
        )
        auc_df = auc_df.merge(
            eligible[["symptom_id", "frequency_octile"]],
            on="symptom_id",
            how="left",
        )
        plot_individual_symptom_auc_by_frequency(
            eligible,
            os.path.join(out_dir, "individual_symptom_auc_by_frequency.png"),
        )

    auc_df.to_csv(
        os.path.join(out_dir, "individual_symptom_auc.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    return auc_df


def plot_individual_symptom_auc_by_frequency(eligible: pd.DataFrame, out_path: str) -> None:
    octile_order = [f"B{i}" for i in range(1, 9)]
    data = [
        eligible.loc[
            eligible["frequency_octile"].astype(str) == label,
            "TCM-ES_minus_SVD_AUC",
        ].dropna().to_numpy()
        for label in octile_order
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    bp = ax.boxplot(
        data,
        positions=np.arange(8),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for box in bp["boxes"]:
        box.set_facecolor(MODEL_COLORS["Original"])
        box.set_alpha(0.45)
        box.set_edgecolor("black")

    rng = np.random.default_rng(42)
    for i, values in enumerate(data):
        if len(values) == 0:
            continue
        jitter = rng.normal(0, 0.055, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=9,
            color=MODEL_COLORS["Original"],
            alpha=0.55,
            edgecolors="none",
        )
    ax.axhline(0, linestyle="--", linewidth=0.8, color="0.45")
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(octile_order)
    ax.set_ylabel("AUC difference\nTCM-ES − SVD")
    ax.set_xlabel("Symptom frequency octile")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def run_formula_symptom_ranking_diagnostics(
    clinical_df: pd.DataFrame,
    symptom2id: Dict[str, int],
    symptom_freq: np.ndarray,
    symptom_logfreq: np.ndarray,
    sources: Dict[str, Dict],
    out_dir: str,
    metric: str,
    n_boot: int,
    calipers: List[Optional[float]],
    min_pos: int,
    min_neg: int,
    seed: int,
) -> None:
    ensure_dir(out_dir)
    candidate_df = build_candidate_symptom_table(
        clinical_df,
        symptom2id,
        symptom_freq,
        symptom_logfreq,
    )
    candidate_df.to_csv(
        os.path.join(out_dir, "candidate_symptoms.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    scored_df = compute_symptom_formula_scores(candidate_df, sources, metric)
    scored_df.to_csv(
        os.path.join(out_dir, "formula_symptom_scores.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    pair_table = build_pos_neg_pair_table(scored_df)
    pair_table.to_csv(
        os.path.join(out_dir, "positive_negative_pairs.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    summarize_overall_case_balanced_pairwise_auc(
        pair_table,
        os.path.join(out_dir, "overall_case_balanced_pairwise_auc"),
        n_boot,
        seed,
    )
    matched = build_frequency_matched_pairs(scored_df)
    run_frequency_matched_pairwise_ranking(
        matched,
        os.path.join(out_dir, "frequency_matched_pairwise_ranking"),
        calipers,
        n_boot,
        seed,
    )
    run_individual_symptom_auc_by_frequency(
        scored_df,
        os.path.join(out_dir, "individual_symptom_auc_by_frequency"),
        min_pos,
        min_neg,
    )


# -----------------------------------------------------------------------------
# Bidirectional complete-set retrieval
# -----------------------------------------------------------------------------

def _summarize_retrieval_repeats(result: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    repeat_summary = (
        result.groupby(["Source", "repeat"])
        .agg(
            n_cases=("case_id", "nunique"),
            n_candidates=("n_candidates", "mean"),
            hit_at_1=("hit_at_1", "mean"),
            hit_at_5=("hit_at_5", "mean"),
            hit_at_10=("hit_at_10", "mean"),
        )
        .reset_index()
    )

    rows = []
    for source_name, sdf in repeat_summary.groupby("Source", sort=False):
        rec = {
            "Source": source_name,
            "n_repeats": int(sdf["repeat"].nunique()),
            "n_cases": float(sdf["n_cases"].mean()),
            "n_candidates": float(sdf["n_candidates"].mean()),
        }
        for metric in ["hit_at_1", "hit_at_5", "hit_at_10"]:
            values = sdf[metric].to_numpy(dtype=float)
            sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            sem = sd / np.sqrt(len(values)) if len(values) > 1 else np.nan
            rec[metric] = float(values.mean())
            rec[f"{metric}_sd"] = sd
            rec[f"{metric}_sem"] = sem
        rows.append(rec)
    return repeat_summary, pd.DataFrame(rows)


def _plot_retrieval_summary(
    summary: pd.DataFrame,
    title: str,
    out_path: str,
) -> None:
    metrics = ["hit_at_1", "hit_at_5", "hit_at_10"]
    sem_cols = ["hit_at_1_sem", "hit_at_5_sem", "hit_at_10_sem"]
    labels = ["Hit@1", "Hit@5", "Hit@10"]
    summary = summary.copy()
    summary["Source"] = pd.Categorical(
        summary["Source"],
        SET_SOURCE_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("Source")

    x = np.arange(3)
    width = 0.80 / max(len(summary), 1)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for i, (_, row) in enumerate(summary.iterrows()):
        source = str(row["Source"])
        y = row[metrics].to_numpy(dtype=float)
        yerr = row[sem_cols].to_numpy(dtype=float)
        ax.bar(
            x + (i - (len(summary) - 1) / 2) * width,
            y,
            width=width,
            yerr=yerr,
            capsize=2.0,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
            label=SET_SOURCE_LABELS[source],
            color=SET_SOURCE_COLORS[source],
            edgecolor="none",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Top-k hit rate")
    ax.set_title(title, pad=6)
    ax.legend(frameon=False, ncol=1, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _sample_single_target_candidate_sets(
    n: int,
    n_candidates: int,
    n_repeats: int,
    seed: int,
) -> List[List[np.ndarray]]:
    candidate_sets = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        repeat_sets = []
        for i in range(n):
            negatives = np.array([j for j in range(n) if j != i], dtype=int)
            n_neg = min(n_candidates - 1, len(negatives))
            sampled = rng.choice(negatives, size=n_neg, replace=False)
            repeat_sets.append(np.concatenate([[i], sampled]))
        candidate_sets.append(repeat_sets)
    return candidate_sets


def run_formula_to_alleviated_symptom_pattern_retrieval(
    clinical_df: pd.DataFrame,
    symptom2id: Dict[str, int],
    sources: Dict[str, Dict],
    out_dir: str,
    metric: str,
    n_candidates: int,
    n_repeats: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    rows = []
    for case_row_idx, row in clinical_df.iterrows():
        initial = [s for s in unique_list(row["Initial_symptoms"]) if s in symptom2id]
        allev_set = set(row["Alleviated_symptoms"])
        allev = [s for s in initial if s in allev_set]
        if len(allev) == 0:
            continue
        rows.append({
            "case_row_idx": int(case_row_idx),
            "case_id": row["Cases_id"],
            "alleviated_symptom_ids": [int(symptom2id[s]) for s in allev],
            "n_alleviated_symptoms": int(len(allev)),
        })
    retrieval_df = pd.DataFrame(rows).reset_index(drop=True)
    retrieval_df.to_csv(
        os.path.join(out_dir, "formula_to_alleviated_symptom_pattern_candidates.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    if len(retrieval_df) < 2:
        raise ValueError("Insufficient cases for formula-to-alleviated retrieval.")

    idx = retrieval_df["case_row_idx"].to_numpy(dtype=int)
    score_cache = {}
    for source_name in SOURCE_ORDER:
        source = sources[source_name]
        formula = source["formula_embeddings"][idx]
        if source_name == "Original":
            target = source["improved_symptom_set_embeddings"][idx]
            result_name = "Original_encoder_set"
        else:
            target = mean_pool_symptom_set_embeddings(
                retrieval_df["alleviated_symptom_ids"].tolist(),
                source["symptom_embeddings"],
            )
            result_name = source_name
        score_cache[result_name] = score_query_to_candidates(formula, target, metric)

    candidate_sets = _sample_single_target_candidate_sets(
        len(retrieval_df),
        n_candidates,
        n_repeats,
        seed,
    )
    result_rows = []
    for source_name, score_matrix in score_cache.items():
        for repeat in range(n_repeats):
            for i in range(len(retrieval_df)):
                cand = candidate_sets[repeat][i]
                ranked = cand[np.argsort(-score_matrix[i, cand])]
                rank = int(np.where(ranked == i)[0][0]) + 1
                result_rows.append({
                    "Source": source_name,
                    "repeat": int(repeat),
                    "case_id": retrieval_df["case_id"].iloc[i],
                    "case_row_idx": int(retrieval_df["case_row_idx"].iloc[i]),
                    "rank": rank,
                    "n_candidates": int(len(cand)),
                    "hit_at_1": int(rank <= 1),
                    "hit_at_5": int(rank <= 5),
                    "hit_at_10": int(rank <= 10),
                })
    result = pd.DataFrame(result_rows)
    repeat_summary, summary = _summarize_retrieval_repeats(result)
    result.to_csv(
        os.path.join(out_dir, "formula_to_alleviated_symptom_pattern_case_metrics.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    repeat_summary.to_csv(
        os.path.join(out_dir, "formula_to_alleviated_symptom_pattern_repeat_summary.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    summary.to_csv(
        os.path.join(out_dir, "formula_to_alleviated_symptom_pattern_summary.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    _plot_retrieval_summary(
        summary,
        "Formula to alleviated symptom-pattern",
        os.path.join(out_dir, "formula_to_alleviated_symptom_pattern.png"),
    )
    return result, summary


def run_initial_symptom_pattern_to_formula_retrieval(
    clinical_df: pd.DataFrame,
    symptom2id: Dict[str, int],
    sources: Dict[str, Dict],
    out_dir: str,
    metric: str,
    n_candidates: int,
    n_repeats: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    rows = []
    for case_row_idx, row in clinical_df.iterrows():
        initial = [s for s in unique_list(row["Initial_symptoms"]) if s in symptom2id]
        formula = unique_list(row["Formula"])
        if len(initial) == 0 or len(formula) == 0:
            continue
        rows.append({
            "case_row_idx": int(case_row_idx),
            "case_id": row["Cases_id"],
            "formula_key": make_formula_key(formula),
        })
    retrieval_df = pd.DataFrame(rows).reset_index(drop=True)
    retrieval_df.to_csv(
        os.path.join(out_dir, "initial_symptom_pattern_to_formula_candidates.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    if len(retrieval_df) < 2:
        raise ValueError("Insufficient cases for initial-to-formula retrieval.")

    idx = retrieval_df["case_row_idx"].to_numpy(dtype=int)
    key_to_indices = {}
    for i, key in enumerate(retrieval_df["formula_key"].astype(str)):
        key_to_indices.setdefault(key, []).append(i)
    positive_sets = [
        np.asarray(key_to_indices[key], dtype=int)
        for key in retrieval_df["formula_key"].astype(str)
    ]

    score_cache = {}
    for source_name in SOURCE_ORDER:
        source = sources[source_name]
        query = source["case_symptom_embeddings"][idx]
        candidates = source["formula_embeddings"][idx]
        result_name = "Original_encoder_set" if source_name == "Original" else source_name
        score_cache[result_name] = score_query_to_candidates(query, candidates, metric)

    candidate_sets = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        repeat_sets = []
        for i in range(len(retrieval_df)):
            positives = np.unique(positive_sets[i])
            positive_set = set(int(x) for x in positives)
            negatives = np.asarray(
                [j for j in range(len(retrieval_df)) if j not in positive_set],
                dtype=int,
            )
            # The pool size is kept at n_candidates whenever enough negatives exist.
            # A single observed target row is included; other identical-formula rows
            # count as correct if they appear among sampled candidates.
            target = np.asarray([i], dtype=int)
            n_neg = min(n_candidates - 1, len(negatives))
            sampled_neg = rng.choice(negatives, size=n_neg, replace=False)
            repeat_sets.append(np.concatenate([target, sampled_neg]))
        candidate_sets.append(repeat_sets)

    result_rows = []
    for source_name, score_matrix in score_cache.items():
        for repeat in range(n_repeats):
            for i in range(len(retrieval_df)):
                cand = candidate_sets[repeat][i]
                positives = set(int(x) for x in positive_sets[i])
                ranked = cand[np.argsort(-score_matrix[i, cand])]
                rank = next(
                    r for r, j in enumerate(ranked, start=1) if int(j) in positives
                )
                result_rows.append({
                    "Source": source_name,
                    "repeat": int(repeat),
                    "case_id": retrieval_df["case_id"].iloc[i],
                    "case_row_idx": int(retrieval_df["case_row_idx"].iloc[i]),
                    "rank": int(rank),
                    "n_candidates": int(len(cand)),
                    "hit_at_1": int(rank <= 1),
                    "hit_at_5": int(rank <= 5),
                    "hit_at_10": int(rank <= 10),
                })

    result = pd.DataFrame(result_rows)
    repeat_summary, summary = _summarize_retrieval_repeats(result)
    result.to_csv(
        os.path.join(out_dir, "initial_symptom_pattern_to_formula_case_metrics.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    repeat_summary.to_csv(
        os.path.join(out_dir, "initial_symptom_pattern_to_formula_repeat_summary.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    summary.to_csv(
        os.path.join(out_dir, "initial_symptom_pattern_to_formula_summary.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    _plot_retrieval_summary(
        summary,
        "Initial symptom-pattern to formula",
        os.path.join(out_dir, "initial_symptom_pattern_to_formula.png"),
    )
    return result, summary


def run_bidirectional_complete_set_retrieval(
    clinical_df: pd.DataFrame,
    symptom2id: Dict[str, int],
    sources: Dict[str, Dict],
    out_dir: str,
    metric: str,
    n_candidates: int,
    n_repeats: int,
    seed: int,
) -> None:
    ensure_dir(out_dir)
    run_initial_symptom_pattern_to_formula_retrieval(
        clinical_df,
        symptom2id,
        sources,
        os.path.join(out_dir, "initial_symptom_pattern_to_formula"),
        metric,
        n_candidates,
        n_repeats,
        seed,
    )
    run_formula_to_alleviated_symptom_pattern_retrieval(
        clinical_df,
        symptom2id,
        sources,
        os.path.join(out_dir, "formula_to_alleviated_symptom_pattern"),
        metric,
        n_candidates,
        n_repeats,
        seed,
    )


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "General TCM clinical distance-based analyses. By default, run the main "
            "formula-to-individual-symptom and bidirectional complete-set retrieval analyses; "
            "formula-symptom ranking diagnostics are optional."
        )
    )
    parser.add_argument("--clinical-data", required=True)
    parser.add_argument("--symptom-list", required=True)
    parser.add_argument("--tcm-clinical-emb-dir", required=True)
    parser.add_argument("--tcm-emb-dir", required=True)
    parser.add_argument("--cooc-svd-emb-dir", required=True)
    parser.add_argument("--cooc-graph-emb-dir", required=True)
    parser.add_argument("--clinical-case-subdir", default="general_clinical_cases")
    parser.add_argument(
        "--symptom-frequency-file",
        default=None,
        help=(
            "Training-set symptom-frequency CSV used only when "
            "--run-ranking-diagnostics is enabled."
        ),
    )
    parser.add_argument("--out-dir", required=True)

    parser.add_argument(
        "--symptom-metric",
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Formula-to-individual-symptom metric used by optional ranking diagnostics.",
    )
    parser.add_argument(
        "--set-metric",
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Metric for bidirectional complete-set retrieval.",
    )
    parser.add_argument("--n-candidates", type=int, default=50)
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--calipers", default="none,1.0,0.75,0.5")
    parser.add_argument("--individual-min-pos", type=int, default=5)
    parser.add_argument("--individual-min-neg", type=int, default=5)
    parser.add_argument(
        "--run-ranking-diagnostics",
        action="store_true",
        help=(
            "Run optional formula-symptom ranking and symptom-frequency diagnostics. "
            "These analyses require --symptom-frequency-file."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None) -> None:
    set_publication_style()
    args = build_parser().parse_args(argv)
    ensure_dir(args.out_dir)

    if args.n_candidates < 2:
        raise ValueError("--n-candidates must be >= 2")
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be >= 1")

    if args.run_ranking_diagnostics and not args.symptom_frequency_file:
        raise ValueError(
            "--symptom-frequency-file is required when --run-ranking-diagnostics is enabled."
        )

    clinical_df = load_clinical_cases(args.clinical_data)
    symptom_list, symptom2id = load_standard_terms(args.symptom_list)
    sources = load_embedding_sources(args, clinical_df)

    run_formula_to_individual_symptom_relationships(
        clinical_df,
        symptom2id,
        sources,
        os.path.join(args.out_dir, "formula_to_individual_symptom_relationships"),
    )

    run_bidirectional_complete_set_retrieval(
        clinical_df,
        symptom2id,
        sources,
        os.path.join(args.out_dir, "bidirectional_complete_set_retrieval"),
        args.set_metric,
        args.n_candidates,
        args.n_repeats,
        args.seed,
    )

    if args.run_ranking_diagnostics:
        symptom_freq, symptom_logfreq = load_symptom_frequency(
            args.symptom_frequency_file,
            symptom_list,
        )
        run_formula_symptom_ranking_diagnostics(
            clinical_df,
            symptom2id,
            symptom_freq,
            symptom_logfreq,
            sources,
            os.path.join(args.out_dir, "formula_symptom_ranking_diagnostics"),
            args.symptom_metric,
            args.n_bootstrap,
            parse_calipers(args.calipers),
            args.individual_min_pos,
            args.individual_min_neg,
            args.seed,
        )

    print("Done.")
    print("Clinical data:", args.clinical_data)
    print("Results saved to:", args.out_dir)


if __name__ == "__main__":
    main()
