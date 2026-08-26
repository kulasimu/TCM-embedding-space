#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PCA projection of TCM-ES embeddings.

The script loads record-level and individual symptom/herb embeddings from a
specified embedding directory. A PCA model can either be fitted from the
selected embedding set or loaded from a previously saved model for direct
projection.

Two PCA component settings are used:
    - variance components: number of components used to summarize cumulative
      explained variance when fitting a new PCA;
    - PCA components: number of leading components retained for visualization
      and downstream analysis.
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def load_pickle(path: str):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def terms_to_ids(term_list: Sequence[str], term2id: Dict[str, int]) -> List[int]:
    if not isinstance(term_list, list):
        return []
    return [term2id[x] for x in term_list if x in term2id]


def count_term_frequency(
    id_lists: Sequence[Sequence[int]],
    vocab_size: int,
) -> np.ndarray:
    """Count the number of selected records containing each term."""
    freq = np.zeros(vocab_size, dtype=int)
    for ids in id_lists:
        for idx in set(ids):
            freq[idx] += 1
    return freq


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit or apply PCA to TCM-ES and related embedding sets."
    )

    parser.add_argument(
        "--formula-data",
        default="data/TCM_formulas/TCM_formula_data_example_2000.pkl",
        help="Formula-record dataframe aligned with the embedding files.",
    )

    parser.add_argument(
        "--embedding-dir",
        default="results/embeddings/TCM_embeddings",
        help=(
            "Directory containing selected formula-record embeddings and "
            "individual symptom/herb embeddings."
        ),
    )

    parser.add_argument(
        "--symptom-list",
        default="core/standard_TCM_entities/symptom_list.pkl",
        help="Standardized symptom vocabulary.",
    )

    parser.add_argument(
        "--herb-list",
        default="core/standard_TCM_entities/herb_list.pkl",
        help="Standardized herb vocabulary.",
    )

    parser.add_argument(
        "--pca-embedding-type",
        choices=[
            "average",
            "record_symptom",
            "record_herb",
            "individual_herb",
            "individual_symptom",
        ],
        default="average",
        help=(
            "Embedding set used to fit a new PCA. 'average' uses the mean of "
            "matched symptom-pattern and formula embeddings."
        ),
    )

    parser.add_argument(
        "--variance-components",
        type=int,
        default=200,
        help=(
            "Number of components used for cumulative explained-variance "
            "analysis when fitting a new PCA."
        ),
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=6,
        help=(
            "Number of leading principal components retained for projection, "
            "visualization, and downstream analysis."
        ),
    )

    parser.add_argument(
        "--pca-model",
        default=None,
        help=(
            "Optional fitted PCA model. If provided, the model is loaded and "
            "only transform() is applied; no PCA is fitted."
        ),
    )

    parser.add_argument(
        "--pca-scale",
        default=None,
        help=(
            "Optional saved PCA visualization scale (.npy). If omitted, a new "
            "scale is calculated from the projected embeddings."
        ),
    )

    parser.add_argument(
        "--out-dir",
        default="results/Embedding_space_analysis",
        help="Output directory.",
    )

    args = parser.parse_args(argv)

    if args.variance_components < 1:
        parser.error("--variance-components must be >= 1")
    if args.pca_components < 1:
        parser.error("--pca-components must be >= 1")

    return args


# ---------------------------------------------------------------------
# PCA fitting and projection
# ---------------------------------------------------------------------

def run_embedding_pca(
    fit_embeddings: np.ndarray,
    out_dir: str,
    n_components: int = 6,
):
    """Fit and save the PCA model used for TCM-ES visualization and analysis."""
    pca_dir = os.path.join(out_dir, "pca")
    ensure_dir(pca_dir)

    fit_embeddings = np.asarray(fit_embeddings)
    max_components = min(fit_embeddings.shape[0], fit_embeddings.shape[1])
    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} is too large for embedding shape "
            f"{fit_embeddings.shape}. Maximum allowed: {max_components}"
        )

    print("\n========== Fitting PCA ==========")
    print("PCA fit embeddings:", fit_embeddings.shape)
    print("PCA components:", n_components)

    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(fit_embeddings)

    pca_model_path = os.path.join(pca_dir, "TCM_embedding_pca_model.pkl")
    with open(pca_model_path, "wb") as fp:
        pickle.dump(pca, fp)

    explained_df = pd.DataFrame({
        "PC": [f"PC{i + 1}" for i in range(n_components)],
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(
            pca.explained_variance_ratio_
        ),
    })
    explained_path = os.path.join(pca_dir, "pca_explained_variance.csv")
    explained_df.to_csv(explained_path, index=False)

    print("Saved PCA model to:", pca_model_path)
    print("Saved PCA explained variance to:", explained_path)

    return pca, explained_df


def load_pca_model(path: str):
    """Load a fitted PCA model."""
    with open(path, "rb") as fp:
        pca = pickle.load(fp)

    if not hasattr(pca, "transform"):
        raise TypeError(f"File does not contain a fitted PCA-like model: {path}")

    return pca


def project_embeddings_to_pca(
    embeddings: np.ndarray,
    pca,
    n_components: Optional[int] = None,
):
    """Project embeddings using a fitted PCA model."""
    embeddings = np.asarray(embeddings)
    coords = np.asarray(pca.transform(embeddings))

    if n_components is None:
        return coords

    if n_components > coords.shape[1]:
        raise ValueError(
            f"Requested {n_components} PCA components, but the loaded model "
            f"provides only {coords.shape[1]}."
        )

    return coords[:, :n_components]


def select_pca_embeddings(
    embedding_type: str,
    record_symptom_embeddings: np.ndarray,
    record_herb_embeddings: np.ndarray,
    individual_symptom_embeddings: np.ndarray,
    individual_herb_embeddings: np.ndarray,
):
    """Select the embedding set used to fit a new PCA model."""
    if embedding_type == "average":
        return 0.5 * (record_symptom_embeddings + record_herb_embeddings)
    if embedding_type == "record_symptom":
        return record_symptom_embeddings
    if embedding_type == "record_herb":
        return record_herb_embeddings
    if embedding_type == "individual_herb":
        return individual_herb_embeddings
    if embedding_type == "individual_symptom":
        return individual_symptom_embeddings

    raise ValueError(f"Unknown PCA embedding type: {embedding_type}")


# ---------------------------------------------------------------------
# PCA explained variance and visualization scale
# ---------------------------------------------------------------------

def plot_pca_cumulative_variance(
    embeddings: np.ndarray,
    out_dir: str,
    n_components: int = 200,
    threshold: float = 0.95,
    extra_components_after_threshold: int = 3,
    prefix: str = "record_embedding_pca_cumulative_explained_variance",
    plot_title: Optional[str] = None,
):
    """Fit a PCA for cumulative explained-variance visualization."""
    pca_dir = os.path.join(out_dir, "pca")
    ensure_dir(pca_dir)

    embeddings = np.asarray(embeddings)
    max_allowed_components = min(embeddings.shape[0], embeddings.shape[1])
    n_components = min(n_components, max_allowed_components)

    print("\n========== PCA cumulative explained variance ==========")
    print("Variance-analysis embeddings:", embeddings.shape)
    print("Variance-analysis components:", n_components)

    pca_tmp = PCA(n_components=n_components, svd_solver="full")
    pca_tmp.fit(embeddings)

    explained_ratio = pca_tmp.explained_variance_ratio_
    cumulative_ratio = np.cumsum(explained_ratio)

    reached = np.where(cumulative_ratio >= threshold)[0]
    if len(reached) > 0:
        threshold_pc = int(reached[0]) + 1
        n_plot = min(
            threshold_pc + extra_components_after_threshold,
            n_components,
        )
    else:
        threshold_pc = None
        n_plot = n_components

    x = np.arange(1, n_plot + 1)
    indiv_y = explained_ratio[:n_plot]
    cum_y = cumulative_ratio[:n_plot]

    plot_df = pd.DataFrame({
        "PC": x,
        "explained_variance_ratio": indiv_y,
        "cumulative_explained_variance_ratio": cum_y,
    })
    plot_df.to_csv(
        os.path.join(pca_dir, f"{prefix}.csv"),
        index=False,
    )

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax1 = plt.subplots(figsize=(4.6, 3.6), dpi=200)
    if plot_title is not None:
        ax1.set_title(plot_title, fontsize=11, pad=10)

    ax1.bar(
        x,
        indiv_y,
        width=0.75,
        color="#78C2BD",
        edgecolor="none",
        label="Individual",
        zorder=2,
    )
    ax1.set_xlabel("Principal components", fontsize=10)
    ax1.set_ylabel("Individual variance ratio", fontsize=10)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(axis="both", labelsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, zorder=1)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        cum_y,
        color="#2F3B52",
        marker="o",
        markersize=3.2,
        linewidth=1.5,
        label="Cumulative",
        zorder=3,
    )
    ax2.axhline(
        threshold,
        linestyle="--",
        linewidth=1.0,
        color="gray",
    )
    ax2.set_ylabel("Cumulative variance ratio", fontsize=10)
    ax2.set_ylim(min(0.0, cum_y[0] - 0.05), 1.02)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.spines["top"].set_visible(False)

    if threshold_pc is not None:
        ax2.axvline(
            threshold_pc,
            linestyle=":",
            linewidth=1.0,
            color="gray",
        )

    ax2.text(
        x[-1] - 0.4,
        threshold - 0.05,
        f"{int(threshold * 100)}%",
        ha="right",
        va="bottom",
        fontsize=10,
        color="grey",
    )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(
        h2 + h1,
        l2 + l1,
        frameon=False,
        fontsize=8.5,
        loc="center right",
    )

    plt.tight_layout()

    pdf_path = os.path.join(pca_dir, f"{prefix}.pdf")
    png_path = os.path.join(pca_dir, f"{prefix}.png")
    svg_path = os.path.join(pca_dir, f"{prefix}.svg")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved PCA cumulative variance plot to:", pdf_path)
    return plot_df


def compute_pca_scale(
    coords_list: List[np.ndarray],
    out_dir: str,
):
    """Compute the 95th-percentile coordinate scale used for visualization."""
    pca_dir = os.path.join(out_dir, "pca")
    ensure_dir(pca_dir)

    all_coords = np.vstack([np.asarray(x) for x in coords_list])
    scale = np.percentile(np.abs(all_coords), 95, axis=0)
    scale[scale == 0] = 1.0

    np.save(
        os.path.join(pca_dir, "pca_95percent_scale.npy"),
        scale,
    )
    pd.DataFrame({
        "PC": [f"PC{i + 1}" for i in range(len(scale))],
        "p95_abs_coordinate": scale,
    }).to_csv(
        os.path.join(pca_dir, "pca_95percent_scale.csv"),
        index=False,
    )

    return scale


def load_pca_scale(path: str, n_components: int) -> np.ndarray:
    scale = np.asarray(np.load(path), dtype=np.float64)
    if scale.ndim != 1:
        raise ValueError("PCA scale must be a one-dimensional array.")
    if len(scale) < n_components:
        raise ValueError(
            f"PCA scale contains {len(scale)} components, but "
            f"{n_components} are required."
        )
    return scale[:n_components]


def apply_pca_scale(
    coords: np.ndarray,
    scale: np.ndarray,
):
    """Scale PCA coordinates for visualization only."""
    return np.asarray(coords) / np.asarray(scale)


# ---------------------------------------------------------------------
# PCA output
# ---------------------------------------------------------------------

def save_pca_coordinates(
    coords: np.ndarray,
    metadata_df: pd.DataFrame,
    out_dir: str,
    name: str,
):
    """Save PCA coordinates with row-aligned metadata."""
    pca_dir = os.path.join(out_dir, "pca")
    ensure_dir(pca_dir)

    coords = np.asarray(coords)
    pc_cols = [f"PC{i + 1}" for i in range(coords.shape[1])]

    coords_df = pd.concat(
        [
            metadata_df.reset_index(drop=True).copy(),
            pd.DataFrame(coords, columns=pc_cols),
        ],
        axis=1,
    )

    csv_path = os.path.join(pca_dir, f"{name}.csv")
    coords_df.to_csv(csv_path, index=False, encoding="utf_8_sig")
    print("Saved PCA coordinates to:", csv_path)


def summarize_top_pc_entities(
    embeddings: np.ndarray,
    entity_names: List[str],
    pca,
    out_dir: str,
    entity_type: str,
    top_k: int = 50,
    n_pcs: int = 6,
    freq: Optional[np.ndarray] = None,
    min_freq: Optional[float] = None,
):
    """Save the highest-projection entities on each retained PC."""
    pca_dir = os.path.join(out_dir, "pca")
    ensure_dir(pca_dir)

    coords = project_embeddings_to_pca(
        embeddings=embeddings,
        pca=pca,
        n_components=n_pcs,
    )

    if freq is not None and min_freq is not None:
        freq = np.asarray(freq)
        candidate_ids = np.where(freq >= min_freq)[0]
    else:
        candidate_ids = np.arange(len(entity_names))

    rows = []
    for pc in range(coords.shape[1]):
        top_ids = candidate_ids[
            np.argsort(-coords[candidate_ids, pc])[:top_k]
        ]
        for rank, idx in enumerate(top_ids, start=1):
            rows.append({
                "entity_type": entity_type,
                "PC": f"PC{pc + 1}",
                "rank": rank,
                "entity_id": int(idx),
                "entity_name": entity_names[idx],
                "projection": float(coords[idx, pc]),
                "frequency": float(freq[idx]) if freq is not None else np.nan,
            })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(
        pca_dir,
        f"{entity_type}_top{top_k}_pc_entities.csv",
    )
    out_df.to_csv(out_path, index=False, encoding="utf_8_sig")
    print("Saved:", out_path)

    return out_df


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    ensure_dir(args.out_dir)

    print("Embedding directory:", args.embedding_dir)

    # -----------------------------------------------------------------
    # Load formula records and standardized vocabularies
    # -----------------------------------------------------------------
    formula_data_pd = pd.read_pickle(args.formula_data).reset_index(drop=True)
    for col in ["Symptoms", "Herbs"]:
        if col not in formula_data_pd.columns:
            raise ValueError(f"Formula dataframe must contain column: {col}")

    symptom_list = load_pickle(args.symptom_list)
    herb_list = load_pickle(args.herb_list)

    sym2id = {s: i for i, s in enumerate(symptom_list)}
    herb2id = {h: i for i, h in enumerate(herb_list)}

    # -----------------------------------------------------------------
    # Load record-level and individual embeddings
    # -----------------------------------------------------------------
    selected_formula_ids = [
        int(x)
        for x in load_pickle(
            os.path.join(
                args.embedding_dir,
                "selected_formula_ids(line_number).pkl",
            )
        )
    ]

    selected_formula_symptom_embeddings = np.asarray(
        load_pickle(
            os.path.join(
                args.embedding_dir,
                "selected_formula_symptom_embeddings.pkl",
            )
        )
    )

    selected_formula_herb_embeddings = np.asarray(
        load_pickle(
            os.path.join(
                args.embedding_dir,
                "selected_formula_herb_embeddings.pkl",
            )
        )
    )

    individual_symptom_embeddings = np.asarray(
        load_pickle(
            os.path.join(
                args.embedding_dir,
                "individual_symptom_embeddings.pkl",
            )
        )
    )

    individual_herb_embeddings = np.asarray(
        load_pickle(
            os.path.join(
                args.embedding_dir,
                "individual_herb_embeddings.pkl",
            )
        )
    )

    if not (
        len(selected_formula_ids)
        == selected_formula_symptom_embeddings.shape[0]
        == selected_formula_herb_embeddings.shape[0]
    ):
        raise ValueError(
            "selected_formula_ids and record-level embeddings are not aligned."
        )

    if individual_symptom_embeddings.shape[0] != len(symptom_list):
        raise ValueError(
            "individual_symptom_embeddings rows do not match symptom_list length."
        )

    if individual_herb_embeddings.shape[0] != len(herb_list):
        raise ValueError(
            "individual_herb_embeddings rows do not match herb_list length."
        )

    embedding_dim = selected_formula_symptom_embeddings.shape[1]
    for name, array in [
        ("selected_formula_herb_embeddings", selected_formula_herb_embeddings),
        ("individual_symptom_embeddings", individual_symptom_embeddings),
        ("individual_herb_embeddings", individual_herb_embeddings),
    ]:
        if array.shape[1] != embedding_dim:
            raise ValueError(
                f"{name} has embedding dimension {array.shape[1]}, "
                f"expected {embedding_dim}."
            )

    selected_pd = (
        formula_data_pd
        .iloc[selected_formula_ids]
        .copy()
        .reset_index(drop=True)
    )

    selected_symptom_ids = [
        terms_to_ids(x, sym2id)
        for x in selected_pd["Symptoms"].tolist()
    ]
    selected_herb_ids = [
        terms_to_ids(x, herb2id)
        for x in selected_pd["Herbs"].tolist()
    ]

    symptom_freq = count_term_frequency(
        selected_symptom_ids,
        vocab_size=len(symptom_list),
    )
    herb_freq = count_term_frequency(
        selected_herb_ids,
        vocab_size=len(herb_list),
    )

    # -----------------------------------------------------------------
    # Save selected formula data and metadata
    # -----------------------------------------------------------------
    selected_data_dir = os.path.join(args.out_dir, "selected_formula_data")
    ensure_dir(selected_data_dir)

    selected_pd.to_pickle(
        os.path.join(selected_data_dir, "selected_formula_data.pkl")
    )
    selected_pd.to_csv(
        os.path.join(selected_data_dir, "selected_formula_data.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    if "Syndromes" in selected_pd.columns:
        n_syndromes = [
            len(x) if isinstance(x, list) else 0
            for x in selected_pd["Syndromes"].tolist()
        ]
    else:
        n_syndromes = [0] * len(selected_pd)

    metadata_df = pd.DataFrame({
        "selected_formula_row_id": selected_formula_ids,
        "n_symptoms": [len(x) for x in selected_symptom_ids],
        "n_herbs": [len(x) for x in selected_herb_ids],
        "n_syndromes": n_syndromes,
    })

    if "ID" in selected_pd.columns:
        metadata_df["formula_ID"] = selected_pd["ID"].values
    if "Title" in selected_pd.columns:
        metadata_df["Title"] = selected_pd["Title"].values

    metadata_df.to_csv(
        os.path.join(args.out_dir, "analysis_record_metadata.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    individual_herb_metadata_df = pd.DataFrame({
        "entity_id": np.arange(len(herb_list)),
        "entity_name": herb_list,
        "entity_type": "herb",
    })

    individual_symptom_metadata_df = pd.DataFrame({
        "entity_id": np.arange(len(symptom_list)),
        "entity_name": symptom_list,
        "entity_type": "symptom",
    })

    # -----------------------------------------------------------------
    # Select the PCA fitting representation
    # -----------------------------------------------------------------
    fit_embeddings = select_pca_embeddings(
        embedding_type=args.pca_embedding_type,
        record_symptom_embeddings=selected_formula_symptom_embeddings,
        record_herb_embeddings=selected_formula_herb_embeddings,
        individual_symptom_embeddings=individual_symptom_embeddings,
        individual_herb_embeddings=individual_herb_embeddings,
    )

    print("PCA embedding type:", args.pca_embedding_type)
    print("PCA reference embeddings:", fit_embeddings.shape)

    # -----------------------------------------------------------------
    # Fit a new PCA or load a previously fitted PCA model
    # -----------------------------------------------------------------
    if args.pca_model is None:
        pca_type_label_map = {
            "average": "Averaged record embedding",
            "record_symptom": "Record symptom-pattern embedding",
            "record_herb": "Record formula embedding",
            "individual_herb": "Individual herb embedding",
            "individual_symptom": "Individual symptom embedding",
        }

        plot_pca_cumulative_variance(
            embeddings=fit_embeddings,
            out_dir=args.out_dir,
            n_components=args.variance_components,
            threshold=0.95,
            extra_components_after_threshold=3,
            prefix="record_embedding_pca_cumulative_explained_variance",
            plot_title=pca_type_label_map[args.pca_embedding_type],
        )

        pca, _ = run_embedding_pca(
            fit_embeddings=fit_embeddings,
            out_dir=args.out_dir,
            n_components=args.pca_components,
        )
    else:
        print("Loading existing PCA model:", args.pca_model)
        pca = load_pca_model(args.pca_model)

        available_components = getattr(
            pca,
            "n_components_",
            getattr(pca, "n_components", None),
        )
        if available_components is None:
            available_components = np.asarray(pca.components_).shape[0]

        if args.pca_components > int(available_components):
            raise ValueError(
                f"--pca-components={args.pca_components}, but the loaded PCA "
                f"model contains only {available_components} components."
            )

    # -----------------------------------------------------------------
    # Project record-level and individual embeddings into the same PCA space
    # -----------------------------------------------------------------
    formula_herb_pca = project_embeddings_to_pca(
        selected_formula_herb_embeddings,
        pca,
        n_components=args.pca_components,
    )
    formula_symptom_pca = project_embeddings_to_pca(
        selected_formula_symptom_embeddings,
        pca,
        n_components=args.pca_components,
    )
    individual_herb_pca = project_embeddings_to_pca(
        individual_herb_embeddings,
        pca,
        n_components=args.pca_components,
    )
    individual_symptom_pca = project_embeddings_to_pca(
        individual_symptom_embeddings,
        pca,
        n_components=args.pca_components,
    )

    # -----------------------------------------------------------------
    # Apply a common visualization scale
    # -----------------------------------------------------------------
    if args.pca_scale is None:
        pca_scale = compute_pca_scale(
            coords_list=[
                formula_herb_pca,
                formula_symptom_pca,
                individual_herb_pca,
                individual_symptom_pca,
            ],
            out_dir=args.out_dir,
        )
    else:
        print("Loading existing PCA visualization scale:", args.pca_scale)
        pca_scale = load_pca_scale(
            args.pca_scale,
            n_components=args.pca_components,
        )

    formula_herb_pca_scaled = apply_pca_scale(
        formula_herb_pca,
        pca_scale,
    )
    formula_symptom_pca_scaled = apply_pca_scale(
        formula_symptom_pca,
        pca_scale,
    )
    individual_herb_pca_scaled = apply_pca_scale(
        individual_herb_pca,
        pca_scale,
    )
    individual_symptom_pca_scaled = apply_pca_scale(
        individual_symptom_pca,
        pca_scale,
    )

    # -----------------------------------------------------------------
    # Save raw and visualization-scaled PCA coordinates
    # -----------------------------------------------------------------
    save_pca_coordinates(
        formula_herb_pca,
        metadata_df,
        args.out_dir,
        "formula_herb_embedding_pca_coords",
    )
    save_pca_coordinates(
        formula_symptom_pca,
        metadata_df,
        args.out_dir,
        "formula_symptom_embedding_pca_coords",
    )
    save_pca_coordinates(
        formula_herb_pca_scaled,
        metadata_df,
        args.out_dir,
        "formula_herb_embedding_pca_coords_scaled",
    )
    save_pca_coordinates(
        formula_symptom_pca_scaled,
        metadata_df,
        args.out_dir,
        "formula_symptom_embedding_pca_coords_scaled",
    )
    save_pca_coordinates(
        individual_herb_pca,
        individual_herb_metadata_df,
        args.out_dir,
        "individual_herb_embedding_pca_coords",
    )
    save_pca_coordinates(
        individual_symptom_pca,
        individual_symptom_metadata_df,
        args.out_dir,
        "individual_symptom_embedding_pca_coords",
    )
    save_pca_coordinates(
        individual_herb_pca_scaled,
        individual_herb_metadata_df,
        args.out_dir,
        "individual_herb_embedding_pca_coords_scaled",
    )
    save_pca_coordinates(
        individual_symptom_pca_scaled,
        individual_symptom_metadata_df,
        args.out_dir,
        "individual_symptom_embedding_pca_coords_scaled",
    )

    # -----------------------------------------------------------------
    # Rank individual TCM entities along the retained principal components
    # -----------------------------------------------------------------
    summarize_top_pc_entities(
        embeddings=individual_herb_embeddings,
        entity_names=herb_list,
        pca=pca,
        out_dir=args.out_dir,
        entity_type="individual_herb",
        top_k=50,
        n_pcs=args.pca_components,
        freq=herb_freq,
        min_freq=50,
    )

    summarize_top_pc_entities(
        embeddings=individual_symptom_embeddings,
        entity_names=symptom_list,
        pca=pca,
        out_dir=args.out_dir,
        entity_type="individual_symptom",
        top_k=50,
        n_pcs=args.pca_components,
        freq=symptom_freq,
        min_freq=50,
    )

    print("\nDone.")
    print("Outputs saved to:", args.out_dir)


if __name__ == "__main__":
    main()
