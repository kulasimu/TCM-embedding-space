#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Align principal-component directions across independently trained TCM-ES models.

PCA is fitted independently to formula-record embeddings from each repeat model.
For every formula record, the PCA-fitting embedding is the average of its
symptom-pattern and formula embeddings. Repeat-model PCs are matched one-to-one
to the saved main-model PCs by maximizing the summed absolute Pearson
correlations of PC-score vectors across the same formula records.
"""

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align PCA directions across independently trained TCM-ES models "
            "using the same formula records."
        )
    )

    parser.add_argument(
        "--main-pca",
        required=True,
        help="Saved PCA model fitted to the main-model formula-average embeddings.",
    )
    parser.add_argument(
        "--main-embedding-dir",
        required=True,
        help="Directory containing the main-model formula-record embeddings.",
    )
    parser.add_argument(
        "--repeat-embedding-root",
        required=True,
        help="Root directory containing repeat_* embedding directories.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for PCA models, matching table, and correlation panels.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=6,
        help="Number of principal components used for repeat-model matching.",
    )

    parser.add_argument(
        "--formula-symptom-file",
        default="selected_formula_symptom_embeddings.pkl",
        help="Filename for formula-record symptom-pattern embeddings.",
    )
    parser.add_argument(
        "--formula-herb-file",
        default="selected_formula_herb_embeddings.pkl",
        help="Filename for formula-record formula embeddings.",
    )
    parser.add_argument(
        "--formula-ids-file",
        default="selected_formula_ids(line_number).pkl",
        help="Filename for formula-record row IDs.",
    )

    return parser


def load_pickle(path: Path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def save_figure(fig, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    stem = png_path.with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })


def discover_repeat_dirs(root: Path) -> Dict[str, Path]:
    repeat_dirs = sorted(
        (path for path in root.glob("repeat_*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if not repeat_dirs:
        raise FileNotFoundError(f"No repeat_* folders found under: {root}")
    return {path.name: path for path in repeat_dirs}


def load_formula_average_embeddings(
    embedding_dir: Path,
    formula_symptom_file: str,
    formula_herb_file: str,
    formula_ids_file: str,
) -> Tuple[np.ndarray, np.ndarray]:
    symptom_embeddings = np.asarray(
        load_pickle(embedding_dir / formula_symptom_file),
        dtype=np.float64,
    )
    formula_embeddings = np.asarray(
        load_pickle(embedding_dir / formula_herb_file),
        dtype=np.float64,
    )
    formula_ids = np.asarray(
        load_pickle(embedding_dir / formula_ids_file),
        dtype=int,
    )

    if symptom_embeddings.ndim != 2 or formula_embeddings.ndim != 2:
        raise ValueError(
            f"Formula embeddings must be two-dimensional: {embedding_dir}"
        )

    if symptom_embeddings.shape != formula_embeddings.shape:
        raise ValueError(
            f"Symptom-pattern and formula embedding shapes differ in {embedding_dir}: "
            f"{symptom_embeddings.shape} vs {formula_embeddings.shape}."
        )

    if symptom_embeddings.shape[0] != len(formula_ids):
        raise ValueError(
            f"Formula embedding rows and formula IDs are not aligned: {embedding_dir}"
        )

    if len(np.unique(formula_ids)) != len(formula_ids):
        raise ValueError(f"Duplicate formula IDs found in: {embedding_dir}")

    formula_average = 0.5 * (symptom_embeddings + formula_embeddings)
    return formula_average, formula_ids


def align_formula_embeddings(
    main_data: Tuple[np.ndarray, np.ndarray],
    repeat_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Reorder every repeat to the main-model formula-ID order."""
    main_X, main_ids = main_data
    main_ids = np.asarray(main_ids, dtype=int)

    if len(np.unique(main_ids)) != len(main_ids):
        raise ValueError("Duplicate formula IDs found in the main model.")

    main_id_set = set(map(int, main_ids))
    repeat_aligned = {}

    for repeat_name, (repeat_X, repeat_ids) in repeat_data.items():
        repeat_ids = np.asarray(repeat_ids, dtype=int)

        if len(np.unique(repeat_ids)) != len(repeat_ids):
            raise ValueError(f"{repeat_name}: duplicate formula IDs found.")

        repeat_id_set = set(map(int, repeat_ids))
        if repeat_id_set != main_id_set:
            missing = sorted(main_id_set - repeat_id_set)
            extra = sorted(repeat_id_set - main_id_set)
            raise ValueError(
                f"{repeat_name}: formula IDs do not match the main model. "
                f"Missing={missing[:10]}, extra={extra[:10]}."
            )

        if repeat_X.shape[0] != len(repeat_ids):
            raise ValueError(
                f"{repeat_name}: embedding rows and formula IDs are not aligned."
            )

        row_map = {
            int(record_id): row
            for row, record_id in enumerate(repeat_ids)
        }
        repeat_aligned[repeat_name] = repeat_X[
            [row_map[int(record_id)] for record_id in main_ids]
        ]

    return main_X, repeat_aligned, main_ids.copy()


def pc_score_correlation(
    main_scores: np.ndarray,
    repeat_scores: np.ndarray,
) -> np.ndarray:
    """Calculate signed Pearson correlations for all main/repeat PC pairs."""
    if main_scores.shape[0] != repeat_scores.shape[0]:
        raise ValueError("Main and repeat PC-score matrices must contain the same rows.")

    corr = np.empty(
        (main_scores.shape[1], repeat_scores.shape[1]),
        dtype=float,
    )

    for main_pc in range(main_scores.shape[1]):
        for repeat_pc in range(repeat_scores.shape[1]):
            corr[main_pc, repeat_pc] = np.corrcoef(
                main_scores[:, main_pc],
                repeat_scores[:, repeat_pc],
            )[0, 1]

    if not np.isfinite(corr).all():
        raise ValueError("Non-finite PC-score correlations were encountered.")

    return corr


def match_pcs(corr: np.ndarray) -> List[Tuple[int, int, float]]:
    """Match PCs one-to-one by maximizing total absolute correlation."""
    main_idx, repeat_idx = linear_sum_assignment(-np.abs(corr))
    matches = [
        (int(i), int(j), float(corr[i, j]))
        for i, j in zip(main_idx, repeat_idx)
    ]
    return sorted(matches, key=lambda item: item[0])


def plot_correlation_panels(
    correlation_matrices: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Plot signed PC-score correlation matrices for all repeat models."""
    repeat_names = list(correlation_matrices)
    n_models = len(repeat_names)
    if n_models == 0:
        raise ValueError("No repeat-model correlation matrices were provided.")

    n_cols = min(5, n_models)
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11.2 if n_cols == 5 else 2.25 * n_cols,
                 4.6 if n_rows == 2 else 2.3 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)

    panel_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    image = None

    for idx, (ax, repeat_name) in enumerate(zip(axes, repeat_names)):
        corr = correlation_matrices[repeat_name]
        image = ax.imshow(
            corr,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            aspect="equal",
            interpolation="nearest",
        )

        ax.set_title(repeat_name, pad=6, fontsize=10, fontweight="normal")
        if idx < len(panel_letters):
            ax.text(
                -0.18,
                1.06,
                panel_letters[idx],
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="bottom",
                ha="left",
            )

        ax.set_xticks(np.arange(corr.shape[1]))
        ax.set_yticks(np.arange(corr.shape[0]))
        ax.set_xticklabels([str(i) for i in range(1, corr.shape[1] + 1)])
        ax.set_yticklabels([str(i) for i in range(1, corr.shape[0] + 1)])

        ax.set_xticks(np.arange(-0.5, corr.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, corr.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                value = corr[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if abs(value) >= 0.60 else "black",
                )

        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[n_models:]:
        ax.axis("off")

    for row in range(n_rows):
        for col in range(n_cols):
            ax_idx = row * n_cols + col
            if ax_idx >= n_models:
                continue
            axes[ax_idx].tick_params(
                axis="x",
                labelbottom=(row == n_rows - 1),
            )
            axes[ax_idx].tick_params(
                axis="y",
                labelleft=(col == 0),
            )

    fig.supxlabel("Repeat-model principal components", y=0.04, fontsize=10)
    fig.supylabel("Main-model principal components", x=0.03, fontsize=10)

    fig.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.12,
        top=0.90,
        wspace=0.18,
        hspace=0.30,
    )

    cbar_ax = fig.add_axes([0.90, 0.20, 0.012, 0.60])
    colorbar = fig.colorbar(image, cax=cbar_ax)
    colorbar.set_label("PC-score correlation", fontsize=10)
    colorbar.ax.tick_params(labelsize=8)

    save_figure(fig, out_path)
    plt.close(fig)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    set_plot_style()

    if args.n_components < 1:
        raise ValueError("--n-components must be at least 1.")

    out_dir = Path(args.out_dir)
    figure_dir = out_dir / "figures"
    pca_dir = out_dir / "repeat_pca_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    main_pca = load_pickle(Path(args.main_pca))
    main_data = load_formula_average_embeddings(
        Path(args.main_embedding_dir),
        args.formula_symptom_file,
        args.formula_herb_file,
        args.formula_ids_file,
    )

    repeat_dirs = discover_repeat_dirs(Path(args.repeat_embedding_root))
    repeat_data = {
        repeat_name: load_formula_average_embeddings(
            repeat_dir,
            args.formula_symptom_file,
            args.formula_herb_file,
            args.formula_ids_file,
        )
        for repeat_name, repeat_dir in repeat_dirs.items()
    }

    main_X, repeat_X, _ = align_formula_embeddings(
        main_data,
        repeat_data,
    )

    if main_X.shape[1] != int(main_pca.n_features_in_):
        raise ValueError(
            "Main formula-average embedding dimension does not match the saved PCA."
        )

    n_components = min(
        args.n_components,
        int(main_pca.n_components_),
        main_X.shape[0],
        main_X.shape[1],
    )
    if n_components < 1:
        raise ValueError("No PCA components are available for matching.")

    main_scores = main_pca.transform(main_X)[:, :n_components]

    correlation_matrices = {}
    matching_rows = []

    for repeat_name, X in repeat_X.items():
        if X.shape != main_X.shape:
            raise ValueError(
                f"{repeat_name}: aligned embedding shape {X.shape} does not match "
                f"the main-model shape {main_X.shape}."
            )

        repeat_pca = PCA(
            n_components=n_components,
            svd_solver="full",
        )
        repeat_scores = repeat_pca.fit_transform(X)
        save_pickle(
            repeat_pca,
            pca_dir / f"{repeat_name}_pca.pkl",
        )

        corr = pc_score_correlation(
            main_scores,
            repeat_scores,
        )
        correlation_matrices[repeat_name] = corr

        for main_pc, repeat_pc, signed_corr in match_pcs(corr):
            matching_rows.append({
                "repeat_model": repeat_name,
                "main_pc": main_pc + 1,
                "matched_repeat_pc": repeat_pc + 1,
                "signed_correlation": signed_corr,
                "absolute_correlation": abs(signed_corr),
                "sign_flipped": bool(signed_corr < 0),
            })

        print(f"[done] {repeat_name}", flush=True)

    matching_table = pd.DataFrame(matching_rows)
    matching_table.to_csv(
        out_dir / "pc_matching_table.csv",
        index=False,
        encoding="utf_8_sig",
    )

    plot_correlation_panels(
        correlation_matrices,
        figure_dir / "pca_correlation_panels.png",
    )

    print("\nSaved:")
    print(out_dir / "pc_matching_table.csv")
    print(pca_dir)
    print(figure_dir / "pca_correlation_panels.png")


if __name__ == "__main__":
    main()
