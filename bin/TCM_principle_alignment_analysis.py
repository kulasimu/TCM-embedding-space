#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TCM principle alignment analysis.

Run this script AFTER TCM_embedding_pca_projection.py.

This script reads saved PCA coordinates and selected formula data, then runs:
    1. Syndrome-labelled formula distribution in PCA space
    2. Formula-level cold-hot gradient
    3. Formula-level dominant channel (meridian) distribution
    4. Individual herb nature distribution along PCs

It does NOT fit PCA again.
"""

import argparse
import ast
import os
from pathlib import Path
from typing import List, Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata



# ---------------------------------------------------------------------
# Syndrome labels/colors used consistently in the manuscript and SI
# ---------------------------------------------------------------------

SYNDROME_COLOR_MAP = {
    "寒邪犯胃证": "blue",
    "脾寒证": "cornflowerblue",
    "肺热炽盛证": "red",
    "心热证": "darkred",
    "肝血虚证": "darkorange",
    "肾气虚证": "green",
    "风邪袭表证": "cyan",
}

SYNDROME_LABEL_MAP = {
    "寒邪犯胃证": "Cold invading Stomach",
    "脾寒证": "Spleen Cold",
    "肺热炽盛证": "Lung Heat Excess",
    "心热证": "Heart Heat",
    "肝血虚证": "Liver Blood Deficiency",
    "肾气虚证": "Kidney Qi Deficiency",
    "风邪袭表证": "Wind attacking Exterior",
}


# ---------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["mathtext.fontset"] = "dejavusans"


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def ensure_dir(path: str):
    """Create directory if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_list_cell(x):
    """
    Parse list-like columns.

    In pkl files, Symptoms / Herbs / Syndromes are real lists.
    In csv files, they may be strings like "['咳嗽', '发热']".
    """
    if isinstance(x, list):
        return x

    if isinstance(x, tuple):
        return list(x)

    if pd.isna(x):
        return []

    if isinstance(x, str):
        try:
            y = ast.literal_eval(x)
            return y if isinstance(y, list) else []
        except Exception:
            return []

    return []


def pc_columns(df: pd.DataFrame) -> List[str]:
    """Return PC columns in correct numeric order."""
    cols = [c for c in df.columns if c.startswith("PC")]
    return sorted(cols, key=lambda x: int(x.replace("PC", "")))


def load_selected_formula_data(analysis_dir: str) -> pd.DataFrame:
    """
    Load selected formula records saved by TCM_embedding_pca_projection.py.

    Expected:
        analysis_dir/selected_formula_data/selected_formula_data.pkl
    or:
        analysis_dir/selected_formula_data/selected_formula_data.csv
    """
    data_dir = os.path.join(analysis_dir, "selected_formula_data")

    pkl_path = os.path.join(data_dir, "selected_formula_data.pkl")
    csv_path = os.path.join(data_dir, "selected_formula_data.csv")

    if os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path).reset_index(drop=True)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path).reset_index(drop=True)
    else:
        raise FileNotFoundError(
            "Cannot find selected formula data. Expected:\n"
            f"{pkl_path}\n"
            f"{csv_path}"
        )

    for col in ["Symptoms", "Herbs", "Syndromes"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_cell)

    return df


def load_coord_table(analysis_dir: str, relative_path: str) -> pd.DataFrame:
    """Load PCA coordinate table."""
    path = os.path.join(analysis_dir, relative_path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find coordinate file: {path}")

    return pd.read_csv(path)


def load_herb_property_table(path: str) -> pd.DataFrame:
    """
    Load herb nature / property table.

    Expected format:
        first column = herb name
        remaining columns = herb property indicators

    This script assumes:
        first 5 property columns = cold, cool, neutral, warm, hot
        last 12 property columns = meridian indicators
    """
    df = pd.read_csv(path)

    if df.shape[1] < 6:
        raise ValueError("Herb property table should contain herb name + property columns.")

    df = df.rename(columns={df.columns[0]: "herb"})

    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


def setup_output_dirs(analysis_dir: str):
    """
    Create output folders for principle alignment results.

    Input analysis_dir should be the model analysis root, e.g.
        results/TCM_embedding_analysis/original(average)

    Outputs are saved to:
        analysis_dir/principle_alignment/figures
        analysis_dir/principle_alignment/fig_data
    """

    out_dir = os.path.join(analysis_dir, "principle_alignment")
    fig_dir = os.path.join(out_dir, "figures")
    data_dir = os.path.join(out_dir, "fig_data")

    ensure_dir(fig_dir)
    ensure_dir(data_dir)

    return fig_dir, data_dir


def save_current_figure(fig, fig_dir: str, name: str):
    """Save matplotlib figure in multiple formats."""
    fig.savefig(os.path.join(fig_dir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, f"{name}.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=600, bbox_inches="tight")


def style_3d_axis(ax):
    """Simple 3D axis style."""
    ax.set_xlabel("PC1", labelpad=6)
    ax.set_ylabel("PC2", labelpad=6)
    ax.set_zlabel("PC3", labelpad=6)

    ax.tick_params(labelsize=8, pad=1)
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=22, azim=38)


# ---------------------------------------------------------------------
# 1. Syndrome-labelled formula distribution
# ---------------------------------------------------------------------

def plot_syndrome_embedding_distribution(
    selected_df: pd.DataFrame,
    coord_raw_df: pd.DataFrame,
    coord_scaled_df: pd.DataFrame,
    fig_dir: str,
    data_dir: str,
    selected_syndromes: Sequence[str],
    output_prefix: str,
    title_suffix: str,
):
    """
    Plot formula records with selected syndrome labels in PC1-PC3 space.

    """
    pc_raw_cols = pc_columns(coord_raw_df)[:3]
    pc_scaled_cols = pc_columns(coord_scaled_df)[:3]

    if len(pc_raw_cols) < 3 or len(pc_scaled_cols) < 3:
        raise ValueError("Coordinate tables must contain at least PC1-PC3.")

    rows = []


    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")

    for k, syn in enumerate(selected_syndromes):
        ids = [
            i for i, syns in enumerate(selected_df["Syndromes"].tolist())
            if syn in syns
        ]

        if len(ids) == 0:
            continue

        sub = coord_scaled_df.iloc[ids].copy()

        color = SYNDROME_COLOR_MAP.get(syn, "gray")
        label = SYNDROME_LABEL_MAP.get(syn, syn)
        ax.scatter(
            sub[pc_scaled_cols[0]],
            sub[pc_scaled_cols[1]],
            sub[pc_scaled_cols[2]],
            s=14,
            marker="o",
            facecolors="none",
            edgecolors=color,
            linewidths=0.7,
            alpha=0.55,
            label=label,
        )

        for i in ids:
            rows.append({
                "row_index": i,
                "syndrome": syn,
                "Title": selected_df.iloc[i].get("Title", ""),
                "PC1_raw": coord_raw_df.iloc[i][pc_raw_cols[0]],
                "PC2_raw": coord_raw_df.iloc[i][pc_raw_cols[1]],
                "PC3_raw": coord_raw_df.iloc[i][pc_raw_cols[2]],
                "PC1_scaled": coord_scaled_df.iloc[i][pc_scaled_cols[0]],
                "PC2_scaled": coord_scaled_df.iloc[i][pc_scaled_cols[1]],
                "PC3_scaled": coord_scaled_df.iloc[i][pc_scaled_cols[2]],
            })

    style_3d_axis(ax)
    ax.set_title(f"Syndrome-labelled formulas\n{title_suffix}", fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    save_current_figure(fig, fig_dir, output_prefix)
    plt.close(fig)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(
        os.path.join(data_dir, f"{output_prefix}_plot_data.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    print("Saved syndrome embedding distribution.")

    return out_df


# ---------------------------------------------------------------------
# 2. Formula-level cold-hot gradient
# ---------------------------------------------------------------------

def calculate_formula_coldhot_score(
    selected_df: pd.DataFrame,
    herb_prop_df: pd.DataFrame,
):
    """
    Calculate formula-level cold-hot score.

    The first five herb property columns are assumed to be:
        cold, cool, neutral, warm, hot

    Scores:
        cold = 1
        cool = 2
        neutral = 3
        warm = 4
        hot = 5

    Formula score = average score of its herbs.
    """
    nature_cols = list(herb_prop_df.columns[1:6])
    score_map = np.array([1, 2, 3, 4, 5], dtype=float)

    herb_score = {}

    for _, row in herb_prop_df.iterrows():
        values = row[nature_cols].values.astype(float)
        herb_score[row["herb"]] = float(score_map[int(np.argmax(values))])

    formula_scores = []
    formula_herb_scores = []

    for herbs in selected_df["Herbs"].tolist():
        scores = [herb_score[h] for h in herbs if h in herb_score]

        formula_herb_scores.append(scores)
        formula_scores.append(np.mean(scores) if len(scores) > 0 else np.nan)

    formula_scores = np.asarray(formula_scores, dtype=float)
    formula_scores[np.isnan(formula_scores)] = 3.0

    return formula_scores, formula_herb_scores


def plot_formula_coldhot_gradient(
    selected_df: pd.DataFrame,
    coord_raw_df: pd.DataFrame,
    coord_scaled_df: pd.DataFrame,
    herb_prop_df: pd.DataFrame,
    fig_dir: str,
    data_dir: str,
    output_prefix: str,
    title_suffix: str,
):
    """
    Color formula PCA coordinates by formula-level cold-hot score.

    """
    pc_raw_cols = pc_columns(coord_raw_df)[:3]
    pc_scaled_cols = pc_columns(coord_scaled_df)[:3]

    coldhot_score, herb_scores = calculate_formula_coldhot_score(
        selected_df,
        herb_prop_df,
    )

    score_rank = rankdata(coldhot_score, method="average")
    score_quantile = (score_rank - 1) / max(len(score_rank) - 1, 1)

    fig = plt.figure(figsize=(5.0, 4.5))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        coord_scaled_df[pc_scaled_cols[0]],
        coord_scaled_df[pc_scaled_cols[1]],
        coord_scaled_df[pc_scaled_cols[2]],
        c=score_quantile,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        s=11,
        marker="o",
        alpha=0.60,
        linewidths=0,
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08)
    cbar.set_label("Cold-hot tendency", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Cold", "Neutral", "Hot"])
    cbar.ax.tick_params(labelsize=8)

    style_3d_axis(ax)
    ax.set_title(f"Formula cold-hot gradient\n{title_suffix}", fontsize=11, pad=10)

    plt.tight_layout()
    save_current_figure(fig, fig_dir, output_prefix)
    plt.close(fig)

    out_df = pd.DataFrame({
        "row_index": np.arange(len(selected_df)),
        "Title": selected_df["Title"].values if "Title" in selected_df.columns else "",

        "PC1_raw": coord_raw_df[pc_raw_cols[0]],
        "PC2_raw": coord_raw_df[pc_raw_cols[1]],
        "PC3_raw": coord_raw_df[pc_raw_cols[2]],

        "PC1_scaled": coord_scaled_df[pc_scaled_cols[0]],
        "PC2_scaled": coord_scaled_df[pc_scaled_cols[1]],
        "PC3_scaled": coord_scaled_df[pc_scaled_cols[2]],

        "coldhot_score": coldhot_score,
        "coldhot_quantile": score_quantile,
        "herb_scores": herb_scores,
    })

    out_df.to_csv(
        os.path.join(data_dir, f"{output_prefix}_plot_data.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    print("Saved formula cold-hot gradient.")

    return out_df


# ---------------------------------------------------------------------
# 3. Formula-level channel (meridian) attribution
# ---------------------------------------------------------------------

def calculate_formula_dominant_meridian(
    selected_df: pd.DataFrame,
    herb_prop_df: pd.DataFrame,
    meridian_cols: Optional[Sequence[str]] = None,
):
    """
    Calculate dominant meridian for each formula.

    By default, the last 12 property columns are used as meridian indicators.
    Formula dominant meridian = meridian with the largest count among its herbs.
    """
    if meridian_cols is None:
        meridian_cols = list(herb_prop_df.columns[-12:])

    herb_to_meridians = {}

    for _, row in herb_prop_df.iterrows():
        herb_to_meridians[row["herb"]] = [
            m for m in meridian_cols
            if float(row[m]) > 0
        ]

    dominant = []
    count_rows = []

    for herbs in selected_df["Herbs"].tolist():
        counts = {m: 0 for m in meridian_cols}

        for h in herbs:
            for m in herb_to_meridians.get(h, []):
                counts[m] += 1

        best = max(counts, key=counts.get)

        dominant.append(best if counts[best] > 0 else "Other")
        count_rows.append(counts)

    return np.asarray(dominant), count_rows, list(meridian_cols)


def plot_formula_meridian_distribution(
    selected_df: pd.DataFrame,
    coord_raw_df: pd.DataFrame,
    coord_scaled_df: pd.DataFrame,
    herb_prop_df: pd.DataFrame,
    fig_dir: str,
    data_dir: str,
    selected_meridians: Sequence[str],
    output_prefix: str,
    title_suffix: str,
):
    """
    Plot formula PCA coordinates by dominant channel (meridian).

    """
    pc_raw_cols = pc_columns(coord_raw_df)[:3]
    pc_scaled_cols = pc_columns(coord_scaled_df)[:3]

    dominant, meridian_counts, _ = calculate_formula_dominant_meridian(
        selected_df,
        herb_prop_df,
    )

    plot_labels = np.asarray([
        m if m in selected_meridians else "Other"
        for m in dominant
    ])

    labels = list(selected_meridians) + ["Other"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")

    for k, lab in enumerate(labels):
        ids = np.where(plot_labels == lab)[0]

        if len(ids) == 0:
            continue

        ax.scatter(
            coord_scaled_df.iloc[ids][pc_scaled_cols[0]],
            coord_scaled_df.iloc[ids][pc_scaled_cols[1]],
            coord_scaled_df.iloc[ids][pc_scaled_cols[2]],
            s=12,
            marker="o",
            facecolors="none",
            edgecolors=[colors[k]],
            linewidths=0.7,
            alpha=0.50,
            label=lab,
        )

    style_3d_axis(ax)
    ax.set_title(f"Formula dominant meridian\n{title_suffix}", fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    save_current_figure(fig, fig_dir, output_prefix)
    plt.close(fig)

    out_df = pd.DataFrame({
        "row_index": np.arange(len(selected_df)),
        "Title": selected_df["Title"].values if "Title" in selected_df.columns else "",

        "PC1_raw": coord_raw_df[pc_raw_cols[0]],
        "PC2_raw": coord_raw_df[pc_raw_cols[1]],
        "PC3_raw": coord_raw_df[pc_raw_cols[2]],

        "PC1_scaled": coord_scaled_df[pc_scaled_cols[0]],
        "PC2_scaled": coord_scaled_df[pc_scaled_cols[1]],
        "PC3_scaled": coord_scaled_df[pc_scaled_cols[2]],

        "dominant_meridian": dominant,
        "plot_meridian": plot_labels,
        "meridian_counts": meridian_counts,
    })

    out_df.to_csv(
        os.path.join(data_dir, f"{output_prefix}_plot_data.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    print("Saved formula meridian distribution.")

    return out_df


# ---------------------------------------------------------------------
# 4. Individual herb nature distribution along PCs
# ---------------------------------------------------------------------

def assign_herb_nature_group(herb_prop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign herbs into three broad nature groups:
        Cold & Cool
        Neutral
        Warm & Hot
    """
    nature_cols = list(herb_prop_df.columns[1:6])
    nature_idx = np.argmax(
        herb_prop_df[nature_cols].values.astype(float),
        axis=1,
    )

    groups = []

    for idx in nature_idx:
        if idx < 2:
            groups.append("Cold & Cool")
        elif idx == 2:
            groups.append("Neutral")
        else:
            groups.append("Warm & Hot")

    return pd.DataFrame({
        "herb": herb_prop_df["herb"].values,
        "herb_nature_group": groups,
    })


def plot_herb_nature_pc_distribution(
    herb_coord_raw_df: pd.DataFrame,
    herb_coord_scaled_df: pd.DataFrame,
    herb_prop_df: pd.DataFrame,
    fig_dir: str,
    data_dir: str,
    n_pcs: int = 3,
):
    """
    Compare individual herb PC coordinates across herb nature groups.

    """
    raw_pc_cols = pc_columns(herb_coord_raw_df)[:n_pcs]
    scaled_pc_cols = pc_columns(herb_coord_scaled_df)[:n_pcs]

    nature_df = assign_herb_nature_group(herb_prop_df)

    df = herb_coord_raw_df.merge(
        nature_df,
        left_on="entity_name",
        right_on="herb",
        how="left",
    )

    rows = []
    for pc_i, pc in enumerate(raw_pc_cols):
        pc_scaled = scaled_pc_cols[pc_i]

        # Rank herbs by their raw projection on this PC.
        # Larger projection = larger rank, consistent with scipy.stats.rankdata default.
        pc_rank = rankdata(df[pc].values)

        for idx, row in df.iterrows():
            rows.append({
                "entity_id": row.get("entity_id", np.nan),
                "entity_name": row["entity_name"],
                "PC": pc,
                "projection_raw": row[pc],
                "projection_scaled": herb_coord_scaled_df.iloc[idx][pc_scaled],
                "projection_rank": pc_rank[idx],
                "herb_nature_group": row["herb_nature_group"],
            })

    plot_df = pd.DataFrame(rows).dropna(subset=["herb_nature_group"])

    groups = ["Cold & Cool", "Neutral", "Warm & Hot"]

    group_colors = {
        "Cold & Cool": "#4C78A8",
        "Neutral": "#54A24B",
        "Warm & Hot": "#C44E52",
    }

    fig, axes = plt.subplots(
        1,
        len(raw_pc_cols),
        figsize=(3.0 * len(raw_pc_cols), 3.2),
    )

    if len(raw_pc_cols) == 1:
        axes = [axes]

    rng = np.random.default_rng(1)

    for ax, pc in zip(axes, raw_pc_cols):
        values = [
            plot_df.loc[
                (plot_df["PC"] == pc) &
                (plot_df["herb_nature_group"] == g),
                "projection_rank",
            ].values
            for g in groups
        ]

        bp = ax.boxplot(
            values,
            tick_labels=groups,
            patch_artist=True,
            showfliers=False,
            widths=0.55,
        )

        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor("white")
            patch.set_edgecolor(group_colors[g])
            patch.set_linewidth(1.2)

        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.2)

        # Light jittered points.
        for i, g in enumerate(groups, start=1):
            y = plot_df.loc[
                (plot_df["PC"] == pc) &
                (plot_df["herb_nature_group"] == g),
                "projection_rank",
            ].values

            x = i + rng.normal(0, 0.045, size=len(y))

            ax.scatter(
                x,
                y,
                s=8,
                alpha=0.25,
                color=group_colors[g],
                linewidths=0,
            )

        ax.set_title(pc, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Projection rank" if ax is axes[0] else "")

        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)

    plt.tight_layout()
    save_current_figure(fig, fig_dir, "herb_nature_pc_rank_distribution")
    plt.close(fig)

    plot_df.to_csv(
        os.path.join(data_dir, "herb_nature_pc_rank_distribution_plot_data.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    print("Saved herb nature PC distribution.")

    return plot_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="TCM principle alignment analysis based on saved PCA coordinates."
    )

    parser.add_argument(
        "--analysis-dir",
        required=True,
        help="Output directory generated by TCM_embedding_pca_projection.py.",
    )

    parser.add_argument(
        "--record-herb-coord-file",
        default="pca/formula_herb_embedding_pca_coords.csv",
        help="Raw record-level herb-side PCA coordinates, used for analysis.",
    )

    parser.add_argument(
        "--record-herb-coord-file-scaled",
        default="pca/formula_herb_embedding_pca_coords_scaled.csv",
        help="Scaled record-level herb-side PCA coordinates, used for visualization.",
    )

    parser.add_argument(
        "--record-symptom-coord-file",
        default="pca/formula_symptom_embedding_pca_coords.csv",
        help="Raw record-level symptom-side PCA coordinates, used for analysis.",
    )

    parser.add_argument(
        "--record-symptom-coord-file-scaled",
        default="pca/formula_symptom_embedding_pca_coords_scaled.csv",
        help="Scaled record-level symptom-side PCA coordinates, used for visualization.",
    )

    parser.add_argument(
        "--herb-coord-file",
        default="pca/individual_herb_embedding_pca_coords.csv",
        help="Raw individual herb PCA coordinates, used for analysis.",
    )

    parser.add_argument(
        "--herb-coord-file-scaled",
        default="pca/individual_herb_embedding_pca_coords_scaled.csv",
        help="Scaled individual herb PCA coordinates, used for visualization.",
    )

    parser.add_argument(
        "--herb-natures",
        default="core/standard_TCM_entities/herb_natures.csv",
        help="Herb nature/property CSV.",
    )

    parser.add_argument(
        "--plot",
        choices=["all", "syndrome", "coldhot", "meridian", "herb_nature"],
        default="all",
        help="Which analysis to run.",
    )

    parser.add_argument(
        "--selected-syndromes",
        nargs="*",
        default=[
            "寒邪犯胃证",
            "脾寒证",
            "肺热炽盛证",
            "心热证",
            "肝血虚证",
            "肾气虚证",
            "风邪袭表证",
        ],
        help="Syndromes to display in the syndrome PCA plot.",
    )

    parser.add_argument(
        "--selected-meridians",
        nargs="*",
        default=["心", "肝", "脾", "胃", "肾", "肺"],
        help="Dominant meridians to display; all others are grouped as Other.",
    )

    parser.add_argument(
        "--embedding-side",
        choices=["both", "symptom", "herb"],
        default="both",
        help=(
            "Embedding side used for syndrome-labelled PCA visualization. "
            "'both' plots symptom-pattern and formula embeddings; "
            "'symptom' or 'herb' plots one side only."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    fig_dir, data_dir = setup_output_dirs(args.analysis_dir)

    run_syndrome = args.plot in ["all", "syndrome"]
    run_coldhot = args.plot in ["all", "coldhot"]
    run_meridian = args.plot in ["all", "meridian"]
    run_herb_nature = args.plot in ["all", "herb_nature"]

    selected_df = None
    herb_prop_df = None

    # Formula-level analyses use the selected records saved by the PCA step.
    if run_syndrome or run_coldhot or run_meridian:
        selected_df = load_selected_formula_data(args.analysis_dir)

    # Herb annotations are required for cold-hot, channel, and herb-nature analyses.
    if run_coldhot or run_meridian or run_herb_nature:
        herb_prop_df = load_herb_property_table(args.herb_natures)
        print("Herb property table:", herb_prop_df.shape)

    # ------------------------------------------------------------
    # Syndrome-labelled symptom-pattern / formula embeddings
    # ------------------------------------------------------------
    if run_syndrome:
        if "Syndromes" not in selected_df.columns:
            raise ValueError("selected formula data must contain a Syndromes column.")

        if args.embedding_side in ["both", "herb"]:
            record_herb_coord_raw_df = load_coord_table(
                args.analysis_dir,
                args.record_herb_coord_file,
            )
            record_herb_coord_scaled_df = load_coord_table(
                args.analysis_dir,
                args.record_herb_coord_file_scaled,
            )

            if len(selected_df) != len(record_herb_coord_raw_df):
                raise ValueError(
                    "selected formula data and record-herb raw coordinates "
                    "have different lengths."
                )
            if len(selected_df) != len(record_herb_coord_scaled_df):
                raise ValueError(
                    "selected formula data and record-herb scaled coordinates "
                    "have different lengths."
                )

            plot_syndrome_embedding_distribution(
                selected_df=selected_df,
                coord_raw_df=record_herb_coord_raw_df,
                coord_scaled_df=record_herb_coord_scaled_df,
                fig_dir=fig_dir,
                data_dir=data_dir,
                selected_syndromes=args.selected_syndromes,
                output_prefix="syndrome_distribution_record_herb",
                title_suffix="formula embedding",
            )

        if args.embedding_side in ["both", "symptom"]:
            record_symptom_coord_raw_df = load_coord_table(
                args.analysis_dir,
                args.record_symptom_coord_file,
            )
            record_symptom_coord_scaled_df = load_coord_table(
                args.analysis_dir,
                args.record_symptom_coord_file_scaled,
            )

            if len(selected_df) != len(record_symptom_coord_raw_df):
                raise ValueError(
                    "selected formula data and record-symptom raw coordinates "
                    "have different lengths."
                )
            if len(selected_df) != len(record_symptom_coord_scaled_df):
                raise ValueError(
                    "selected formula data and record-symptom scaled coordinates "
                    "have different lengths."
                )

            plot_syndrome_embedding_distribution(
                selected_df=selected_df,
                coord_raw_df=record_symptom_coord_raw_df,
                coord_scaled_df=record_symptom_coord_scaled_df,
                fig_dir=fig_dir,
                data_dir=data_dir,
                selected_syndromes=args.selected_syndromes,
                output_prefix="syndrome_distribution_record_symptom",
                title_suffix="symptom-pattern embedding",
            )

    # ------------------------------------------------------------
    # Formula-level conventional TCM annotations
    # ------------------------------------------------------------
    if run_coldhot or run_meridian:
        record_herb_coord_raw_df = load_coord_table(
            args.analysis_dir,
            args.record_herb_coord_file,
        )
        record_herb_coord_scaled_df = load_coord_table(
            args.analysis_dir,
            args.record_herb_coord_file_scaled,
        )

        if len(selected_df) != len(record_herb_coord_raw_df):
            raise ValueError(
                "selected formula data and record-herb raw coordinates "
                "have different lengths."
            )
        if len(selected_df) != len(record_herb_coord_scaled_df):
            raise ValueError(
                "selected formula data and record-herb scaled coordinates "
                "have different lengths."
            )

        if run_coldhot:
            plot_formula_coldhot_gradient(
                selected_df=selected_df,
                coord_raw_df=record_herb_coord_raw_df,
                coord_scaled_df=record_herb_coord_scaled_df,
                herb_prop_df=herb_prop_df,
                fig_dir=fig_dir,
                data_dir=data_dir,
                output_prefix="coldhot_gradient_record_herb",
                title_suffix="formula embedding",
            )

        if run_meridian:
            plot_formula_meridian_distribution(
                selected_df=selected_df,
                coord_raw_df=record_herb_coord_raw_df,
                coord_scaled_df=record_herb_coord_scaled_df,
                herb_prop_df=herb_prop_df,
                fig_dir=fig_dir,
                data_dir=data_dir,
                selected_meridians=args.selected_meridians,
                output_prefix="meridian_distribution_record_herb",
                title_suffix="formula embedding",
            )

    # ------------------------------------------------------------
    # Individual-herb nature distribution along principal components
    # ------------------------------------------------------------
    if run_herb_nature:
        individual_herb_coord_raw_df = load_coord_table(
            args.analysis_dir,
            args.herb_coord_file,
        )
        individual_herb_coord_scaled_df = load_coord_table(
            args.analysis_dir,
            args.herb_coord_file_scaled,
        )

        print("Individual herb raw coordinates:", individual_herb_coord_raw_df.shape)
        print("Individual herb scaled coordinates:", individual_herb_coord_scaled_df.shape)

        plot_herb_nature_pc_distribution(
            herb_coord_raw_df=individual_herb_coord_raw_df,
            herb_coord_scaled_df=individual_herb_coord_scaled_df,
            herb_prop_df=herb_prop_df,
            fig_dir=fig_dir,
            data_dir=data_dir,
            n_pcs=3,
        )

    print("Done. Results saved to:")
    print("  figures:", fig_dir)
    print("  fig data:", data_dir)


if __name__ == "__main__":
    main()