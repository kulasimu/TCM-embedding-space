#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Repeated-training robustness evaluation for TCM-ES embeddings.

The script reproduces the record-level stability analyses used for repeated
TCM-ES training runs:

1. Generate symptom-pattern and formula embeddings for the same eligible
   formula records using each repeated model.
2. Quantify global stability by Spearman correlation of cosine-similarity
   values for the same randomly sampled record pairs across repeats.
3. Quantify local stability by nearest-neighbour Jaccard overlap across
   repeat-model pairs.
4. Visualize per-record local stability on the primary-model PCA space.

For the manuscript setting, record embeddings are defined as the average of
the symptom-pattern and formula embeddings.
"""

import argparse
import glob
import os
import pickle
from itertools import combinations
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: str):
    """Create a directory if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_pickle(path: str):
    """Load a pickle file."""
    with open(path, "rb") as fp:
        return pickle.load(fp)


def save_pickle(obj, path: str):
    """Save an object as a pickle file."""
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization used for cosine similarity."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def parse_k_values(k_values: str) -> List[int]:
    """Parse a comma-separated list of positive neighbourhood sizes."""
    values = [int(x.strip()) for x in k_values.split(",") if x.strip()]
    if len(values) == 0 or any(k <= 0 for k in values):
        raise ValueError("Neighbourhood sizes must be positive integers.")
    return sorted(set(values))


def set_plot_style():
    """Set plotting parameters used by robustness figures."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 600


def save_figure(fig, out_path_without_ext: str):
    """Save a figure as PNG, PDF, and SVG."""
    fig.savefig(out_path_without_ext + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_path_without_ext + ".pdf", bbox_inches="tight")
    fig.savefig(out_path_without_ext + ".svg", bbox_inches="tight")


# -----------------------------------------------------------------------------
# TCM-ES embedding generation for repeated models
# -----------------------------------------------------------------------------

class TCMEmbeddingGenerator:
    """Generate symptom-pattern and formula embeddings from a trained TCM-ES model."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        symptom_list_dir: str = "core/standard_TCM_entities/symptom_list.pkl",
        symptom_semantic_dir: str = "core/standard_TCM_entities/symptom_semantic_encodings.pkl",
        herb_list_dir: str = "core/standard_TCM_entities/herb_list.pkl",
    ):
        self.model_dir = model_dir
        self.device = device
        self.symptom_list_dir = symptom_list_dir
        self.symptom_semantic_dir = symptom_semantic_dir
        self.herb_list_dir = herb_list_dir

        self.n_herb_seq = 30
        self.n_sym_seq = 40
        self.max_herb_len = self.n_herb_seq + 1
        self.max_sym_len = self.n_sym_seq + 1

        self._load_model()
        self._load_vocabularies()

    def _load_model(self):
        self.model = torch.load(
            self.model_dir,
            map_location=self.device,
            weights_only=False,
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_vocabularies(self):
        with open(self.herb_list_dir, "rb") as fp:
            self.herb_list = pickle.load(fp)
        with open(self.symptom_list_dir, "rb") as fp:
            self.symptom_list = pickle.load(fp)
        with open(self.symptom_semantic_dir, "rb") as fp:
            self.symptom_semantics = pickle.load(fp)

    def _generate_herb_data(self, herb_id_seq: List[int], in_or_out: str = "input"):
        if in_or_out == "input":
            seq = [len(self.herb_list) + 1] + herb_id_seq
        else:
            seq = herb_id_seq + [len(self.herb_list) + 2]
        return seq + [len(self.herb_list)] * (self.max_herb_len - len(seq))

    def _generate_symptom_data_from_ids(
        self,
        symptom_id_seq: List[int],
        in_or_out: str = "input",
    ):
        if in_or_out == "input":
            seq = [len(self.symptom_list) + 1] + symptom_id_seq
        else:
            seq = symptom_id_seq + [len(self.symptom_list) + 2]
        seq = seq + [len(self.symptom_list)] * (self.max_sym_len - len(seq))
        return [self.symptom_semantics[s] for s in seq]

    def generate_symptom_embedding(self, case_symptoms: List[List[str]]) -> np.ndarray:
        """Generate symptom-pattern embeddings from standardized symptom lists."""
        herb_inputs = []
        symptom_inputs = []

        for symptoms in case_symptoms:
            symptom_ids = [
                self.symptom_list.index(s)
                for s in symptoms
                if s in self.symptom_list
            ]
            symptom_inputs.append(
                self._generate_symptom_data_from_ids(symptom_ids, "input")
            )
            herb_inputs.append(self._generate_herb_data([], "input"))

        herb_tensor = torch.IntTensor(np.asarray(herb_inputs)).to(self.device)
        symptom_tensor = torch.FloatTensor(np.asarray(symptom_inputs)).to(self.device)

        with torch.no_grad():
            _, _, z1, _, _ = self.model(symptom_tensor, herb_tensor)

        return z1.cpu().numpy().astype(np.float32)

    def generate_herb_embedding(self, case_herbs: List[List[str]]) -> np.ndarray:
        """Generate formula embeddings from standardized herb lists."""
        herb_inputs = []
        symptom_inputs = []

        for herbs in case_herbs:
            herb_ids = [
                self.herb_list.index(h)
                for h in herbs
                if h in self.herb_list
            ]
            symptom_inputs.append(
                self._generate_symptom_data_from_ids([], "input")
            )
            herb_inputs.append(self._generate_herb_data(herb_ids, "input"))

        herb_tensor = torch.IntTensor(np.asarray(herb_inputs)).to(self.device)
        symptom_tensor = torch.FloatTensor(np.asarray(symptom_inputs)).to(self.device)

        with torch.no_grad():
            _, _, _, z2, _ = self.model(symptom_tensor, herb_tensor)

        return z2.cpu().numpy().astype(np.float32)


def find_repeat_models(repeat_model_root: str, model_pattern: str) -> List[Tuple[str, str]]:
    """Find one selected checkpoint under each repeat_* directory."""
    repeat_dirs = sorted(glob.glob(os.path.join(repeat_model_root, "repeat_*")))
    if len(repeat_dirs) == 0:
        raise FileNotFoundError(
            f"No repeat_* directories found under: {repeat_model_root}"
        )

    repeat_models = []
    for repeat_dir in repeat_dirs:
        repeat_name = os.path.basename(repeat_dir)
        model_files = sorted(glob.glob(os.path.join(repeat_dir, model_pattern)))
        if len(model_files) != 1:
            raise ValueError(
                f"Expected exactly one checkpoint in {repeat_dir} matching "
                f"{model_pattern}, found {len(model_files)}."
            )
        repeat_models.append((repeat_name, model_files[0]))

    return repeat_models


def select_formula_ids(
    formula_df: pd.DataFrame,
    min_symptoms: int = 3,
    min_herbs: int = 3,
    max_symptoms: int = 40,
    max_herbs: int = 30,
) -> List[int]:
    """Select model-eligible formula records using sequence-length constraints."""
    selected = []

    for i in range(len(formula_df)):
        symptoms = formula_df.iloc[i]["Symptoms"]
        herbs = formula_df.iloc[i]["Herbs"]

        n_symptoms = len(symptoms) if isinstance(symptoms, list) else 0
        n_herbs = len(herbs) if isinstance(herbs, list) else 0

        if (
            min_symptoms <= n_symptoms <= max_symptoms
            and min_herbs <= n_herbs <= max_herbs
        ):
            selected.append(i)

    return selected


def _repeat_embedding_files(out_dir: str) -> Tuple[str, str, str]:
    return (
        os.path.join(out_dir, "selected_formula_ids(line_number).pkl"),
        os.path.join(out_dir, "selected_formula_symptom_embeddings.pkl"),
        os.path.join(out_dir, "selected_formula_herb_embeddings.pkl"),
    )


def generate_embeddings_for_one_model(
    model_path: str,
    repeat_name: str,
    formula_df: pd.DataFrame,
    selected_formula_ids: List[int],
    out_dir: str,
    device: str,
    batch_size: int,
    symptom_list: str,
    symptom_semantic: str,
    herb_list: str,
    overwrite: bool = False,
):
    """Generate row-aligned formula-record embeddings for one repeated model."""
    ensure_dir(out_dir)
    ids_path, symptom_path, herb_path = _repeat_embedding_files(out_dir)

    if all(os.path.exists(p) for p in (ids_path, symptom_path, herb_path)) and not overwrite:
        existing_ids = [int(x) for x in load_pickle(ids_path)]
        if existing_ids != [int(x) for x in selected_formula_ids]:
            raise ValueError(
                f"Existing embeddings for {repeat_name} use different formula IDs. "
                "Use --overwrite-embeddings to regenerate them."
            )
        print(f"[{repeat_name}] embeddings already exist. Skipped.")
        return

    symptom_inputs = formula_df.iloc[selected_formula_ids]["Symptoms"].tolist()
    herb_inputs = formula_df.iloc[selected_formula_ids]["Herbs"].tolist()

    generator = TCMEmbeddingGenerator(
        model_dir=model_path,
        device=device,
        symptom_list_dir=symptom_list,
        symptom_semantic_dir=symptom_semantic,
        herb_list_dir=herb_list,
    )

    symptom_chunks = []
    herb_chunks = []
    n_records = len(selected_formula_ids)

    for start in range(0, n_records, batch_size):
        end = min(start + batch_size, n_records)
        print(f"[{repeat_name}] embeddings: {start}-{end}", flush=True)
        symptom_chunks.append(
            generator.generate_symptom_embedding(symptom_inputs[start:end])
        )
        herb_chunks.append(
            generator.generate_herb_embedding(herb_inputs[start:end])
        )

    symptom_embeddings = np.vstack(symptom_chunks).astype(np.float32)
    herb_embeddings = np.vstack(herb_chunks).astype(np.float32)

    save_pickle(selected_formula_ids, ids_path)
    save_pickle(symptom_embeddings, symptom_path)
    save_pickle(herb_embeddings, herb_path)

    print(f"[{repeat_name}] saved to: {out_dir}")
    print(f"[{repeat_name}] symptom embeddings: {symptom_embeddings.shape}")
    print(f"[{repeat_name}] formula embeddings: {herb_embeddings.shape}")


def generate_repeat_embeddings(args):
    """Generate formula-record embeddings for all repeated model checkpoints."""
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    formula_df = pd.read_pickle(args.formula_data).reset_index(drop=True)

    if args.selected_formula_ids is None:
        selected_formula_ids = select_formula_ids(
            formula_df,
            min_symptoms=args.min_symptoms,
            min_herbs=args.min_herbs,
            max_symptoms=args.max_symptoms,
            max_herbs=args.max_herbs,
        )
    else:
        selected_formula_ids = [
            int(x) for x in load_pickle(args.selected_formula_ids)
        ]

    if len(selected_formula_ids) == 0:
        raise ValueError("No eligible formula records were selected.")

    print("Selected formula records:", len(selected_formula_ids))

    repeat_models = find_repeat_models(args.repeat_model_root, args.model_pattern)
    print("Repeated model checkpoints:")
    for repeat_name, model_path in repeat_models:
        print(" ", repeat_name, model_path)

    for repeat_name, model_path in repeat_models:
        repeat_out_dir = os.path.join(args.embedding_out_dir, repeat_name)
        generate_embeddings_for_one_model(
            model_path=model_path,
            repeat_name=repeat_name,
            formula_df=formula_df,
            selected_formula_ids=selected_formula_ids,
            out_dir=repeat_out_dir,
            device=device,
            batch_size=args.batch_size,
            symptom_list=args.symptom_list,
            symptom_semantic=args.symptom_semantic,
            herb_list=args.herb_list,
            overwrite=args.overwrite_embeddings,
        )


# -----------------------------------------------------------------------------
# Load repeated embeddings
# -----------------------------------------------------------------------------

def load_repeated_embeddings(embedding_out_dir: str, embedding_type: str):
    """Load row-aligned embeddings from all repeat_* directories."""
    repeat_dirs = sorted(glob.glob(os.path.join(embedding_out_dir, "repeat_*")))
    if len(repeat_dirs) == 0:
        raise FileNotFoundError(
            f"No repeat_* embedding directories found under: {embedding_out_dir}"
        )

    repeat_names = []
    selected_ids_ref = None
    z_list = []

    for repeat_dir in repeat_dirs:
        repeat_name = os.path.basename(repeat_dir)
        ids_path, symptom_path, herb_path = _repeat_embedding_files(repeat_dir)

        selected_ids = [int(x) for x in load_pickle(ids_path)]
        z_symptom = np.asarray(load_pickle(symptom_path), dtype=np.float32)
        z_herb = np.asarray(load_pickle(herb_path), dtype=np.float32)

        if selected_ids_ref is None:
            selected_ids_ref = selected_ids
        elif selected_ids != selected_ids_ref:
            raise ValueError(
                f"Selected formula IDs are not aligned for {repeat_name}."
            )

        if z_symptom.shape != z_herb.shape:
            raise ValueError(
                f"Symptom and formula embedding shapes differ for {repeat_name}: "
                f"{z_symptom.shape} vs {z_herb.shape}."
            )

        if embedding_type == "symptom":
            z = z_symptom
        elif embedding_type == "herb":
            z = z_herb
        elif embedding_type == "average":
            z = 0.5 * (z_symptom + z_herb)
        else:
            raise ValueError("embedding_type must be symptom, herb, or average.")

        repeat_names.append(repeat_name)
        z_list.append(z.astype(np.float32))

    shapes = [z.shape for z in z_list]
    if len(set(shapes)) != 1:
        raise ValueError(f"Repeated embeddings have inconsistent shapes: {shapes}")

    print("Loaded repeats:", repeat_names)
    print("Embedding shape per repeat:", z_list[0].shape)

    return repeat_names, selected_ids_ref, z_list


# -----------------------------------------------------------------------------
# Global pairwise-similarity stability
# -----------------------------------------------------------------------------

def sample_record_pairs(n_records: int, n_pairs: int, seed: int) -> np.ndarray:
    """Sample exactly n_pairs unique unordered non-self record pairs."""
    n_records = int(n_records)
    n_pairs = int(n_pairs)

    if n_records < 2:
        raise ValueError("At least two records are required to sample pairs.")

    max_unique_pairs = n_records * (n_records - 1) // 2
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive.")
    if n_pairs > max_unique_pairs:
        raise ValueError(
            f"Requested {n_pairs} pairs, but only {max_unique_pairs} unique "
            "non-self pairs are available."
        )

    rng = np.random.default_rng(seed)
    pair_codes = set()

    while len(pair_codes) < n_pairs:
        remaining = n_pairs - len(pair_codes)
        batch_size = max(1024, int(np.ceil(remaining * 1.5)))

        i = rng.integers(0, n_records, size=batch_size)
        j = rng.integers(0, n_records, size=batch_size)
        keep = i != j
        i = i[keep]
        j = j[keep]

        lo = np.minimum(i, j)
        hi = np.maximum(i, j)
        codes = lo.astype(np.int64) * n_records + hi.astype(np.int64)
        pair_codes.update(codes.tolist())

    codes = np.asarray(sorted(pair_codes)[:n_pairs], dtype=np.int64)
    pairs = np.column_stack((codes // n_records, codes % n_records)).astype(np.int64)
    return pairs


def analyze_global_similarity_stability(
    repeat_names: List[str],
    z_list: List[np.ndarray],
    out_dir: str,
    n_pairs: int,
    seed: int,
):
    """Compare global cosine-similarity structure across repeated models."""
    ensure_dir(out_dir)

    pairs = sample_record_pairs(
        n_records=z_list[0].shape[0],
        n_pairs=n_pairs,
        seed=seed,
    )

    pair_table = pd.DataFrame({
        "record_i": pairs[:, 0],
        "record_j": pairs[:, 1],
    })
    pair_table.to_csv(
        os.path.join(out_dir, "sampled_record_pairs.csv"),
        index=False,
    )

    similarity_vectors = []
    for z in z_list:
        z_norm = l2_normalize(z)
        similarities = np.sum(
            z_norm[pairs[:, 0]] * z_norm[pairs[:, 1]],
            axis=1,
        )
        similarity_vectors.append(similarities.astype(np.float32))

    n_repeat = len(repeat_names)
    spearman_mat = np.eye(n_repeat, dtype=float)
    rows = []

    for a, b in combinations(range(n_repeat), 2):
        spearman = float(
            spearmanr(similarity_vectors[a], similarity_vectors[b]).correlation
        )
        spearman_mat[a, b] = spearman_mat[b, a] = spearman
        rows.append({
            "repeat_a": repeat_names[a],
            "repeat_b": repeat_names[b],
            "n_sampled_pairs": int(len(pairs)),
            "spearman_similarity_correlation": spearman,
        })

    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(
        os.path.join(out_dir, "repeat_pair_global_similarity_correlation.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    values = pair_df["spearman_similarity_correlation"].to_numpy(dtype=float)
    summary_df = pd.DataFrame([{
        "metric": "spearman_similarity_correlation",
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_repeat_pairs": int(len(values)),
        "n_sampled_record_pairs": int(len(pairs)),
    }])
    summary_df.to_csv(
        os.path.join(out_dir, "global_similarity_stability_summary.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    plot_repeat_correlation_heatmap(
        spearman_mat,
        repeat_names,
        os.path.join(out_dir, "fig_repeat_pair_spearman_similarity_heatmap"),
    )
    plot_spearman_correlation_distribution(
        pair_df,
        os.path.join(out_dir, "fig_global_similarity_correlation_distribution"),
    )

    return pair_df, summary_df, spearman_mat


def plot_repeat_correlation_heatmap(
    corr_mat: np.ndarray,
    repeat_names: List[str],
    out_path: str,
):
    """Plot repeat-pair Spearman correlation heatmap."""
    fig, ax = plt.subplots(figsize=(5.2, 4.5))

    vmin = max(0.0, float(np.nanmin(corr_mat)) - 0.02)
    im = ax.imshow(corr_mat, vmin=vmin, vmax=1.0)

    ax.set_xticks(np.arange(len(repeat_names)))
    ax.set_yticks(np.arange(len(repeat_names)))
    ax.set_xticklabels(repeat_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(repeat_names, fontsize=8)
    ax.set_title("Repeat-pair global similarity stability", fontsize=11, pad=10)

    for i in range(corr_mat.shape[0]):
        for j in range(corr_mat.shape[1]):
            ax.text(
                j,
                i,
                f"{corr_mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if corr_mat[i, j] > 0.75 else "black",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman correlation", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_spearman_correlation_distribution(pair_df: pd.DataFrame, out_path: str):
    """Plot the distribution of repeat-pair Spearman correlations."""
    y = pair_df["spearman_similarity_correlation"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    bp = ax.boxplot([y], tick_labels=["Spearman"], showfliers=False, patch_artist=True)
    bp["boxes"][0].set_facecolor("white")
    bp["boxes"][0].set_linewidth(1.2)

    rng = np.random.default_rng(1)
    x = 1 + rng.normal(0, 0.035, size=len(y))
    ax.scatter(x, y, s=18, alpha=0.65, linewidths=0.3, edgecolors="black")

    ax.set_ylabel("Correlation")
    ax.set_title("Global similarity stability", fontsize=11)
    ax.set_ylim(max(0.0, float(np.min(y)) - 0.05), 1.02)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Local nearest-neighbour stability
# -----------------------------------------------------------------------------

def topk_neighbors(z: np.ndarray, max_k: int, block_size: int = 512) -> np.ndarray:
    """Compute cosine nearest neighbours for each record."""
    z = l2_normalize(z)
    n_records = z.shape[0]

    if max_k <= 0 or max_k >= n_records:
        raise ValueError(
            f"max_k must be between 1 and n_records - 1; got {max_k} "
            f"for {n_records} records."
        )

    out = np.zeros((n_records, max_k), dtype=np.int32)
    all_ids = np.arange(n_records)

    for start in range(0, n_records, block_size):
        end = min(start + block_size, n_records)
        similarities = z[start:end] @ z.T
        row_ids = all_ids[start:end]
        similarities[np.arange(end - start), row_ids] = -np.inf

        part = np.argpartition(
            -similarities,
            kth=max_k - 1,
            axis=1,
        )[:, :max_k]
        order = np.argsort(
            -similarities[np.arange(end - start)[:, None], part],
            axis=1,
        )
        out[start:end] = part[np.arange(end - start)[:, None], order]

    return out


def expected_random_jaccard(n_records: int, k: int) -> float:
    """Approximate expected Jaccard overlap between two random k-NN sets."""
    candidate_count = int(n_records) - 1
    k = int(k)

    if candidate_count <= 0 or k <= 0:
        return np.nan
    if k >= candidate_count:
        return 1.0

    return float(k / (2 * candidate_count - k))


def _mean_jaccard_for_record(nn_a: np.ndarray, nn_b: np.ndarray, k: int) -> float:
    set_a = set(nn_a[:k])
    set_b = set(nn_b[:k])
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else np.nan


def analyze_nn_stability(
    repeat_names: List[str],
    z_list: List[np.ndarray],
    out_dir: str,
    k_values: Sequence[int],
    local_stability_k: int,
    block_size: int,
):
    """Evaluate nearest-neighbour consistency across repeated models."""
    ensure_dir(out_dir)

    k_values = sorted(set(int(k) for k in k_values))
    max_k = max(max(k_values), int(local_stability_k))
    n_repeat = len(repeat_names)
    n_records = z_list[0].shape[0]

    if n_repeat < 2:
        raise ValueError("At least two repeated models are required for stability analysis.")
    if max_k >= n_records:
        raise ValueError(
            f"Largest neighbourhood size ({max_k}) must be smaller than "
            f"the number of records ({n_records})."
        )

    nn_list = []
    for repeat_name, z in zip(repeat_names, z_list):
        print(f"[NN] computing top-{max_k} neighbours for {repeat_name}", flush=True)
        nn_list.append(topk_neighbors(z, max_k=max_k, block_size=block_size))

    rows = []
    per_record_sum = np.zeros(n_records, dtype=float)
    per_record_count = np.zeros(n_records, dtype=int)

    for a, b in combinations(range(n_repeat), 2):
        for k in k_values:
            jaccards = np.asarray([
                _mean_jaccard_for_record(nn_list[a][i], nn_list[b][i], k)
                for i in range(n_records)
            ], dtype=float)

            rows.append({
                "repeat_a": repeat_names[a],
                "repeat_b": repeat_names[b],
                "k": int(k),
                "mean_nn_jaccard": float(np.nanmean(jaccards)),
                "std_nn_jaccard_across_records": float(
                    np.nanstd(jaccards, ddof=1)
                ),
            })

        local_values = np.asarray([
            _mean_jaccard_for_record(
                nn_list[a][i],
                nn_list[b][i],
                int(local_stability_k),
            )
            for i in range(n_records)
        ], dtype=float)
        valid = np.isfinite(local_values)
        per_record_sum[valid] += local_values[valid]
        per_record_count[valid] += 1

    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(
        os.path.join(out_dir, "repeat_pair_nn_stability.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    summary_df = pair_df.groupby("k", as_index=False).agg(
        mean_nn_jaccard=("mean_nn_jaccard", "mean"),
        std_nn_jaccard=("mean_nn_jaccard", "std"),
        n_repeat_pairs=("mean_nn_jaccard", "size"),
    )
    summary_df["random_baseline_jaccard"] = summary_df["k"].apply(
        lambda k: expected_random_jaccard(n_records=n_records, k=int(k))
    )
    summary_df.to_csv(
        os.path.join(out_dir, "nn_stability_summary_by_k.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    per_record_local_stability = np.divide(
        per_record_sum,
        per_record_count,
        out=np.full(n_records, np.nan, dtype=float),
        where=per_record_count > 0,
    )
    pd.DataFrame({
        "record_index_in_selected_data": np.arange(n_records),
        f"mean_nn_jaccard_at_{local_stability_k}": per_record_local_stability,
        "n_repeat_pairs": per_record_count,
    }).to_csv(
        os.path.join(
            out_dir,
            f"per_record_local_stability_nn_jaccard_at_{local_stability_k}.csv",
        ),
        index=False,
        encoding="utf_8_sig",
    )

    plot_nn_jaccard_curve(
        summary_df,
        os.path.join(out_dir, "fig_nn_jaccard_by_k"),
    )

    return pair_df, summary_df, per_record_local_stability


def plot_nn_jaccard_curve(summary_df: pd.DataFrame, out_path: str):
    """Plot NN-Jaccard stability across neighbourhood sizes."""
    df = summary_df.sort_values("k")
    x = df["k"].to_numpy(dtype=int)
    y = df["mean_nn_jaccard"].to_numpy(dtype=float)
    sd = df["std_nn_jaccard"].fillna(0).to_numpy(dtype=float)
    baseline = df["random_baseline_jaccard"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.errorbar(
        x,
        y,
        yerr=sd,
        marker="o",
        linewidth=1.8,
        markersize=5.5,
        capsize=3.5,
        label="Repeated TCM-ES",
    )
    ax.plot(
        x,
        baseline,
        linestyle="--",
        linewidth=1.4,
        marker="s",
        markersize=4.0,
        label="Random baseline",
    )

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("Neighbourhood size k")
    ax.set_ylabel("Mean NN-Jaccard@k")
    ax.set_title("Nearest-neighbour stability", fontsize=11)

    ymax = max(float(np.max(y + sd)), float(np.max(baseline)))
    ax.set_ylim(0, min(1.0, ymax + 0.08))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Local stability on the primary-model PCA space
# -----------------------------------------------------------------------------

def _coordinate_key(df: pd.DataFrame) -> str:
    if "selected_formula_row_id" in df.columns:
        return "selected_formula_row_id"
    if "formula_row_id" in df.columns:
        return "formula_row_id"
    raise ValueError(
        "PCA coordinate files must contain selected_formula_row_id or formula_row_id."
    )


def load_primary_average_pca_coordinates(
    primary_pca_analysis_dir: str,
    symptom_coord_file: str = "pca/formula_symptom_embedding_pca_coords_scaled.csv",
    herb_coord_file: str = "pca/formula_herb_embedding_pca_coords_scaled.csv",
) -> pd.DataFrame:
    """Load symptom/formula PCA coordinates and return their record-wise average."""
    symptom_path = os.path.join(primary_pca_analysis_dir, symptom_coord_file)
    herb_path = os.path.join(primary_pca_analysis_dir, herb_coord_file)

    if not os.path.exists(symptom_path):
        raise FileNotFoundError(f"Cannot find symptom PCA coordinates: {symptom_path}")
    if not os.path.exists(herb_path):
        raise FileNotFoundError(f"Cannot find formula PCA coordinates: {herb_path}")

    symptom_df = pd.read_csv(symptom_path)
    herb_df = pd.read_csv(herb_path)
    symptom_key = _coordinate_key(symptom_df)
    herb_key = _coordinate_key(herb_df)

    required_pcs = ["PC1", "PC2", "PC3"]
    for pc in required_pcs:
        if pc not in symptom_df.columns or pc not in herb_df.columns:
            raise ValueError(
                f"Both primary-model coordinate files must contain {pc}."
            )

    symptom_part = symptom_df[[symptom_key] + required_pcs].copy()
    herb_part = herb_df[[herb_key] + required_pcs].copy()
    symptom_part = symptom_part.rename(columns={symptom_key: "selected_formula_row_id"})
    herb_part = herb_part.rename(columns={herb_key: "selected_formula_row_id"})

    symptom_part["selected_formula_row_id"] = symptom_part["selected_formula_row_id"].astype(int)
    herb_part["selected_formula_row_id"] = herb_part["selected_formula_row_id"].astype(int)

    merged = symptom_part.merge(
        herb_part,
        on="selected_formula_row_id",
        suffixes=("_symptom", "_herb"),
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(symptom_part) or len(merged) != len(herb_part):
        raise ValueError(
            "Primary-model symptom and formula PCA coordinate files are not row-aligned."
        )

    out = pd.DataFrame({
        "selected_formula_row_id": merged["selected_formula_row_id"].astype(int),
    })
    for pc in required_pcs:
        out[pc] = 0.5 * (
            merged[f"{pc}_symptom"].to_numpy(dtype=float)
            + merged[f"{pc}_herb"].to_numpy(dtype=float)
        )

    return out


def merge_stability_with_primary_pca(
    selected_formula_ids: List[int],
    per_record_local_stability: np.ndarray,
    primary_pca_analysis_dir: str,
    primary_pca_symptom_coord_file: str,
    primary_pca_herb_coord_file: str,
) -> pd.DataFrame:
    """Merge repeated-training local stability with primary-model PCA coordinates."""
    coord_df = load_primary_average_pca_coordinates(
        primary_pca_analysis_dir,
        symptom_coord_file=primary_pca_symptom_coord_file,
        herb_coord_file=primary_pca_herb_coord_file,
    )

    stability_df = pd.DataFrame({
        "selected_formula_row_id": [int(x) for x in selected_formula_ids],
        "local_stability": np.asarray(per_record_local_stability, dtype=float),
    })

    merged = coord_df.merge(
        stability_df,
        on="selected_formula_row_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(stability_df):
        raise ValueError(
            "Primary-model PCA coordinates do not cover all formula records used "
            "in the repeated-training stability analysis."
        )

    return merged


def local_stability_color_range(merged: pd.DataFrame) -> Tuple[float, float]:
    """Use the 5th-95th percentile range for visualization."""
    values = merged["local_stability"].to_numpy(dtype=float)
    vmin = float(np.nanpercentile(values, 5))
    vmax = float(np.nanpercentile(values, 95))

    if vmin == vmax:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))

    return vmin, vmax


def plot_local_stability_on_primary_pca_2d(
    merged: pd.DataFrame,
    fig_dir: str,
    vmin: float,
    vmax: float,
):
    """Plot local stability on primary-model PC1-PC2."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sc = ax.scatter(
        merged["PC1"],
        merged["PC2"],
        c=merged["local_stability"],
        s=12,
        alpha=0.72,
        linewidths=0,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label("Local stability", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Primary-model PC1")
    ax.set_ylabel("Primary-model PC2")
    ax.set_title("Repeated-training local stability", fontsize=11)
    ax.grid(linestyle="--", linewidth=0.45, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(
        fig,
        os.path.join(fig_dir, "fig_local_stability_on_primary_pca_PC1_PC2"),
    )
    plt.close(fig)


def plot_local_stability_on_primary_pca_3d(
    merged: pd.DataFrame,
    fig_dir: str,
    vmin: float,
    vmax: float,
):
    """Plot local stability on primary-model PC1-PC2-PC3."""
    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        merged["PC1"],
        merged["PC2"],
        merged["PC3"],
        c=merged["local_stability"],
        s=10,
        alpha=0.72,
        linewidths=0,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label("Local stability", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Primary-model PC1", labelpad=8)
    ax.set_ylabel("Primary-model PC2", labelpad=8)
    ax.set_zlabel("Primary-model PC3", labelpad=8)
    ax.set_title("Repeated-training local stability", fontsize=11)
    ax.view_init(elev=22, azim=38)

    save_figure(
        fig,
        os.path.join(fig_dir, "fig_local_stability_on_primary_pca_PC1_PC2_PC3"),
    )
    plt.close(fig)


def plot_local_stability_on_primary_pca(
    selected_formula_ids: List[int],
    per_record_local_stability: np.ndarray,
    primary_pca_analysis_dir: str,
    primary_pca_symptom_coord_file: str,
    primary_pca_herb_coord_file: str,
    out_dir: str,
) -> pd.DataFrame:
    """Map per-record local stability to the primary-model PCA space."""
    fig_dir = os.path.join(out_dir, "primary_pca_visualization")
    ensure_dir(fig_dir)

    merged = merge_stability_with_primary_pca(
        selected_formula_ids=selected_formula_ids,
        per_record_local_stability=per_record_local_stability,
        primary_pca_analysis_dir=primary_pca_analysis_dir,
        primary_pca_symptom_coord_file=primary_pca_symptom_coord_file,
        primary_pca_herb_coord_file=primary_pca_herb_coord_file,
    )

    merged.to_csv(
        os.path.join(fig_dir, "local_stability_on_primary_pca_plot_data.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    vmin, vmax = local_stability_color_range(merged)
    plot_local_stability_on_primary_pca_2d(merged, fig_dir, vmin, vmax)
    plot_local_stability_on_primary_pca_3d(merged, fig_dir, vmin, vmax)

    return merged


# -----------------------------------------------------------------------------
# Repeated-training robustness workflow
# -----------------------------------------------------------------------------

def run_stability_analysis(args):
    """Run global and local repeated-training robustness analyses."""
    repeat_names, selected_formula_ids, z_list = load_repeated_embeddings(
        args.embedding_out_dir,
        args.embedding_type,
    )

    stability_dir = os.path.join(args.out_dir, "stability")
    ensure_dir(stability_dir)

    analyze_global_similarity_stability(
        repeat_names=repeat_names,
        z_list=z_list,
        out_dir=stability_dir,
        n_pairs=args.n_pairs,
        seed=args.seed,
    )

    _, _, per_record_local_stability = analyze_nn_stability(
        repeat_names=repeat_names,
        z_list=z_list,
        out_dir=stability_dir,
        k_values=parse_k_values(args.neighbor_k_values),
        local_stability_k=args.local_stability_k,
        block_size=args.nn_block_size,
    )

    plot_local_stability_on_primary_pca(
        selected_formula_ids=selected_formula_ids,
        per_record_local_stability=per_record_local_stability,
        primary_pca_analysis_dir=args.primary_pca_analysis_dir,
        primary_pca_symptom_coord_file=args.primary_pca_symptom_coord_file,
        primary_pca_herb_coord_file=args.primary_pca_herb_coord_file,
        out_dir=args.out_dir,
    )


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate TCM-ES embedding robustness across repeated training runs."
    )

    parser.add_argument(
        "--mode",
        choices=["all", "generate", "analyze"],
        default="all",
        help="Generate repeated embeddings, run analysis, or perform both.",
    )

    parser.add_argument(
        "--formula-data",
        default="data/TCM_formulas/TCM_formula_data_example_2000.pkl",
        help="Formula dataframe used to generate repeated-model embeddings.",
    )
    parser.add_argument(
        "--repeat-model-root",
        default="core/trained_model_repeated",
        help="Directory containing repeat_* model folders.",
    )
    parser.add_argument(
        "--model-pattern",
        default="model_epoch_*.pkl",
        help="Checkpoint filename pattern within each repeat_* folder.",
    )
    parser.add_argument(
        "--selected-formula-ids",
        default=None,
        help="Optional pickle file containing formula row IDs to embed.",
    )
    parser.add_argument(
        "--embedding-out-dir",
        default="results/embeddings/TCM_embeddings_repeated",
        help="Output directory for repeated-model embeddings.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/TCM_embedding_repeated_stability",
        help="Output directory for robustness results.",
    )

    parser.add_argument(
        "--symptom-list",
        default="core/standard_TCM_entities/symptom_list.pkl",
    )
    parser.add_argument(
        "--symptom-semantic",
        default="core/standard_TCM_entities/symptom_semantic_encodings.pkl",
    )
    parser.add_argument(
        "--herb-list",
        default="core/standard_TCM_entities/herb_list.pkl",
    )

    parser.add_argument("--min-symptoms", type=int, default=3)
    parser.add_argument("--min-herbs", type=int, default=3)
    parser.add_argument("--max-symptoms", type=int, default=40)
    parser.add_argument("--max-herbs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite-embeddings", action="store_true")

    parser.add_argument(
        "--embedding-type",
        choices=["symptom", "herb", "average"],
        default="average",
        help=(
            "Record representation used for robustness analysis. "
            "The manuscript setting is 'average'."
        ),
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=200000,
        help="Number of unique non-self record pairs for global stability.",
    )
    parser.add_argument(
        "--neighbor-k-values",
        default="5,10,20,50,100,200",
        help="Neighbourhood sizes used for the NN-Jaccard sensitivity curve.",
    )
    parser.add_argument(
        "--local-stability-k",
        type=int,
        default=20,
        help="Neighbourhood size used for per-record local stability.",
    )
    parser.add_argument("--nn-block-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--primary-pca-analysis-dir",
        default="results/TCM_embedding_analysis/original(average)",
        help="Primary-model PCA analysis directory from Step 3.1.",
    )
    parser.add_argument(
        "--primary-pca-symptom-coord-file",
        default="pca/formula_symptom_embedding_pca_coords_scaled.csv",
        help="Symptom-pattern PCA coordinates under the primary analysis directory.",
    )
    parser.add_argument(
        "--primary-pca-herb-coord-file",
        default="pca/formula_herb_embedding_pca_coords_scaled.csv",
        help="Formula PCA coordinates under the primary analysis directory.",
    )

    return parser.parse_args(argv)


def main():
    args = parse_args()
    set_plot_style()
    ensure_dir(args.out_dir)

    if args.mode in ["all", "generate"]:
        generate_repeat_embeddings(args)

    if args.mode in ["all", "analyze"]:
        run_stability_analysis(args)


if __name__ == "__main__":
    main()
