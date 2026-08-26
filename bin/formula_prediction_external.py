#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
External TCM-PD formula prediction using frozen TCM-ES symptom-pattern embeddings.

The official TCM-PD train/test split is retained. Ten percent of the mapped
training records are used for validation. External test records with high
symptom- and herb-set overlap with the internal TCM-ES training corpus can be
excluded using the manuscript leakage-control rule.

Two downstream readouts are evaluated:
    1. TCM-ES MLP head using standardized [z, z^2] features.
    2. TCM-ES ridge head using the same standardized [z, z^2] features.

Only frozen symptom-pattern embeddings are used as model-derived inputs.
Formula/herb-side embeddings are not used for prediction.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


MODEL_MLP = "TCM-ES MLP head"
MODEL_RIDGE = "TCM-ES ridge head"


# =============================================================================
# 1. Basic utilities
# =============================================================================

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_pickle(path: str):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def save_pickle(obj, path: str) -> None:
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def require_file(path: str, name: str) -> str:
    if path and os.path.exists(path):
        return path
    raise FileNotFoundError(f"Missing {name}: {path}")


def unique_list(items: Iterable) -> List:
    out, seen = [], set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.lower().startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA requested but unavailable. Falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def set_plot_style() -> None:
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig, out_stem: str) -> None:
    fig.savefig(out_stem + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    fig.savefig(out_stem + ".svg", bbox_inches="tight")


# =============================================================================
# 2. External data and evaluation split
# =============================================================================

def load_external_inputs(args) -> Dict:
    symptom_embeddings_path = os.path.join(
        args.external_embedding_dir,
        "external_all_valid_symptom_embeddings.pkl",
    )

    paths = {
        "external_formula_data": require_file(
            args.external_formula_data,
            "external all-valid formula data",
        ),
        "external_symptom_embeddings": require_file(
            symptom_embeddings_path,
            "external all-valid symptom embeddings",
        ),
        "herb_list": require_file(args.herb_list, "herb list"),
    }

    formula_df = pd.read_pickle(paths["external_formula_data"]).reset_index(drop=True)
    herb_list = list(load_pickle(paths["herb_list"]))
    symptom_embeddings = np.asarray(
        load_pickle(paths["external_symptom_embeddings"]),
        dtype=np.float32,
    )

    if "Train_test_split" not in formula_df.columns:
        raise ValueError("External dataframe must contain column: Train_test_split")
    if symptom_embeddings.ndim != 2:
        raise ValueError("External symptom embeddings must be a 2D array.")
    if symptom_embeddings.shape[0] != len(formula_df):
        raise ValueError(
            f"Embedding rows {symptom_embeddings.shape[0]} != external records {len(formula_df)}"
        )

    return {
        "paths": paths,
        "formula_df": formula_df,
        "herb_list": herb_list,
        "X": symptom_embeddings,
    }


def build_herb_label_matrix(
    formula_df: pd.DataFrame,
    herb_list: List[str],
) -> np.ndarray:
    """Build the observed-herb multilabel matrix for TCM-PD records."""
    herb2id = {herb: i for i, herb in enumerate(herb_list)}
    Y = np.zeros((len(formula_df), len(herb_list)), dtype=np.uint8)

    for i, herbs in enumerate(formula_df["Herbs"]):
        for herb in unique_list(herbs):
            j = herb2id.get(herb)
            if j is not None:
                Y[i, j] = 1

    return Y


def build_external_train_val_test_indices(
    formula_df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Retain the official test split and sample validation records from official training data."""
    split_labels = formula_df["Train_test_split"].astype(str).str.strip().str.lower()
    train_pool = split_labels[split_labels == "train"].index.tolist()
    test_idx = sorted(split_labels[split_labels == "test"].index.tolist())

    if not train_pool:
        raise ValueError("No external training records found.")
    if not test_idx:
        raise ValueError("No external test records found.")

    val_size = max(1, int(len(train_pool) * float(val_ratio)))
    if val_size >= len(train_pool):
        raise ValueError("Validation subset must be smaller than the external training pool.")

    rng = random.Random(int(seed))
    val_idx = sorted(rng.sample(train_pool, val_size))
    val_set = set(val_idx)
    train_idx = sorted(i for i in train_pool if i not in val_set)
    return train_idx, val_idx, test_idx


def jaccard_similarity(items_a, items_b) -> float:
    a = set(items_a)
    b = set(items_b)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def find_best_reference_training_match(
    external_symptoms,
    external_herbs,
    reference_formula_df: pd.DataFrame,
    reference_train_idx: List[int],
    symptom_threshold: float,
    herb_threshold: float,
) -> Dict:
    """
    Match one external test record to the internal training record with the
    largest mean symptom/herb Jaccard similarity.
    """
    best_ref_idx = None
    best_symptom_jaccard = 0.0
    best_herb_jaccard = 0.0
    best_mean_jaccard = -1.0

    for ref_idx in reference_train_idx:
        ref_row = reference_formula_df.iloc[int(ref_idx)]
        symptom_jaccard = jaccard_similarity(external_symptoms, ref_row["Symptoms"])
        herb_jaccard = jaccard_similarity(external_herbs, ref_row["Herbs"])
        mean_jaccard = 0.5 * (symptom_jaccard + herb_jaccard)

        if best_ref_idx is None or mean_jaccard > best_mean_jaccard:
            best_ref_idx = int(ref_idx)
            best_symptom_jaccard = float(symptom_jaccard)
            best_herb_jaccard = float(herb_jaccard)
            best_mean_jaccard = float(mean_jaccard)

    if best_ref_idx is None:
        raise ValueError("Reference internal training split is empty.")

    excluded = (
        best_symptom_jaccard >= float(symptom_threshold)
        and best_herb_jaccard >= float(herb_threshold)
    )

    return {
        "matched_reference_train_row": best_ref_idx,
        "symptom_jaccard": best_symptom_jaccard,
        "herb_jaccard": best_herb_jaccard,
        "mean_jaccard": best_mean_jaccard,
        "excluded": bool(excluded),
    }


def filter_external_test_against_internal_training(
    formula_df: pd.DataFrame,
    test_idx: List[int],
    reference_formula_df: pd.DataFrame,
    reference_train_idx: List[int],
    symptom_threshold: float,
    herb_threshold: float,
) -> Tuple[List[int], pd.DataFrame]:
    """Apply the manuscript symptom-and-herb Jaccard leakage-control filter."""
    kept = []
    rows = []

    for external_idx in test_idx:
        external_row = formula_df.iloc[int(external_idx)]
        match = find_best_reference_training_match(
            external_symptoms=external_row["Symptoms"],
            external_herbs=external_row["Herbs"],
            reference_formula_df=reference_formula_df,
            reference_train_idx=reference_train_idx,
            symptom_threshold=symptom_threshold,
            herb_threshold=herb_threshold,
        )

        rows.append({
            "external_all_valid_row_idx": int(external_idx),
            **match,
        })
        if not match["excluded"]:
            kept.append(int(external_idx))

    filtering_df = pd.DataFrame(rows)
    if len(filtering_df):
        filtering_df = filtering_df.sort_values(
            ["excluded", "mean_jaccard"],
            ascending=[False, False],
        ).reset_index(drop=True)

    return sorted(kept), filtering_df


def build_external_evaluation_split(args, formula_df: pd.DataFrame):
    """Build and save the manuscript TCM-PD train/validation/test evaluation split."""
    train_idx, val_idx, test_idx_raw = build_external_train_val_test_indices(
        formula_df=formula_df,
        val_ratio=args.external_val_ratio,
        seed=args.seed,
    )

    if args.test_filter_rule == "none":
        test_idx = test_idx_raw
        filtering_df = pd.DataFrame({
            "external_all_valid_row_idx": test_idx_raw,
            "excluded": False,
        })
    else:
        reference_formula_path = require_file(
            args.reference_formula_data,
            "reference formula data required to rebuild the external split",
        )
        reference_train_path = require_file(
            os.path.join(args.reference_split_dir, "train_idx.pkl"),
            "reference training indices required to rebuild the external split",
        )

        reference_formula_df = pd.read_pickle(reference_formula_path).reset_index(drop=True)
        reference_train_idx = sorted(int(x) for x in load_pickle(reference_train_path))

        bad_idx = [i for i in reference_train_idx if i < 0 or i >= len(reference_formula_df)]
        if bad_idx:
            raise ValueError(f"Reference training indices out of range: {bad_idx[:10]}")

        test_idx, filtering_df = filter_external_test_against_internal_training(
            formula_df=formula_df,
            test_idx=test_idx_raw,
            reference_formula_df=reference_formula_df,
            reference_train_idx=reference_train_idx,
            symptom_threshold=args.external_symptom_jaccard,
            herb_threshold=args.external_herb_jaccard,
        )

    if not test_idx:
        raise ValueError("No external test records remain after overlap filtering.")

    split_info = {
        "external_train_pool_size": int(len(train_idx) + len(val_idx)),
        "external_val_ratio": float(args.external_val_ratio),
        "train_size": int(len(train_idx)),
        "validation_size": int(len(val_idx)),
        "test_size_before_filtering": int(len(test_idx_raw)),
        "test_size_after_filtering": int(len(test_idx)),
        "excluded_test_size": int(len(test_idx_raw) - len(test_idx)),
        "test_filter_rule": args.test_filter_rule,
        "external_symptom_jaccard": float(args.external_symptom_jaccard),
        "external_herb_jaccard": float(args.external_herb_jaccard),
    }
    return train_idx, val_idx, test_idx, filtering_df, split_info


def load_or_build_external_split(args, formula_df: pd.DataFrame):
    """Load cached external split files, or build them when requested or absent."""
    split_dir = args.external_split_dir or os.path.join(args.out_dir, "data_split")
    ensure_dir(split_dir)

    paths = {
        "train": os.path.join(split_dir, "external_train_idx.pkl"),
        "val": os.path.join(split_dir, "external_val_idx.pkl"),
        "test": os.path.join(split_dir, "external_test_idx.pkl"),
        "metadata": os.path.join(split_dir, "external_split_metadata.json"),
        "filtering": os.path.join(split_dir, "external_test_similarity_filtering.csv"),
    }

    split_exists = all(os.path.exists(paths[k]) for k in ["train", "val", "test"])

    if args.rebuild_external_split or not split_exists:
        train_idx, val_idx, test_idx, filtering_df, split_info = build_external_evaluation_split(
            args,
            formula_df,
        )
        save_pickle(train_idx, paths["train"])
        save_pickle(val_idx, paths["val"])
        save_pickle(test_idx, paths["test"])
        filtering_df.to_csv(paths["filtering"], index=False, encoding="utf_8_sig")
        with open(paths["metadata"], "w", encoding="utf-8") as fp:
            json.dump(split_info, fp, ensure_ascii=False, indent=2)
        split_info["split_source"] = "rebuilt"
    else:
        train_idx = sorted(int(x) for x in load_pickle(paths["train"]))
        val_idx = sorted(int(x) for x in load_pickle(paths["val"]))
        test_idx = sorted(int(x) for x in load_pickle(paths["test"]))
        split_info = {
            "split_source": "cached",
            "split_dir": split_dir,
            "train_size": len(train_idx),
            "validation_size": len(val_idx),
            "test_size": len(test_idx),
        }

    n = len(formula_df)
    all_idx = train_idx + val_idx + test_idx
    if any(i < 0 or i >= n for i in all_idx):
        raise ValueError("Cached external split contains indices outside the external dataframe range.")
    if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
        raise ValueError("External train/validation/test split files overlap.")

    return train_idx, val_idx, test_idx, split_info


def indices_to_mask(indices: List[int], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(indices, dtype=int)] = True
    return mask


# =============================================================================
# 3. Embedding feature construction
# =============================================================================

def zscore_poly2_fit_transform(X_fit: np.ndarray, *Xs: np.ndarray):
    """Fit z-score statistics on fitting records and return [z, z^2] features."""
    X_fit = np.asarray(X_fit, dtype=np.float32)
    mu = X_fit.mean(axis=0, keepdims=True)
    sd = X_fit.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)

    outputs = []
    for X in (X_fit,) + Xs:
        Z = (np.asarray(X, dtype=np.float32) - mu) / sd
        outputs.append(np.concatenate([Z, Z * Z], axis=1).astype(np.float32))
    return outputs


# =============================================================================
# 4. Ridge and MLP readouts
# =============================================================================

def fit_ridge_multilabel(F: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form multilabel ridge regression with an unregularized intercept."""
    A = np.concatenate(
        [F.astype(np.float32), np.ones((F.shape[0], 1), dtype=np.float32)],
        axis=1,
    ).astype(np.float64)

    G = A.T @ A
    reg = np.eye(A.shape[1], dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0
    W = np.linalg.solve(G + reg, A.T @ Y.astype(np.float64))
    return W.astype(np.float32)


def predict_ridge(F: np.ndarray, W: np.ndarray) -> np.ndarray:
    A = np.concatenate(
        [F.astype(np.float32), np.ones((F.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    return (A @ W).astype(np.float32)


class EmbeddingMLPHead(nn.Module):
    """Two-hidden-layer multilabel MLP readout used for TCM-PD prediction."""

    def __init__(self, input_dim: int, n_labels: int, hidden1: int, hidden2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_labels),
        )

    def forward(self, x):
        return self.net(x)


def predict_model(model, X, device, batch_size=1024):
    model.eval()
    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )
    outputs = []
    with torch.no_grad():
        for xb in loader:
            outputs.append(model(xb.to(device)).cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def train_mlp_head(
    X_train,
    Y_train,
    X_val,
    Y_val,
    args,
    device,
    n_epochs=None,
    use_validation=True,
):
    model = EmbeddingMLPHead(
        input_dim=X_train.shape[1],
        n_labels=Y_train.shape[1],
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    max_epochs = int(n_epochs or args.epochs)
    best_state = None
    best_epoch = 0
    best_r20 = -1.0
    patience_left = int(args.patience)
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)

        row = {"epoch": epoch, "train_loss": total_loss / max(1, n_seen)}

        if use_validation:
            val_scores = predict_model(model, X_val, device, args.batch_size)
            val_metrics = evaluate_scores(val_scores, Y_val, "validation", [20])
            r20 = float(val_metrics["R@20"])
            row["val_R@20"] = r20

            if r20 > best_r20:
                best_r20 = r20
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience_left = int(args.patience)
            else:
                patience_left -= 1

            if epoch == 1 or epoch % args.print_every == 0:
                print(
                    f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
                    f"val_R@20={r20:.4f} best={best_r20:.4f}",
                    flush=True,
                )

            history.append(row)
            if patience_left <= 0:
                break
        else:
            if epoch == 1 or epoch % args.print_every == 0:
                print(f"epoch={epoch:03d} loss={row['train_loss']:.4f}", flush=True)
            history.append(row)

    if use_validation:
        if best_state is None:
            raise RuntimeError("No validation checkpoint was selected.")
        model.load_state_dict(best_state)
    else:
        best_epoch = max_epochs
        best_r20 = np.nan

    return model, pd.DataFrame(history), {
        "best_epoch": int(best_epoch),
        "best_val_R@20": None if np.isnan(best_r20) else float(best_r20),
    }


# =============================================================================
# 5. Evaluation
# =============================================================================

def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), scores.shape[1])
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, part], axis=1)
    return part[row, order]


def evaluate_scores(scores, Y_true, model_name: str, k_values: List[int]) -> Dict:
    scores = np.asarray(scores, dtype=np.float32)
    Y_true = np.asarray(Y_true, dtype=np.uint8)
    true_sizes = Y_true.sum(axis=1).astype(float)

    out = {
        "Model": model_name,
        "N test records": int(Y_true.shape[0]),
    }

    for k in k_values:
        idx = topk_indices(scores, k)
        hits = Y_true[np.arange(Y_true.shape[0])[:, None], idx].sum(axis=1).astype(float)
        precision = hits / float(k)
        recall = np.divide(
            hits,
            true_sizes,
            out=np.zeros_like(hits),
            where=true_sizes > 0,
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )

        out[f"P@{k}"] = float(precision.mean())
        out[f"R@{k}"] = float(recall.mean())
        out[f"F1@{k}"] = float(f1.mean())

    return out


# =============================================================================
# 6. Published TCM-PD benchmark reference
# =============================================================================

def make_literature_reference_table() -> pd.DataFrame:
    """Literature-reported TCM-PD results displayed as contextual references."""
    model_meta = {
        "LinkLDA": ("LinkLDA", "Erosheva et al. 2004"),
        "Link-PLSA-LDA": ("Link-PLSA-LDA", "Nallapati & Cohen 2008"),
        "PTM": ("PTM", "Yao et al. 2018"),
        "KDHR": ("KDHR", "Yang et al. 2022"),
        "TCMPR": ("TCMPR", "Dong et al. 2021"),
        "PresRecST": ("PresRecST", "Dong et al. 2024"),
    }

    metrics = {
        "LinkLDA":       [0.2040, 0.1393, 0.1656, 0.1624, 0.2187, 0.1864, 0.1222, 0.3278, 0.1780],
        "Link-PLSA-LDA": [0.2059, 0.1432, 0.1689, 0.1651, 0.2264, 0.1910, 0.1238, 0.3350, 0.1808],
        "PTM":           [0.2121, 0.1401, 0.1688, 0.1698, 0.2221, 0.1924, 0.1288, 0.3364, 0.1863],
        "KDHR":          [0.2138, 0.1510, 0.1770, 0.1660, 0.2284, 0.1922, 0.1251, 0.3414, 0.1832],
        "TCMPR":         [0.1833, 0.1338, 0.1547, 0.1447, 0.2127, 0.1722, 0.1151, 0.3351, 0.1713],
        "PresRecST":     [0.2238, 0.1512, 0.1805, 0.1749, 0.2338, 0.2001, 0.1290, 0.3465, 0.1879],
    }

    metric_cols = [
        "P@5", "R@5", "F1@5",
        "P@10", "R@10", "F1@10",
        "P@20", "R@20", "F1@20",
    ]

    rows = []
    for model in model_meta:
        display_name, reference = model_meta[model]
        rows.append({
            "Model": model,
            "Display_model": display_name,
            "Dataset": "TCM-PD literature benchmark",
            **dict(zip(metric_cols, metrics[model])),
            "Reference": reference,
            "Comparison_note": "Contextual benchmark reference; evaluation subsets are not strictly matched.",
        })
    return pd.DataFrame(rows)


def combine_literature_and_tcm_es(literature_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in [MODEL_MLP, MODEL_RIDGE]:
        row = summary_df.loc[summary_df["Model"] == model_name].iloc[0]
        rows.append({
            "Model": model_name,
            "Display_model": model_name,
            "Dataset": "TCM-PD mapped and overlap-filtered test set",
            **{metric: float(row[metric]) for metric in [
                "P@5", "R@5", "F1@5",
                "P@10", "R@10", "F1@10",
                "P@20", "R@20", "F1@20",
            ]},
            "Reference": "This study",
            "Comparison_note": "TCM-ES evaluation uses standardized mapping and overlap filtering.",
        })
    return pd.concat([literature_df, pd.DataFrame(rows)], ignore_index=True)


def plot_external_benchmark(combined_df: pd.DataFrame, out_dir: str) -> None:
    """Plot the external TCM-PD benchmark using the original figure style."""
    model_order = [
        "LinkLDA",
        "Link-PLSA-LDA",
        "PTM",
        "KDHR",
        "TCMPR",
        "PresRecST",
        MODEL_MLP,
        MODEL_RIDGE,
    ]

    available = set(combined_df["Display_model"].astype(str))
    model_order = [name for name in model_order if name in available]
    if not model_order:
        raise ValueError("No expected model labels were found in combined benchmark results.")

    plot_df = combined_df.set_index("Display_model").loc[model_order].copy()

    # Preserve the publication-style colors used in the original script.
    color_map = {
        "LinkLDA": "#B9C0C8",
        "Link-PLSA-LDA": "#A8BFA4",
        "PTM": "#BEA6C7",
        "KDHR": "#D58F78",
        "TCMPR": "#9CA9B5",
        "PresRecST": "#65A5A2",
        MODEL_MLP: "#0B5FA5",
        MODEL_RIDGE: "#4E88C7",
    }

    panel_specs = [
        {"title": "Precision", "metrics": ["P@5", "P@10", "P@20"], "tick_labels": ["5", "10", "20"]},
        {"title": "Recall", "metrics": ["R@5", "R@10", "R@20"], "tick_labels": ["5", "10", "20"]},
        {"title": "F1 score", "metrics": ["F1@5", "F1@10", "F1@20"], "tick_labels": ["5", "10", "20"]},
    ]

    missing_metrics = [
        metric
        for panel in panel_specs
        for metric in panel["metrics"]
        if metric not in plot_df.columns
    ]
    if missing_metrics:
        raise ValueError(f"Missing benchmark metrics in combined_df: {missing_metrics}")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), sharey=False)

    n_models = len(model_order)
    group_centers = np.arange(3, dtype=float)
    total_group_width = 0.84
    bar_width = total_group_width / n_models
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_width
    panel_letters = ["A", "B", "C"]

    def nice_upper_limit(max_value):
        target = float(max_value) * 1.14
        return max(0.10, float(np.ceil(target / 0.05) * 0.05))

    for panel_index, (ax, panel) in enumerate(zip(axes, panel_specs)):
        metrics = panel["metrics"]
        panel_values = plot_df[metrics].astype(float).to_numpy()

        for model_index, model_name in enumerate(model_order):
            is_ours = model_name.startswith("TCM-ES")
            model_values = panel_values[model_index]
            bar_positions = group_centers + offsets[model_index]

            bars = ax.bar(
                bar_positions,
                model_values,
                width=bar_width * 0.92,
                color=color_map[model_name],
                edgecolor="#17212B" if is_ours else "white",
                linewidth=1.05 if is_ours else 0.55,
                alpha=1.0 if is_ours else 0.92,
                zorder=3 if is_ours else 2,
            )

            if is_ours:
                for rectangle, value in zip(bars, model_values):
                    ax.annotate(
                        f"{float(value):.3f}",
                        xy=(rectangle.get_x() + rectangle.get_width() / 2.0, rectangle.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=7.3,
                        fontweight="bold",
                        color=color_map[model_name],
                        clip_on=False,
                    )

        ax.set_ylim(0.0, nice_upper_limit(float(np.nanmax(panel_values))))
        ax.set_xticks(group_centers)
        ax.set_xticklabels(panel["tick_labels"], rotation=0, ha="center", fontsize=10)
        ax.set_xlabel("Recommendation cutoff, k", fontsize=10)
        ax.set_title(panel["title"], fontsize=10.5, fontweight="bold", pad=9)
        ax.tick_params(axis="both", which="major", labelsize=9.5, length=3, width=0.8)
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.28, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.text(
            -0.15,
            1.06,
            panel_letters[panel_index],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
        )

    axes[0].set_ylabel("Metric value", fontsize=10.5)

    legend_handles = [
        Patch(
            facecolor=color_map[model_name],
            edgecolor="#17212B" if model_name.startswith("TCM-ES") else "white",
            linewidth=1.05 if model_name.startswith("TCM-ES") else 0.55,
            label=model_name,
        )
        for model_name in model_order
    ]

    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=5,
        frameon=False,
        fontsize=8.8,
        columnspacing=1.15,
        handlelength=1.6,
        handletextpad=0.48,
    )
    for legend_text in legend.get_texts():
        if legend_text.get_text().startswith("TCM-ES"):
            legend_text.set_fontweight("bold")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.84], pad=0.7, w_pad=1.25)
    save_figure(fig, os.path.join(out_dir, "fig_literature_metric_grouped_bars_three_panels"))
    plt.close(fig)


# =============================================================================
# 7. Command-line interface and workflow
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="External TCM-PD formula prediction from frozen TCM-ES symptom-pattern embeddings."
    )

    parser.add_argument(
        "--external-formula-data",
        default="data/TCM_PD_external/TCM_PD_formula_data_external_all_valid.pkl",
    )
    parser.add_argument(
        "--external-embedding-dir",
        default="results/embeddings/TCM_PD_external/original",
    )
    parser.add_argument(
        "--herb-list",
        default="core/standard_TCM_entities/herb_list.pkl",
    )

    parser.add_argument(
        "--external-split-dir",
        default=None,
        help="Directory containing cached external train/validation/test split files.",
    )
    parser.add_argument(
        "--rebuild-external-split",
        action="store_true",
        help="Force reconstruction of the external evaluation split and overlap filter.",
    )
    parser.add_argument(
        "--reference-formula-data",
        default="data/TCM_formulas/TCM_formula_data.pkl",
        help="Internal formula corpus used only when rebuilding the external overlap filter.",
    )
    parser.add_argument(
        "--reference-split-dir",
        default="data/training",
        help="Internal split directory used only when rebuilding the external overlap filter.",
    )
    parser.add_argument(
        "--test-filter-rule",
        choices=["symptom_and_herb", "none"],
        default="symptom_and_herb",
    )
    parser.add_argument("--external-val-ratio", type=float, default=0.1)
    parser.add_argument("--external-symptom-jaccard", type=float, default=0.8)
    parser.add_argument("--external-herb-jaccard", type=float, default=0.8)

    parser.add_argument("--out-dir", default="results/TCM_formula_prediction_external")
    parser.add_argument("--k-values", default="5,10,20")
    parser.add_argument("--ridge-lambda", type=float, default=10.0)

    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    set_plot_style()
    ensure_dir(args.out_dir)
    device = resolve_device(args.device)

    k_values = sorted(set(int(x.strip()) for x in args.k_values.split(",") if x.strip()))
    if k_values != [5, 10, 20]:
        print(f"[Info] Evaluating requested k values: {k_values}", flush=True)

    print("[1/6] Loading external data and frozen embeddings...", flush=True)
    data = load_external_inputs(args)
    formula_df = data["formula_df"]
    X = data["X"]
    Y = build_herb_label_matrix(formula_df, data["herb_list"])

    print("[2/6] Loading or building the external evaluation split...", flush=True)
    train_idx, val_idx, test_idx, split_info = load_or_build_external_split(args, formula_df)

    n = len(formula_df)
    train_mask = indices_to_mask(train_idx, n)
    val_mask = indices_to_mask(val_idx, n)
    test_mask = indices_to_mask(test_idx, n)
    trainval_mask = train_mask | val_mask

    Y_train = Y[train_mask]
    Y_val = Y[val_mask]
    Y_test = Y[test_mask]

    print(
        f"External records: total={n}, train={train_mask.sum()}, "
        f"validation={val_mask.sum()}, test={test_mask.sum()}",
        flush=True,
    )

    print("[3/6] Selecting the MLP epoch on the validation subset...", flush=True)
    F_train, F_val = zscore_poly2_fit_transform(X[train_mask], X[val_mask])
    _, validation_history, best_info = train_mlp_head(
        X_train=F_train,
        Y_train=Y_train,
        X_val=F_val,
        Y_val=Y_val,
        args=args,
        device=device,
        use_validation=True,
    )
    validation_history.to_csv(
        os.path.join(args.out_dir, "external_validation_training_history.csv"),
        index=False,
    )

    print("[4/6] Refitting MLP and ridge readouts on external train+validation records...", flush=True)
    Y_fit = Y[trainval_mask]
    F_fit, F_test = zscore_poly2_fit_transform(X[trainval_mask], X[test_mask])

    final_epochs = max(1, int(best_info["best_epoch"]))
    mlp_final, final_history, _ = train_mlp_head(
        X_train=F_fit,
        Y_train=Y_fit,
        X_val=F_test,
        Y_val=Y_test,
        args=args,
        device=device,
        n_epochs=final_epochs,
        use_validation=False,
    )
    final_history.to_csv(
        os.path.join(args.out_dir, "external_final_refit_history.csv"),
        index=False,
    )

    scores_mlp = predict_model(mlp_final, F_test, device, args.batch_size)
    ridge_weights = fit_ridge_multilabel(F_fit, Y_fit, args.ridge_lambda)
    scores_ridge = predict_ridge(F_test, ridge_weights)

    print("[5/6] Evaluating external test performance...", flush=True)
    summary = pd.DataFrame([
        evaluate_scores(scores_mlp, Y_test, MODEL_MLP, k_values),
        evaluate_scores(scores_ridge, Y_test, MODEL_RIDGE, k_values),
    ])
    summary.to_csv(
        os.path.join(args.out_dir, "external_formula_prediction_summary.csv"),
        index=False,
    )

    print("[6/6] Writing contextual TCM-PD benchmark comparison...", flush=True)
    literature = make_literature_reference_table()
    literature.to_csv(
        os.path.join(args.out_dir, "published_tcm_pd_reference.csv"),
        index=False,
    )
    combined = combine_literature_and_tcm_es(literature, summary)
    combined.to_csv(
        os.path.join(args.out_dir, "external_formula_prediction_with_published_reference.csv"),
        index=False,
    )
    plot_external_benchmark(combined, args.out_dir)

    metadata = {
        "task": "external TCM-PD formula prediction",
        "input_paths": data["paths"],
        "external_split": split_info,
        "n_external_records": int(n),
        "n_train_records": int(train_mask.sum()),
        "n_validation_records": int(val_mask.sum()),
        "n_test_records": int(test_mask.sum()),
        "input_features": "train-fitted standardized [z_TCM-ES, z_TCM-ES^2]",
        "models": [MODEL_MLP, MODEL_RIDGE],
        "best_epoch_from_validation": int(best_info["best_epoch"]),
        "best_validation_R@20": best_info["best_val_R@20"],
        "hyperparameters": {
            "k_values": k_values,
            "ridge_lambda": float(args.ridge_lambda),
            "hidden1": int(args.hidden1),
            "hidden2": int(args.hidden2),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
        },
        "comparison_note": (
            "Published TCM-PD values are contextual references because standardized entity mapping "
            "and overlap filtering change the evaluated TCM-ES test subset."
        ),
    }
    with open(os.path.join(args.out_dir, "external_run_metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
