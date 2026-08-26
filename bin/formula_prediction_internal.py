#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Held-out internal formula prediction from frozen TCM-ES symptom-pattern embeddings.

Four readouts are evaluated on the same held-out test records:
    1. TCM-ES MLP head using standardized [z, z^2] features.
    2. TCM-ES ridge head using the same standardized [z, z^2] features.
    3. Raw symptom multi-hot ridge baseline.
    4. Shuffled-embedding ridge control, averaged across repeated shuffles.

The task is retrospective recovery/ranking of herbs recorded in historical
formula records. It is not prospective clinical prescription recommendation.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


MODEL_MLP = "TCM-ES MLP head"
MODEL_RIDGE = "TCM-ES ridge head"
MODEL_MULTIHOT = "Multi-hot baseline"
MODEL_SHUFFLED = "Shuffled embedding baseline"
MODEL_ORDER = [MODEL_MLP, MODEL_RIDGE, MODEL_MULTIHOT, MODEL_SHUFFLED]


# =============================================================================
# 1. Basic utilities
# =============================================================================

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_pickle(path: str):
    with open(path, "rb") as fp:
        return pickle.load(fp)


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
# 2. Data loading and feature construction
# =============================================================================

def load_inputs(args) -> Dict:
    selected_ids_path = os.path.join(
        args.embedding_dir,
        "selected_formula_ids(line_number).pkl",
    )
    symptom_embeddings_path = os.path.join(
        args.embedding_dir,
        "selected_formula_symptom_embeddings.pkl",
    )

    paths = {
        "formula_data": require_file(args.formula_data, "formula data"),
        "herb_list": require_file(args.herb_list, "herb list"),
        "symptom_list": require_file(args.symptom_list, "symptom list"),
        "selected_formula_ids": require_file(selected_ids_path, "selected formula IDs"),
        "symptom_embeddings": require_file(
            symptom_embeddings_path,
            "selected formula symptom embeddings",
        ),
        "train_idx": require_file(args.train_idx, "train split indices"),
        "val_idx": require_file(args.val_idx, "validation split indices"),
        "test_idx": require_file(args.test_idx, "test split indices"),
    }

    formula_df = pd.read_pickle(paths["formula_data"]).reset_index(drop=True)
    herb_list = list(load_pickle(paths["herb_list"]))
    symptom_list = list(load_pickle(paths["symptom_list"]))
    selected_ids = np.asarray(load_pickle(paths["selected_formula_ids"]), dtype=int)
    symptom_embeddings = np.asarray(load_pickle(paths["symptom_embeddings"]), dtype=np.float32)

    if symptom_embeddings.ndim != 2:
        raise ValueError("Symptom embeddings must be a 2D array.")
    if symptom_embeddings.shape[0] != len(selected_ids):
        raise ValueError(
            f"Embedding rows {symptom_embeddings.shape[0]} != selected IDs {len(selected_ids)}"
        )
    if selected_ids.size and (selected_ids.min() < 0 or selected_ids.max() >= len(formula_df)):
        raise ValueError("Selected formula IDs are outside the formula dataframe range.")

    return {
        "paths": paths,
        "formula_df": formula_df,
        "herb_list": herb_list,
        "symptom_list": symptom_list,
        "selected_ids": selected_ids,
        "X": symptom_embeddings,
        "train_idx": set(int(x) for x in load_pickle(paths["train_idx"])),
        "val_idx": set(int(x) for x in load_pickle(paths["val_idx"])),
        "test_idx": set(int(x) for x in load_pickle(paths["test_idx"])),
    }


def build_label_and_symptom_matrices(
    formula_df: pd.DataFrame,
    selected_ids: np.ndarray,
    herb_list: List[str],
    symptom_list: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build observed-herb labels and raw symptom multi-hot inputs."""
    herb2id = {h: i for i, h in enumerate(herb_list)}
    symptom2id = {s: i for i, s in enumerate(symptom_list)}

    Y = np.zeros((len(selected_ids), len(herb_list)), dtype=np.uint8)
    S = np.zeros((len(selected_ids), len(symptom_list)), dtype=np.uint8)

    for i, row_id in enumerate(selected_ids):
        row = formula_df.iloc[int(row_id)]

        for herb in unique_list(row["Herbs"]):
            j = herb2id.get(herb)
            if j is not None:
                Y[i, j] = 1

        for symptom in unique_list(row["Symptoms"]):
            j = symptom2id.get(symptom)
            if j is not None:
                S[i, j] = 1

    return Y, S


def build_split_masks(selected_ids, train_idx, val_idx, test_idx):
    train_mask = np.array([int(x) in train_idx for x in selected_ids], dtype=bool)
    val_mask = np.array([int(x) in val_idx for x in selected_ids], dtype=bool)
    test_mask = np.array([int(x) in test_idx for x in selected_ids], dtype=bool)

    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Empty train/validation/test split after selected-ID alignment.")

    overlap = (
        (train_mask & val_mask).any()
        or (train_mask & test_mask).any()
        or (val_mask & test_mask).any()
    )
    if overlap:
        raise ValueError("Train/validation/test masks overlap.")

    return train_mask, val_mask, test_mask


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
# 3. Ridge readouts and shuffled-embedding control
# =============================================================================

def fit_ridge_multilabel(F: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form multi-label ridge regression with an unregularized intercept."""
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


def fit_predict_shuffled_embedding_ridge(
    X_fit_raw: np.ndarray,
    Y_fit: np.ndarray,
    X_test_raw: np.ndarray,
    Y_test: np.ndarray,
    ridge_lambda: float,
    repeats: int,
    seed: int,
    k_values: List[int],
):
    """Evaluate ridge readouts after random reassignment of embedding vectors."""
    if repeats <= 0:
        raise ValueError("random_control_repeats must be positive.")

    detail_rows = []
    score_list = []

    for rep in range(repeats):
        rng = np.random.default_rng(seed + 10000 + rep)
        X_fit_shuffled = X_fit_raw[rng.permutation(len(X_fit_raw))]
        X_test_shuffled = X_test_raw[rng.permutation(len(X_test_raw))]

        F_fit, F_test = zscore_poly2_fit_transform(
            X_fit_shuffled,
            X_test_shuffled,
        )
        W = fit_ridge_multilabel(F_fit, Y_fit, ridge_lambda)
        scores = predict_ridge(F_test, W)
        score_list.append(scores)

        metrics = evaluate_scores(
            scores,
            Y_test,
            MODEL_SHUFFLED,
            k_values,
        )
        metrics["repeat"] = rep + 1
        detail_rows.append(metrics)

    detail_df = pd.DataFrame(detail_rows)
    summary = {"Model": MODEL_SHUFFLED}
    for col in detail_df.columns:
        if col in {"Model", "repeat", "N test records"}:
            continue
        if pd.api.types.is_numeric_dtype(detail_df[col]):
            summary[col] = float(detail_df[col].mean())
            summary[col + "_std"] = float(detail_df[col].std(ddof=0))
    summary["N test records"] = int(Y_test.shape[0])

    mean_scores = np.mean(np.stack(score_list, axis=0), axis=0).astype(np.float32)
    return mean_scores, summary, detail_df


# =============================================================================
# 4. MLP readout
# =============================================================================

class EmbeddingMLPHead(nn.Module):
    """Two-hidden-layer multilabel MLP readout."""

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


def _make_mlp(input_dim, n_labels, args, device):
    return EmbeddingMLPHead(
        input_dim=input_dim,
        n_labels=n_labels,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    ).to(device)


def _make_optimizer(model, args):
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def _make_training_loader(X, Y, batch_size):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_mlp_with_validation(
    X_train,
    Y_train,
    X_val,
    Y_val,
    args,
    device,
):
    """Select the MLP training epoch using validation Recall@20."""
    model = _make_mlp(X_train.shape[1], Y_train.shape[1], args, device)
    optimizer = _make_optimizer(model, args)
    criterion = nn.BCEWithLogitsLoss()
    loader = _make_training_loader(X_train, Y_train, args.batch_size)

    best_state = None
    best_epoch = 0
    best_val_r20 = -np.inf
    patience_left = int(args.patience)
    history = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)

        val_scores = predict_model(model, X_val, device, args.batch_size)
        val_metrics = evaluate_scores(val_scores, Y_val, "validation", [20])
        val_r20 = float(val_metrics["R@20"])

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, n_seen),
            "val_R@20": val_r20,
        })

        if val_r20 > best_val_r20:
            best_val_r20 = val_r20
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = int(args.patience)
        else:
            patience_left -= 1

        if epoch == 1 or epoch % int(args.print_every) == 0:
            print(
                f"epoch={epoch:03d} "
                f"loss={history[-1]['train_loss']:.4f} "
                f"val_R@20={val_r20:.4f} "
                f"best={best_val_r20:.4f}",
                flush=True,
            )

        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected for the MLP readout.")

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), {
        "best_epoch": int(best_epoch),
        "best_val_R@20": float(best_val_r20),
    }


def fit_mlp_fixed_epochs(X_fit, Y_fit, n_epochs, args, device):
    """Refit the MLP on train+validation for the selected number of epochs."""
    model = _make_mlp(X_fit.shape[1], Y_fit.shape[1], args, device)
    optimizer = _make_optimizer(model, args)
    criterion = nn.BCEWithLogitsLoss()
    loader = _make_training_loader(X_fit, Y_fit, args.batch_size)

    history = []
    for epoch in range(1, int(n_epochs) + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, n_seen),
        })

    return model, pd.DataFrame(history)


# =============================================================================
# 5. Evaluation and figures
# =============================================================================

def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(int(k), scores.shape[1])
    if k_eff <= 0:
        raise ValueError("k must be positive.")

    part = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1)
    return part[rows, order]


def evaluate_scores(scores, Y_true, model_name, k_values):
    """Compute the formula-prediction metrics reported for the internal test set."""
    scores = np.asarray(scores, dtype=np.float32)
    Y_true = np.asarray(Y_true, dtype=np.uint8)
    true_sizes = Y_true.sum(axis=1).astype(float)

    out = {
        "Model": model_name,
        "N test records": int(Y_true.shape[0]),
    }

    for k in k_values:
        k = int(k)
        idx = topk_indices(scores, k)
        k_eff = idx.shape[1]
        hits = Y_true[np.arange(Y_true.shape[0])[:, None], idx].sum(axis=1).astype(float)

        precision = hits / float(k_eff)
        recall = np.divide(
            hits,
            true_sizes,
            out=np.zeros_like(hits),
            where=true_sizes > 0,
        )

        out[f"P@{k}"] = float(precision.mean())
        out[f"R@{k}"] = float(recall.mean())

        if k == 20:
            f1 = np.divide(
                2.0 * precision * recall,
                precision + recall,
                out=np.zeros_like(precision),
                where=(precision + recall) > 0,
            )
            out["F1@20"] = float(f1.mean())

    return out


def save_summary_table(summary: pd.DataFrame, out_dir: str) -> None:
    metric_cols = [
        "Model",
        "P@5", "R@5",
        "P@10", "R@10",
        "P@20", "R@20", "F1@20",
    ]
    available = [col for col in metric_cols if col in summary.columns]
    summary[available].to_csv(
        os.path.join(out_dir, "heldout_formula_prediction_summary.csv"),
        index=False,
    )


def plot_metric_comparison(summary: pd.DataFrame, out_dir: str) -> None:
    """Plot the manuscript four-model comparison using the original figure style."""
    metrics = ["P@5", "R@5", "P@10", "R@10", "P@20", "R@20", "F1@20"]
    metrics = [metric for metric in metrics if metric in summary.columns]

    available_models = summary["Model"].astype(str).tolist()
    ordered_models = [model for model in MODEL_ORDER if model in available_models]
    plot_df = summary.set_index("Model").loc[ordered_models].reset_index()

    values = plot_df[metrics].astype(float).to_numpy()
    labels = plot_df["Model"].tolist()

    # Keep the fixed publication-style colors used in the original analysis.
    color_map = {
        MODEL_MLP: "#3B6FB6",
        MODEL_RIDGE: "#E07B39",
        MODEL_MULTIHOT: "#7B4FA3",
        MODEL_SHUFFLED: "#9D755D",
    }
    colors = [color_map.get(label, "#6E6E6E") for label in labels]

    fig, ax = plt.subplots(figsize=(7.4, 3.15))

    x = np.arange(len(metrics))
    width = min(0.78 / max(1, len(labels)), 0.16)

    for i, label in enumerate(labels):
        offset = (i - (len(labels) - 1) / 2) * width
        ax.bar(
            x + offset,
            values[i],
            width=width,
            label=label,
            color=colors[i],
            edgecolor="black",
            linewidth=0.35,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, min(1.0, max(0.60, float(np.nanmax(values)) + 0.08)))

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=2,
        handlelength=1.4,
        columnspacing=1.2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig_heldout_formula_prediction"))
    plt.close(fig)


# =============================================================================
# 6. Command-line interface
# =============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Held-out internal formula prediction from frozen TCM-ES symptom-pattern embeddings."
    )

    parser.add_argument(
        "--formula-data",
        default="data/TCM_formulas/TCM_formula_data_example_2000.pkl",
    )
    parser.add_argument(
        "--embedding-dir",
        default="results/embeddings/TCM_embeddings",
        help="Directory containing selected formula IDs and symptom-pattern embeddings.",
    )
    parser.add_argument("--herb-list", default="core/standard_TCM_entities/herb_list.pkl")
    parser.add_argument("--symptom-list", default="core/standard_TCM_entities/symptom_list.pkl")
    parser.add_argument("--train-idx", default="data/training/train_idx.pkl")
    parser.add_argument("--val-idx", default="data/training/val_idx.pkl")
    parser.add_argument("--test-idx", default="data/training/test_idx.pkl")
    parser.add_argument("--out-dir", default="results/TCM_formula_prediction")

    # Evaluation settings reported for the held-out internal test set.
    parser.add_argument("--k-values", default="5,10,20")
    parser.add_argument("--ridge-lambda", type=float, default=10.0)
    parser.add_argument("--random-control-repeats", type=int, default=5)

    # MLP architecture reported in the Methods.
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Optimization controls for the downstream MLP readout.
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    return parser.parse_args(argv)


# =============================================================================
# 7. Main workflow
# =============================================================================

def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    set_plot_style()
    ensure_dir(args.out_dir)

    device = resolve_device(args.device)
    k_values = sorted({int(x.strip()) for x in args.k_values.split(",") if x.strip()})
    required_k = {5, 10, 20}
    if not required_k.issubset(k_values):
        raise ValueError("--k-values must include 5, 10, and 20 for the manuscript metrics.")

    print("[1/6] Loading formula data, embeddings, and splits...", flush=True)
    data = load_inputs(args)

    print("[2/6] Building herb labels and symptom multi-hot inputs...", flush=True)
    Y, S = build_label_and_symptom_matrices(
        data["formula_df"],
        data["selected_ids"],
        data["herb_list"],
        data["symptom_list"],
    )
    train_mask, val_mask, test_mask = build_split_masks(
        data["selected_ids"],
        data["train_idx"],
        data["val_idx"],
        data["test_idx"],
    )
    trainval_mask = train_mask | val_mask

    Y_train = Y[train_mask]
    Y_val = Y[val_mask]
    Y_test = Y[test_mask]

    print("[3/6] Selecting the TCM-ES MLP training epoch on validation Recall@20...", flush=True)
    F_train, F_val = zscore_poly2_fit_transform(
        data["X"][train_mask],
        data["X"][val_mask],
    )
    _, validation_history, best_info = train_mlp_with_validation(
        F_train,
        Y_train,
        F_val,
        Y_val,
        args,
        device,
    )
    validation_history.to_csv(
        os.path.join(args.out_dir, "validation_training_history_tcm_es_mlp.csv"),
        index=False,
    )

    print("[4/6] Fitting final readouts on train+validation records...", flush=True)
    Y_fit = Y[trainval_mask]
    Y_test = Y[test_mask]
    S_fit = S[trainval_mask].astype(np.float32)
    S_test = S[test_mask].astype(np.float32)

    F_fit, F_test = zscore_poly2_fit_transform(
        data["X"][trainval_mask],
        data["X"][test_mask],
    )

    mlp_final, final_history = fit_mlp_fixed_epochs(
        F_fit,
        Y_fit,
        max(1, int(best_info["best_epoch"])),
        args,
        device,
    )
    final_history.to_csv(
        os.path.join(args.out_dir, "final_refit_history_tcm_es_mlp.csv"),
        index=False,
    )

    scores_mlp = predict_model(mlp_final, F_test, device, args.batch_size)

    W_tcm_es_ridge = fit_ridge_multilabel(F_fit, Y_fit, args.ridge_lambda)
    scores_tcm_es_ridge = predict_ridge(F_test, W_tcm_es_ridge)

    W_multihot = fit_ridge_multilabel(S_fit, Y_fit, args.ridge_lambda)
    scores_multihot = predict_ridge(S_test, W_multihot)

    print("[5/6] Evaluating the shuffled-embedding control...", flush=True)
    _, shuffled_summary, shuffled_detail = fit_predict_shuffled_embedding_ridge(
        X_fit_raw=data["X"][trainval_mask],
        Y_fit=Y_fit,
        X_test_raw=data["X"][test_mask],
        Y_test=Y_test,
        ridge_lambda=args.ridge_lambda,
        repeats=args.random_control_repeats,
        seed=args.seed,
        k_values=k_values,
    )
    shuffled_detail.to_csv(
        os.path.join(args.out_dir, "shuffled_embedding_baseline_repeats.csv"),
        index=False,
    )

    summary = pd.DataFrame([
        evaluate_scores(scores_mlp, Y_test, MODEL_MLP, k_values),
        evaluate_scores(scores_tcm_es_ridge, Y_test, MODEL_RIDGE, k_values),
        evaluate_scores(scores_multihot, Y_test, MODEL_MULTIHOT, k_values),
        shuffled_summary,
    ])

    print("[6/6] Saving manuscript-aligned metrics and figure data...", flush=True)
    summary.to_csv(
        os.path.join(args.out_dir, "heldout_formula_prediction_full_metrics.csv"),
        index=False,
    )
    save_summary_table(summary, args.out_dir)
    plot_metric_comparison(summary, args.out_dir)

    metadata = {
        "input_paths": data["paths"],
        "n_selected_formula_records": int(len(data["selected_ids"])),
        "n_train_records": int(train_mask.sum()),
        "n_validation_records": int(val_mask.sum()),
        "n_test_records": int(test_mask.sum()),
        "n_final_fit_records": int(trainval_mask.sum()),
        "feature_definition": "standardized [z_TCM-ES, z_TCM-ES^2]",
        "models": MODEL_ORDER,
        "best_epoch_from_validation": int(best_info["best_epoch"]),
        "best_validation_R@20": float(best_info["best_val_R@20"]),
        "settings": {
            "k_values": k_values,
            "ridge_lambda": float(args.ridge_lambda),
            "random_control_repeats": int(args.random_control_repeats),
            "hidden1": int(args.hidden1),
            "hidden2": int(args.hidden2),
            "dropout": float(args.dropout),
            "optimizer": "AdamW",
            "loss": "BCEWithLogitsLoss",
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
        },
        "interpretation": (
            "Retrospective observed-herb recovery from frozen symptom-pattern representations; "
            "not prospective clinical prescription recommendation."
        ),
    }
    with open(os.path.join(args.out_dir, "run_metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    print(f"Results saved to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
