#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Attention-guided token-removal perturbation analysis for TCM-ES.

The analysis uses cross-attention scores extracted from held-out test records.
Symptoms or herbs are removed according to high-attention, random, or
low-attention rankings, and the frozen model is rerun to quantify the change in
opposite-modality reconstruction negative log-likelihood (NLL).

Two directions are evaluated: symptom removal for formula reconstruction and
herb removal for symptom-pattern reconstruction.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


# -----------------------------------------------------------------------------
# Project import setup
# -----------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

# Resolve the repository root used for local imports.
PROJECT_ROOT_CANDIDATES = [SCRIPT_DIR, SCRIPT_DIR.parent]
for candidate in PROJECT_ROOT_CANDIDATES:
    if (candidate / "model.py").exists():
        PROJECT_ROOT = candidate
        break
else:
    PROJECT_ROOT = SCRIPT_DIR

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bin"))

# Required so that torch.load can resolve the pickled model class.
try:
    import model  # noqa: F401
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import model.py. Place this script in the project root or "
        "project/bin, or add the project root to PYTHONPATH."
    ) from exc


# -----------------------------------------------------------------------------
# Constants and styles
# -----------------------------------------------------------------------------

DIRECTION_SYMPTOM = "symptom_removal"
DIRECTION_HERB = "herb_removal"

STRATEGY_TOP = "top_attention"
STRATEGY_RANDOM = "random"
STRATEGY_LOW = "low_attention"
STRATEGY_ORDER = [STRATEGY_TOP, STRATEGY_RANDOM, STRATEGY_LOW]

STRATEGY_LABELS = {
    STRATEGY_TOP: "Top-attention removal",
    STRATEGY_RANDOM: "Random removal",
    STRATEGY_LOW: "Low-attention removal",
}

STRATEGY_COLORS = {
    STRATEGY_TOP: "#D95F5F",
    STRATEGY_RANDOM: "#7A7A7A",
    STRATEGY_LOW: "#4C78A8",
}


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_pickle(path: str | Path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def require_file(path: str | Path, label: str) -> str:
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(text: str) -> torch.device:
    if text.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA is unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(text)


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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig: plt.Figure, path_png: str | Path) -> None:
    path_png = str(path_png)
    fig.savefig(path_png, dpi=600, bbox_inches="tight")
    stem = path_png[:-4] if path_png.lower().endswith(".png") else path_png
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".svg", bbox_inches="tight")


def parse_int_list(text: str) -> List[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or any(x < 1 for x in values):
        raise ValueError("--mask-counts must contain positive integers.")
    return values


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(int(n_boot), len(values)))
    boot = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out


def paired_greater_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Paired one-sided test of a > b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return np.nan

    diff = a - b
    if np.allclose(diff, 0.0):
        return 1.0

    if wilcoxon is not None:
        try:
            return float(
                wilcoxon(
                    a,
                    b,
                    alternative="greater",
                    zero_method="wilcox",
                    correction=False,
                ).pvalue
            )
        except ValueError:
            return 1.0

    # Fallback: paired sign-flip permutation test.
    rng = np.random.default_rng(42)
    observed = float(diff.mean())
    n_perm = 10000
    exceed = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diff))
        exceed += float(np.mean(diff * signs) >= observed)
    return float((exceed + 1) / (n_perm + 1))


# -----------------------------------------------------------------------------
# Attention-file loading
# -----------------------------------------------------------------------------

def load_attention_records(path: str | Path) -> Tuple[List[Dict], Dict]:
    """
    Supports:
      1. one combined PKL containing {'records': [...], ...};
      2. one PKL containing a list of records;
      3. a directory containing cross_attention_*.pkl batch files.
    """
    path = Path(path)

    if path.is_dir():
        files = sorted(path.glob("cross_attention_*.pkl"))
        if not files and (path / "batches").is_dir():
            files = sorted((path / "batches").glob("cross_attention_*.pkl"))
        if not files:
            raise FileNotFoundError(f"No cross_attention_*.pkl files in {path}")

        records: List[Dict] = []
        for file in files:
            obj = load_pickle(file)
            if isinstance(obj, dict) and "records" in obj:
                records.extend(obj["records"])
            elif isinstance(obj, list):
                records.extend(obj)
            else:
                raise TypeError(f"Unsupported attention object in {file}")
        return records, {"source": str(path), "format": "batch_directory"}

    obj = load_pickle(path)
    if isinstance(obj, dict) and "records" in obj:
        metadata = {k: v for k, v in obj.items() if k != "records"}
        return list(obj["records"]), metadata
    if isinstance(obj, list):
        return obj, {"source": str(path), "format": "record_list"}

    raise TypeError(
        "Attention file must be a record list or a dictionary containing 'records'."
    )


def get_record_array(record: Dict, candidate_keys: Sequence[str], label: str) -> np.ndarray:
    for key in candidate_keys:
        if key in record:
            return np.asarray(record[key], dtype=np.float32)
    raise KeyError(f"Record is missing {label}; tried keys: {candidate_keys}")


def standardize_attention_record(record: Dict, index: int) -> Dict:
    symptoms = list(record.get("symptoms", []))
    herbs = list(record.get("herbs", []))

    symptom_importance = get_record_array(
        record,
        [
            "symptom_importance",
            "symptom_importance_for_formula_last_layer",
        ],
        "symptom importance",
    ).reshape(-1)

    herb_importance = get_record_array(
        record,
        [
            "herb_importance",
            "herb_importance_for_symptom_last_layer",
        ],
        "herb importance",
    ).reshape(-1)

    if len(symptoms) != len(symptom_importance):
        raise ValueError(
            f"Record {index}: {len(symptoms)} symptoms but "
            f"{len(symptom_importance)} symptom importance values."
        )
    if len(herbs) != len(herb_importance):
        raise ValueError(
            f"Record {index}: {len(herbs)} herbs but "
            f"{len(herb_importance)} herb importance values."
        )

    return {
        "attention_record_index": int(index),
        "formula_row_id": int(record.get("formula_row_id", index)),
        "formula_record_id": record.get("formula_record_id", None),
        "formula_title": record.get("formula_title", None),
        "symptoms": symptoms,
        "herbs": herbs,
        "symptom_importance": symptom_importance,
        "herb_importance": herb_importance,
    }


# -----------------------------------------------------------------------------
# Sequence preparation
# -----------------------------------------------------------------------------

def remove_indices(tokens: Sequence[str], indices: Iterable[int]) -> List[str]:
    remove_set = set(int(x) for x in indices)
    return [token for i, token in enumerate(tokens) if i not in remove_set]


def choose_removed_indices(
    importance: np.ndarray,
    count: int,
    strategy: str,
    rng: np.random.Generator | None,
) -> np.ndarray:
    n = len(importance)
    if count >= n:
        raise ValueError("At least one token must remain after perturbation.")

    if strategy == STRATEGY_TOP:
        return np.argsort(-importance, kind="stable")[:count]
    if strategy == STRATEGY_LOW:
        return np.argsort(importance, kind="stable")[:count]
    if strategy == STRATEGY_RANDOM:
        if rng is None:
            raise ValueError("Random strategy requires an RNG.")
        return np.sort(rng.choice(n, size=count, replace=False))
    raise ValueError(f"Unknown strategy: {strategy}")


def build_model_batch(
    symptom_token_sets: Sequence[Sequence[str]],
    herb_token_sets: Sequence[Sequence[str]],
    symptom2id: Dict[str, int],
    herb2id: Dict[str, int],
    symptom_semantics: np.ndarray,
    max_sym_len: int,
    max_herb_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    n_sym = len(symptom2id)
    n_herb = len(herb2id)

    sym_pad_id = n_sym
    sym_start_id = n_sym + 1
    sym_end_id = n_sym + 2

    herb_pad_id = n_herb
    herb_start_id = n_herb + 1
    herb_end_id = n_herb + 2

    symptom_inputs = []
    herb_inputs = []
    symptom_targets = []
    herb_targets = []
    symptom_target_lengths = []
    herb_target_lengths = []

    for symptoms, herbs in zip(symptom_token_sets, herb_token_sets):
        sym_ids = [symptom2id[x] for x in symptoms if x in symptom2id]
        herb_ids = [herb2id[x] for x in herbs if x in herb2id]

        if len(sym_ids) > max_sym_len - 1:
            sym_ids = sym_ids[: max_sym_len - 1]
        if len(herb_ids) > max_herb_len - 1:
            herb_ids = herb_ids[: max_herb_len - 1]

        sym_input_ids = [sym_start_id] + sym_ids
        sym_input_ids += [sym_pad_id] * (max_sym_len - len(sym_input_ids))
        symptom_inputs.append([symptom_semantics[i] for i in sym_input_ids])

        herb_input_ids = [herb_start_id] + herb_ids
        herb_input_ids += [herb_pad_id] * (max_herb_len - len(herb_input_ids))
        herb_inputs.append(herb_input_ids)

        sym_target = sym_ids + [sym_end_id]
        symptom_target_lengths.append(len(sym_target))
        sym_target += [sym_pad_id] * (max_sym_len - len(sym_target))
        symptom_targets.append(sym_target)

        herb_target = herb_ids + [herb_end_id]
        herb_target_lengths.append(len(herb_target))
        herb_target += [herb_pad_id] * (max_herb_len - len(herb_target))
        herb_targets.append(herb_target)

    return {
        "symptom_input": torch.tensor(
            np.asarray(symptom_inputs), dtype=torch.float32, device=device
        ),
        "herb_input": torch.tensor(
            np.asarray(herb_inputs), dtype=torch.long, device=device
        ),
        "symptom_target": torch.tensor(
            np.asarray(symptom_targets), dtype=torch.long, device=device
        ),
        "herb_target": torch.tensor(
            np.asarray(herb_targets), dtype=torch.long, device=device
        ),
        "symptom_target_length": torch.tensor(
            symptom_target_lengths, dtype=torch.long, device=device
        ),
        "herb_target_length": torch.tensor(
            herb_target_lengths, dtype=torch.long, device=device
        ),
    }


def token_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Mean negative log-likelihood over real target tokens plus <E>."""
    log_prob = F.log_softmax(logits, dim=-1)
    chosen = log_prob.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    positions = torch.arange(logits.shape[1], device=logits.device)[None, :]
    mask = positions < lengths[:, None]

    loss = -(chosen * mask).sum(dim=1) / lengths.clamp(min=1)
    return loss


@torch.no_grad()
def infer_direction(
    trained_model: torch.nn.Module,
    symptom_token_sets: Sequence[Sequence[str]],
    herb_token_sets: Sequence[Sequence[str]],
    direction: str,
    symptom2id: Dict[str, int],
    herb2id: Dict[str, int],
    symptom_semantics: np.ndarray,
    max_sym_len: int,
    max_herb_len: int,
    device: torch.device,
    inference_batch_size: int,
) -> np.ndarray:
    """Return per-record reconstruction NLL for one cross-modal direction."""
    nll_parts = []

    for start in range(0, len(symptom_token_sets), inference_batch_size):
        end = min(start + inference_batch_size, len(symptom_token_sets))
        batch = build_model_batch(
            symptom_token_sets[start:end],
            herb_token_sets[start:end],
            symptom2id,
            herb2id,
            symptom_semantics,
            max_sym_len,
            max_herb_len,
            device,
        )

        if direction == DIRECTION_SYMPTOM:
            # Symptoms are the perturbed input; reconstruct the unchanged herbs.
            _, herb_logits, _, _, _ = trained_model(
                batch["symptom_input"],
                batch["herb_input"],
                mask_input="herb",
            )
            nll = token_nll(
                herb_logits,
                batch["herb_target"],
                batch["herb_target_length"],
            )

        elif direction == DIRECTION_HERB:
            # Herbs are the perturbed input; reconstruct the unchanged symptoms.
            symptom_logits, _, _, _, _ = trained_model(
                batch["symptom_input"],
                batch["herb_input"],
                mask_input="symptom",
            )
            nll = token_nll(
                symptom_logits,
                batch["symptom_target"],
                batch["symptom_target_length"],
            )
        else:
            raise ValueError(direction)

        nll_parts.append(nll.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(nll_parts)


# -----------------------------------------------------------------------------
# Core perturbation analysis
# -----------------------------------------------------------------------------

def run_perturbation_analysis(
    records: List[Dict],
    trained_model: torch.nn.Module,
    symptom2id: Dict[str, int],
    herb2id: Dict[str, int],
    symptom_semantics: np.ndarray,
    mask_counts: Sequence[int],
    random_repeats: int,
    seed: int,
    outer_batch_size: int,
    inference_batch_size: int,
    max_sym_len: int,
    max_herb_len: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: List[Dict] = []
    baseline_rows: List[Dict] = []

    trained_model.eval()

    for batch_start in range(0, len(records), outer_batch_size):
        batch_end = min(batch_start + outer_batch_size, len(records))
        batch_records = records[batch_start:batch_end]

        print(
            f"Perturbation records: {batch_start}-{batch_end}/{len(records)}",
            flush=True,
        )

        full_symptoms = [r["symptoms"] for r in batch_records]
        full_herbs = [r["herbs"] for r in batch_records]

        # Baseline cross-modal reconstruction losses for complete records.
        base_formula_nll = infer_direction(
            trained_model,
            full_symptoms,
            full_herbs,
            DIRECTION_SYMPTOM,
            symptom2id,
            herb2id,
            symptom_semantics,
            max_sym_len,
            max_herb_len,
            device,
            inference_batch_size,
        )
        base_symptom_nll = infer_direction(
            trained_model,
            full_symptoms,
            full_herbs,
            DIRECTION_HERB,
            symptom2id,
            herb2id,
            symptom_semantics,
            max_sym_len,
            max_herb_len,
            device,
            inference_batch_size,
        )

        for local_i, record in enumerate(batch_records):
            baseline_rows.append({
                "attention_record_index": record["attention_record_index"],
                "formula_row_id": record["formula_row_id"],
                "formula_record_id": record["formula_record_id"],
                "formula_title": record["formula_title"],
                "n_symptoms": len(record["symptoms"]),
                "n_herbs": len(record["herbs"]),
                "baseline_formula_reconstruction_nll": float(base_formula_nll[local_i]),
                "baseline_symptom_reconstruction_nll": float(base_symptom_nll[local_i]),
            })

        for direction in [DIRECTION_SYMPTOM, DIRECTION_HERB]:
            for mask_count in mask_counts:
                valid_local_indices = []
                for local_i, record in enumerate(batch_records):
                    n_tokens = (
                        len(record["symptoms"])
                        if direction == DIRECTION_SYMPTOM
                        else len(record["herbs"])
                    )
                    if n_tokens > mask_count:
                        valid_local_indices.append(local_i)

                if not valid_local_indices:
                    continue

                variant_symptoms: List[List[str]] = []
                variant_herbs: List[List[str]] = []
                variant_meta: List[Dict] = []

                for local_i in valid_local_indices:
                    record = batch_records[local_i]
                    importance = (
                        record["symptom_importance"]
                        if direction == DIRECTION_SYMPTOM
                        else record["herb_importance"]
                    )

                    variants = [
                        (STRATEGY_TOP, -1),
                        (STRATEGY_LOW, -1),
                    ] + [
                        (STRATEGY_RANDOM, repeat)
                        for repeat in range(random_repeats)
                    ]

                    for strategy, repeat in variants:
                        rng = None
                        if strategy == STRATEGY_RANDOM:
                            rng = np.random.default_rng(
                                seed
                                + int(record["formula_row_id"]) * 10007
                                + int(mask_count) * 101
                                + int(repeat)
                            )

                        removed = choose_removed_indices(
                            importance,
                            mask_count,
                            strategy,
                            rng,
                        )

                        if direction == DIRECTION_SYMPTOM:
                            perturbed_symptoms = remove_indices(
                                record["symptoms"], removed
                            )
                            perturbed_herbs = list(record["herbs"])
                        else:
                            perturbed_symptoms = list(record["symptoms"])
                            perturbed_herbs = remove_indices(
                                record["herbs"], removed
                            )

                        variant_symptoms.append(perturbed_symptoms)
                        variant_herbs.append(perturbed_herbs)
                        variant_meta.append({
                            "local_i": local_i,
                            "strategy": strategy,
                            "random_repeat": repeat,
                            "removed_indices": removed,
                        })

                perturbed_nll = infer_direction(
                    trained_model,
                    variant_symptoms,
                    variant_herbs,
                    direction,
                    symptom2id,
                    herb2id,
                    symptom_semantics,
                    max_sym_len,
                    max_herb_len,
                    device,
                    inference_batch_size,
                )

                for variant_i, meta in enumerate(variant_meta):
                    local_i = int(meta["local_i"])
                    record = batch_records[local_i]

                    if direction == DIRECTION_SYMPTOM:
                        baseline_nll = float(base_formula_nll[local_i])
                        removed_tokens = [
                            record["symptoms"][j]
                            for j in meta["removed_indices"]
                        ]
                    else:
                        baseline_nll = float(base_symptom_nll[local_i])
                        removed_tokens = [
                            record["herbs"][j]
                            for j in meta["removed_indices"]
                        ]

                    detail_rows.append({
                        "attention_record_index": record["attention_record_index"],
                        "formula_row_id": record["formula_row_id"],
                        "formula_record_id": record["formula_record_id"],
                        "formula_title": record["formula_title"],
                        "direction": direction,
                        "mask_count": int(mask_count),
                        "strategy": meta["strategy"],
                        "random_repeat": int(meta["random_repeat"]),
                        "n_original_tokens": (
                            len(record["symptoms"])
                            if direction == DIRECTION_SYMPTOM
                            else len(record["herbs"])
                        ),
                        "removed_tokens": "；".join(map(str, removed_tokens)),
                        "baseline_reconstruction_nll": baseline_nll,
                        "perturbed_reconstruction_nll": float(perturbed_nll[variant_i]),
                        "delta_reconstruction_nll": float(
                            perturbed_nll[variant_i] - baseline_nll
                        ),
                    })

    return pd.DataFrame(detail_rows), pd.DataFrame(baseline_rows)


# -----------------------------------------------------------------------------
# Summaries, paired tests, and figures
# -----------------------------------------------------------------------------

def aggregate_random_repeats(detail: pd.DataFrame) -> pd.DataFrame:
    """Average random-removal repeats within each record before paired testing."""
    group_cols = [
        "attention_record_index",
        "formula_row_id",
        "direction",
        "mask_count",
        "strategy",
    ]
    numeric_cols = [
        "baseline_reconstruction_nll",
        "perturbed_reconstruction_nll",
        "delta_reconstruction_nll",
    ]

    return (
        detail.groupby(group_cols, as_index=False)[numeric_cols]
        .mean()
    )


def summarize_record_level(
    record_level: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows = []

    for (direction, mask_count, strategy), sub in record_level.groupby(
        ["direction", "mask_count", "strategy"], sort=False
    ):
        mean, lo, hi = bootstrap_mean_ci(
            sub["delta_reconstruction_nll"].values,
            n_boot,
            seed + int(mask_count) * 101,
        )
        rows.append({
            "direction": direction,
            "mask_count": int(mask_count),
            "strategy": strategy,
            "n_records": int(len(sub)),
            "delta_reconstruction_nll_mean": mean,
            "delta_reconstruction_nll_ci_low": lo,
            "delta_reconstruction_nll_ci_high": hi,
        })

    return pd.DataFrame(rows)


def run_paired_tests(
    record_level: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Paired one-sided tests of top-attention removal versus controls."""
    rows = []

    for (direction, mask_count), sub in record_level.groupby(
        ["direction", "mask_count"], sort=False
    ):
        pivot = sub.pivot_table(
            index="formula_row_id",
            columns="strategy",
            values="delta_reconstruction_nll",
            aggfunc="mean",
        )

        for comparison, reference in [
            ("top_vs_random", STRATEGY_RANDOM),
            ("top_vs_low", STRATEGY_LOW),
        ]:
            needed = [STRATEGY_TOP, reference]
            if not all(x in pivot.columns for x in needed):
                continue

            paired = pivot[needed].dropna()
            top = paired[STRATEGY_TOP].to_numpy(dtype=float)
            ref = paired[reference].to_numpy(dtype=float)
            difference = top - ref

            mean_diff, lo, hi = bootstrap_mean_ci(
                difference,
                n_boot,
                seed + int(mask_count) * 1009,
            )

            rows.append({
                "direction": direction,
                "mask_count": int(mask_count),
                "metric": "delta_reconstruction_nll",
                "comparison": comparison,
                "n_pairs": int(len(paired)),
                "top_mean": float(np.mean(top)),
                "reference_mean": float(np.mean(ref)),
                "mean_paired_difference": mean_diff,
                "difference_ci_low": lo,
                "difference_ci_high": hi,
                "p_one_sided_top_greater": paired_greater_pvalue(top, ref),
            })

    out = pd.DataFrame(rows)
    if len(out):
        out["p_fdr_bh"] = bh_fdr(out["p_one_sided_top_greater"].values)
    return out


def plot_perturbation_summary(summary: pd.DataFrame, out_dir: str | Path) -> None:
    """Create the two-panel cross-modal reconstruction figure."""
    set_publication_style()

    panels = [
        (
            DIRECTION_SYMPTOM,
            "Symptom removal → formula reconstruction",
            "Increase in formula reconstruction NLL",
        ),
        (
            DIRECTION_HERB,
            "Herb removal → symptom reconstruction",
            "Increase in symptom reconstruction NLL",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), squeeze=False)
    axes = axes.ravel()

    for panel_idx, (direction, title, ylabel) in enumerate(panels):
        ax = axes[panel_idx]
        panel_data = summary[summary["direction"] == direction].copy()

        for strategy in STRATEGY_ORDER:
            sub = (
                panel_data[panel_data["strategy"] == strategy]
                .sort_values("mask_count")
            )
            if len(sub) == 0:
                continue

            x = sub["mask_count"].to_numpy(dtype=float)
            y = sub["delta_reconstruction_nll_mean"].to_numpy(dtype=float)
            lo = sub["delta_reconstruction_nll_ci_low"].to_numpy(dtype=float)
            hi = sub["delta_reconstruction_nll_ci_high"].to_numpy(dtype=float)

            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                marker="o",
                markersize=4.2,
                linewidth=1.4,
                capsize=2.2,
                label=STRATEGY_LABELS[strategy],
                color=STRATEGY_COLORS[strategy],
            )

        ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.7)
        ax.set_title(title, pad=7)
        ax.set_xlabel("Number of removed tokens")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.text(
            -0.14,
            1.08,
            chr(ord("A") + panel_idx),
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Attention-guided cross-modal perturbation of held-out symptom–formula records",
        fontsize=10.5,
        y=0.99,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.82,
        bottom=0.24,
        wspace=0.34,
    )

    save_figure(
        fig,
        Path(out_dir) / "fig_attention_perturbation_reconstruction.png",
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attention-guided perturbation analysis for TCM-ES."
    )

    parser.add_argument(
        "--attention-file",
        default="results/attention/original/test/cross_attention_test_all.pkl",
        help=(
            "Combined attention PKL, a record-list PKL, or a directory of "
            "cross_attention_*.pkl batch files."
        ),
    )
    parser.add_argument(
        "--model-file",
        default="core/trained_model/model_epoch_60.pkl",
    )
    parser.add_argument(
        "--symptom-list",
        default="core/standard_TCM_entities/symptom_list.pkl",
    )
    parser.add_argument(
        "--herb-list",
        default="core/standard_TCM_entities/herb_list.pkl",
    )
    parser.add_argument(
        "--symptom-semantics",
        default="core/standard_TCM_entities/symptom_semantic_encodings.pkl",
    )
    parser.add_argument(
        "--test-idx",
        default="data/training/test_idx.pkl",
        help="Used only to verify that saved formula_row_id values belong to test.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/attention/original/test/perturbation_analysis",
    )

    parser.add_argument(
        "--mask-counts",
        default="1,2,3",
        help="Comma-separated numbers of symptoms/herbs removed per record.",
    )
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--outer-batch-size", type=int, default=128)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--max-sym-len", type=int, default=41)
    parser.add_argument("--max-herb-len", type=int, default=31)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    attention_path = require_file(args.attention_file, "attention file/directory")
    model_path = require_file(args.model_file, "trained model")
    symptom_list_path = require_file(args.symptom_list, "symptom list")
    herb_list_path = require_file(args.herb_list, "herb list")
    symptom_semantics_path = require_file(
        args.symptom_semantics, "symptom semantic encodings"
    )

    raw_records, attention_metadata = load_attention_records(attention_path)
    records = [
        standardize_attention_record(record, i)
        for i, record in enumerate(raw_records)
    ]


    symptom_list = load_pickle(symptom_list_path)
    herb_list = load_pickle(herb_list_path)
    symptom_semantics = np.asarray(
        load_pickle(symptom_semantics_path), dtype=np.float32
    )

    symptom2id = {x: i for i, x in enumerate(symptom_list)}
    herb2id = {x: i for i, x in enumerate(herb_list)}

    expected_semantic_rows = len(symptom_list) + 3
    if symptom_semantics.shape[0] < expected_semantic_rows:
        raise ValueError(
            f"Symptom semantic rows {symptom_semantics.shape[0]} < "
            f"required {expected_semantic_rows}."
        )

    # Verify held-out test-set membership when the split file is available.
    split_check = {}
    if os.path.exists(args.test_idx):
        test_ids = set(int(x) for x in load_pickle(args.test_idx))
        record_ids = [int(r["formula_row_id"]) for r in records]
        outside = [x for x in record_ids if x not in test_ids]
        split_check = {
            "n_records": len(record_ids),
            "n_outside_test_split": len(outside),
            "outside_test_examples": outside[:20],
        }
        if outside:
            raise ValueError(
                f"{len(outside)} attention records are outside the held-out test split; "
                f"examples: {outside[:20]}"
            )

    device = resolve_device(args.device)
    print("Device:", device, flush=True)
    print("Loading model:", model_path, flush=True)
    trained_model = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    trained_model.to(device)
    trained_model.eval()

    mask_counts = parse_int_list(args.mask_counts)

    print("Attention records:", len(records), flush=True)
    print("Mask counts:", mask_counts, flush=True)
    print("Random repeats:", args.random_repeats, flush=True)

    detail, baseline = run_perturbation_analysis(
        records=records,
        trained_model=trained_model,
        symptom2id=symptom2id,
        herb2id=herb2id,
        symptom_semantics=symptom_semantics,
        mask_counts=mask_counts,
        random_repeats=int(args.random_repeats),
        seed=int(args.seed),
        outer_batch_size=int(args.outer_batch_size),
        inference_batch_size=int(args.inference_batch_size),
        max_sym_len=int(args.max_sym_len),
        max_herb_len=int(args.max_herb_len),
        device=device,
    )

    record_level = aggregate_random_repeats(detail)
    summary = summarize_record_level(
        record_level,
        n_boot=int(args.bootstrap_repeats),
        seed=int(args.seed),
    )
    paired_tests = run_paired_tests(
        record_level,
        n_boot=int(args.bootstrap_repeats),
        seed=int(args.seed),
    )

    # Large raw table is compressed to reduce disk usage.
    detail.to_csv(
        Path(args.out_dir) / "attention_perturbation_repeat_level.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8",
    )
    baseline.to_csv(
        Path(args.out_dir) / "attention_perturbation_baseline.csv",
        index=False,
        encoding="utf_8_sig",
    )
    record_level.to_csv(
        Path(args.out_dir) / "attention_perturbation_record_level.csv",
        index=False,
        encoding="utf_8_sig",
    )
    summary.to_csv(
        Path(args.out_dir) / "attention_perturbation_summary.csv",
        index=False,
        encoding="utf_8_sig",
    )
    paired_tests.to_csv(
        Path(args.out_dir) / "attention_perturbation_paired_tests.csv",
        index=False,
        encoding="utf_8_sig",
    )

    plot_perturbation_summary(summary, args.out_dir)

    config = vars(args).copy()
    config.update({
        "attention_metadata": attention_metadata,
        "split_check": split_check,
        "n_loaded_records": len(records),
        "analysis_definition": {
            "symptom_removal": (
                "Remove symptoms ranked by saved cross-attention; evaluate the "
                "increase in formula reconstruction NLL."
            ),
            "herb_removal": (
                "Remove herbs ranked by saved cross-attention; evaluate the "
                "increase in symptom reconstruction NLL."
            ),
            "random_control": (
                "Randomly remove the same number of tokens; average within "
                "record over random repeats before paired tests."
            ),
        },
    })
    with open(Path(args.out_dir) / "run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp, ensure_ascii=False, indent=2, default=str)

    print("\nCompleted. Outputs:", flush=True)
    for name in [
        "fig_attention_perturbation_reconstruction.png",
        "attention_perturbation_summary.csv",
        "attention_perturbation_paired_tests.csv",
        "attention_perturbation_record_level.csv",
    ]:
        print(" ", Path(args.out_dir) / name, flush=True)



if __name__ == "__main__":
    main()
