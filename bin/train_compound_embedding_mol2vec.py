#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train herbal-compound embeddings aligned to fixed TCM herb embeddings.

The compound mapper follows the historical Autoencoder 3 objective: continuous
Mol2Vec structural vectors are reconstructed with mean-squared error, while the
256-dimensional bottleneck is aligned to associated herbs with contrastive
loss. For each compound, one associated herb and one valid non-associated herb
are sampled during each loader visit. The structural input dimension is read
from the prepared Mol2Vec feature matrix unless explicitly specified and validated.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, TensorDataset

from model import ContrastiveLoss, Multimodal_AE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train compound embeddings aligned to a TCM herb-embedding space."
    )
    parser.add_argument(
        "--embedding-dir",
        required=True,
        help="Directory containing individual_herb_embeddings.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the trained model, losses, and compound embeddings.",
    )
    parser.add_argument(
        "--herb-embedding-file",
        default="individual_herb_embeddings.pkl",
    )
    parser.add_argument(
        "--herb-list",
        default="core/standard_TCM_entities/herb_list.pkl",
    )
    parser.add_argument(
        "--compound-list",
        default="data/herb_compounds/compound_list.pkl",
    )
    parser.add_argument(
        "--herb-compounds",
        default="data/herb_compounds/herb_compounds.pkl",
    )
    parser.add_argument(
        "--compound-features",
        default="data/herb_compounds/compound_mol2vec_vectors.pkl",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help=(
            "Compound structural-vector dimension. If omitted, the dimension "
            "is detected from the prepared Mol2Vec feature matrix. If provided, "
            "it must match the feature matrix."
        ),
    )
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--print-every", type=int, default=1)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def load_pickle(path: Path):
    with path.open("rb") as fp:
        return pickle.load(fp)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def load_compound_feature_matrix(
    feature_path: Path,
    compound_list: Sequence[str],
) -> Tuple[list[str], np.ndarray, pd.DataFrame, Dict[str, object]]:
    """Load supported Mol2Vec vectors in canonical compound-list order."""
    obj = load_pickle(feature_path)
    if not isinstance(obj, Mapping) or "features" not in obj or "compound_names" not in obj:
        raise ValueError(
            "compound-features must be the payload produced by "
            "prepare_compound_mol2vec_features.py."
        )

    names = list(obj["compound_names"])
    features = np.asarray(obj["features"], dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("Mol2Vec feature matrix must be 2-D.")
    if features.shape[0] != len(names):
        raise ValueError("Mol2Vec feature rows do not match compound_names.")
    if len(names) != len(set(names)):
        raise ValueError("Mol2Vec feature payload contains duplicate compound_names.")
    if not np.isfinite(features).all():
        raise ValueError("Mol2Vec feature matrix contains non-finite values.")

    canonical_set = set(compound_list)
    extra = [name for name in names if name not in canonical_set]
    if extra:
        raise ValueError(
            f"Mol2Vec payload contains {len(extra)} compounds outside compound_list. "
            f"Examples: {extra[:10]}"
        )

    positions = {name: i for i, name in enumerate(names)}
    supported_names = [name for name in compound_list if name in positions]
    if not supported_names:
        raise ValueError("No compound_list entries have usable Mol2Vec vectors.")

    order = np.asarray([positions[name] for name in supported_names], dtype=int)
    matrix = features[order].astype(np.float32, copy=False)

    smiles_all = list(obj.get("smiles", [None] * len(names)))
    source_rows_all = list(obj.get("source_rows", range(len(names))))
    duplicate_counts_all = list(obj.get("duplicate_name_counts", [1] * len(names)))
    mapping = pd.DataFrame(
        {
            "compound": supported_names,
            "smiles": [smiles_all[i] for i in order],
            "source_row": [int(source_rows_all[i]) for i in order],
            "duplicate_name_count": [int(duplicate_counts_all[i]) for i in order],
        }
    )

    saved_dim = obj.get("feature_dim")
    if saved_dim is not None and int(saved_dim) != matrix.shape[1]:
        raise ValueError(
            f"Saved feature_dim ({saved_dim}) does not match matrix dimension "
            f"({matrix.shape[1]})."
        )

    metadata = {
        "feature_method": obj.get("feature_method", "unknown"),
        "feature_dim": int(matrix.shape[1]),
        "canonical_compound_count": int(len(compound_list)),
        "supported_compound_count": int(len(supported_names)),
        "dropped_compound_count": int(len(compound_list) - len(supported_names)),
    }
    return supported_names, matrix, mapping, metadata


def build_herb_labels(
    compound_list: Sequence[str],
    herb_list: Sequence[str],
    herb_compounds: Mapping[str, Iterable[str]],
) -> np.ndarray:
    """Build compound × herb labels with 1/0/-1 annotation semantics.

    1 denotes a documented compound-herb association, 0 denotes an annotated
    herb without that compound, and -1 denotes a herb without compound
    annotations and excludes it from contrastive sampling.
    """

    compound_index = {compound: i for i, compound in enumerate(compound_list)}
    if len(compound_index) != len(compound_list):
        raise ValueError("compound_list contains duplicate names.")

    labels = np.full((len(compound_list), len(herb_list)), -1, dtype=np.int8)

    for herb_idx, herb in enumerate(herb_list):
        compounds = list(herb_compounds.get(herb, []))
        if not compounds:
            continue

        labels[:, herb_idx] = 0
        for compound in compounds:
            compound_idx = compound_index.get(compound)
            if compound_idx is not None:
                labels[compound_idx, herb_idx] = 1

    return labels


def select_training_compounds(
    compound_names: Sequence[str],
    compound_features: np.ndarray,
    herb_labels: np.ndarray,
) -> Tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Keep compounds with at least one eligible positive and negative herb."""
    if compound_features.shape[0] != len(compound_names):
        raise ValueError("Compound names and features are not row-aligned.")
    if herb_labels.shape[0] != len(compound_names):
        raise ValueError("Compound names and herb labels are not row-aligned.")

    positive = np.any(herb_labels == 1, axis=1)
    negative = np.any(herb_labels == 0, axis=1)
    keep = positive & negative
    kept_names = [name for name, flag in zip(compound_names, keep) if flag]
    return (
        kept_names,
        compound_features[keep],
        herb_labels[keep],
        keep,
    )


def prepare_dataset(
    compound_features: np.ndarray,
    herb_labels: np.ndarray,
) -> TensorDataset:
    if compound_features.shape[0] != herb_labels.shape[0]:
        raise ValueError("Compound features and herb labels are not row-aligned.")
    return TensorDataset(
        torch.as_tensor(compound_features, dtype=torch.float32),
        torch.as_tensor(herb_labels, dtype=torch.int8),
    )


def sample_positive_negative_herbs(
    herb_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample one associated and one valid non-associated herb per compound."""
    positive_ids = []
    negative_ids = []

    for row in herb_labels:
        positive_candidates = torch.where(row == 1)[0]
        negative_candidates = torch.where(row == 0)[0]
        if len(positive_candidates) == 0 or len(negative_candidates) == 0:
            raise ValueError("Each training compound requires positive and negative herbs.")

        positive_ids.append(
            positive_candidates[
                torch.randint(len(positive_candidates), (1,), device=row.device)
            ]
        )
        negative_ids.append(
            negative_candidates[
                torch.randint(len(negative_candidates), (1,), device=row.device)
            ]
        )

    return torch.cat(positive_ids), torch.cat(negative_ids)


def compute_losses(
    reconstructed: torch.Tensor,
    compound_inputs: torch.Tensor,
    compound_embeddings: torch.Tensor,
    herb_labels: torch.Tensor,
    herb_embeddings: torch.Tensor,
    contrastive_loss: ContrastiveLoss,
    mse_loss: nn.MSELoss,
    beta: float,
) -> Dict[str, torch.Tensor]:
    """MSE reconstruction plus sampled compound-herb contrastive alignment."""
    reconstruction = mse_loss(reconstructed, compound_inputs)

    positive_ids, negative_ids = sample_positive_negative_herbs(herb_labels)
    positive_herbs = herb_embeddings[positive_ids]
    negative_herbs = herb_embeddings[negative_ids]

    contrastive = torch.mean(
        contrastive_loss(compound_embeddings, positive_herbs, 0)
        + contrastive_loss(compound_embeddings, negative_herbs, 1)
    )
    total = reconstruction + beta * contrastive
    return {
        "total": total,
        "reconstruction": reconstruction,
        "contrastive": contrastive,
    }


def run_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    herb_embeddings: torch.Tensor,
    contrastive_loss: ContrastiveLoss,
    mse_loss: nn.MSELoss,
    beta: float,
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {"total": 0.0, "reconstruction": 0.0, "contrastive": 0.0}
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for compound_inputs, herb_labels in loader:
            compound_inputs = compound_inputs.to(device)
            herb_labels = herb_labels.to(device)

            reconstructed, compound_embeddings = model(compound_inputs)
            losses = compute_losses(
                reconstructed=reconstructed,
                compound_inputs=compound_inputs,
                compound_embeddings=compound_embeddings,
                herb_labels=herb_labels,
                herb_embeddings=herb_embeddings,
                contrastive_loss=contrastive_loss,
                mse_loss=mse_loss,
                beta=beta,
            )

            if training:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

            batch_n = compound_inputs.shape[0]
            sample_count += batch_n
            for name in totals:
                totals[name] += losses[name].item() * batch_n

    if sample_count == 0:
        raise ValueError("Empty data loader.")
    return {name: value / sample_count for name, value in totals.items()}


def average_fold_metrics(fold_metrics: list[Dict[str, float]]) -> Dict[str, float]:
    return {
        key: float(np.mean([metrics[key] for metrics in fold_metrics]))
        for key in fold_metrics[0]
    }


def resolve_input_dimension(configured_dim: Optional[int], detected_dim: int) -> int:
    """Use the detected feature dimension unless an explicit matching value is set."""
    detected_dim = int(detected_dim)
    if detected_dim <= 0:
        raise ValueError("Detected compound input dimension must be positive.")
    if configured_dim is None:
        return detected_dim
    configured_dim = int(configured_dim)
    if configured_dim <= 0:
        raise ValueError("--input-dim must be a positive integer.")
    if configured_dim != detected_dim:
        raise ValueError(
            f"Configured input dimension ({configured_dim}) does not match "
            f"the Mol2Vec feature dimension ({detected_dim})."
        )
    return configured_dim


def build_compound_model(input_dim: int, embedding_dim: int) -> nn.Module:
    """Build Autoencoder 3 using the detected Mol2Vec input dimension."""
    return Multimodal_AE(
        input_dim=input_dim,
        en_dims=[512, embedding_dim],
        de_dims=[embedding_dim, 512],
        out_sig=False,
    )


def generate_compound_embeddings(
    model: nn.Module,
    compound_features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(compound_features), batch_size):
            stop = min(start + batch_size, len(compound_features))
            inputs = torch.as_tensor(
                compound_features[start:stop],
                dtype=torch.float32,
                device=device,
            )
            _, embeddings = model(inputs)
            batches.append(embeddings.cpu().numpy())
    return np.concatenate(batches, axis=0).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    herb_embeddings_np = np.asarray(
        load_pickle(embedding_dir / args.herb_embedding_file),
        dtype=np.float32,
    )
    herb_list = list(load_pickle(Path(args.herb_list)))
    canonical_compound_list = list(load_pickle(Path(args.compound_list)))
    herb_compounds = load_pickle(Path(args.herb_compounds))

    if herb_embeddings_np.ndim != 2:
        raise ValueError("Herb embeddings must be a 2-D matrix.")
    if herb_embeddings_np.shape[0] != len(herb_list):
        raise ValueError(
            f"Herb embedding rows ({herb_embeddings_np.shape[0]}) do not match "
            f"herb_list ({len(herb_list)})."
        )
    if not np.isfinite(herb_embeddings_np).all():
        raise ValueError("Herb embeddings contain non-finite values.")

    compound_names, compound_features, feature_mapping, feature_metadata = (
        load_compound_feature_matrix(
            feature_path=Path(args.compound_features),
            compound_list=canonical_compound_list,
        )
    )

    herb_labels_all = build_herb_labels(
        compound_list=compound_names,
        herb_list=herb_list,
        herb_compounds=herb_compounds,
    )
    (
        training_compound_names,
        training_features,
        training_labels,
        training_keep_mask,
    ) = select_training_compounds(
        compound_names=compound_names,
        compound_features=compound_features,
        herb_labels=herb_labels_all,
    )

    if len(training_compound_names) < args.folds:
        raise ValueError(
            f"Only {len(training_compound_names)} compounds are eligible for training; "
            f"cannot use {args.folds} folds."
        )

    dataset = prepare_dataset(training_features, training_labels)
    detected_input_dim = int(compound_features.shape[1])
    input_dim = resolve_input_dimension(args.input_dim, detected_input_dim)
    embedding_dim = int(herb_embeddings_np.shape[1])

    print(f"Device: {device}")
    print(f"Canonical compounds: {len(canonical_compound_list)}")
    print(f"Mol2Vec-supported compounds: {len(compound_names)}")
    print(f"Training-eligible compounds: {len(training_compound_names)}")
    print(f"Herbs: {len(herb_list)}")
    print(f"Compound feature method: {feature_metadata['feature_method']}")
    print(f"Detected compound input dimension: {detected_input_dim}")
    if args.input_dim is not None:
        print(f"Configured compound input dimension: {input_dim}")
    print(f"TCM embedding dimension: {embedding_dim}")
    print(
        "Eligible alignment herbs: "
        f"{int(np.sum(np.any(training_labels != -1, axis=0)))}"
    )

    model = build_compound_model(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    contrastive_loss = ContrastiveLoss()
    mse_loss = nn.MSELoss()
    herb_embeddings = torch.as_tensor(
        herb_embeddings_np,
        dtype=torch.float32,
        device=device,
    )

    kfold = KFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    fold_splits = list(kfold.split(np.arange(len(dataset))))

    history = []
    for epoch in range(1, args.epochs + 1):
        train_fold_metrics = []
        validation_fold_metrics = []

        for train_ids, validation_ids in fold_splits:
            train_loader = DataLoader(
                Subset(dataset, train_ids.tolist()),
                batch_size=args.batch_size,
                shuffle=True,
            )
            validation_loader = DataLoader(
                Subset(dataset, validation_ids.tolist()),
                batch_size=len(validation_ids),
                shuffle=False,
            )

            train_fold_metrics.append(
                run_loader(
                    model=model,
                    loader=train_loader,
                    device=device,
                    herb_embeddings=herb_embeddings,
                    contrastive_loss=contrastive_loss,
                    mse_loss=mse_loss,
                    beta=args.beta,
                    optimizer=optimizer,
                )
            )
            validation_fold_metrics.append(
                run_loader(
                    model=model,
                    loader=validation_loader,
                    device=device,
                    herb_embeddings=herb_embeddings,
                    contrastive_loss=contrastive_loss,
                    mse_loss=mse_loss,
                    beta=args.beta,
                    optimizer=None,
                )
            )

        train_mean = average_fold_metrics(train_fold_metrics)
        validation_mean = average_fold_metrics(validation_fold_metrics)
        history.append(
            {
                "epoch": epoch,
                "train_total": train_mean["total"],
                "train_reconstruction": train_mean["reconstruction"],
                "train_contrastive": train_mean["contrastive"],
                "validation_total": validation_mean["total"],
                "validation_reconstruction": validation_mean["reconstruction"],
                "validation_contrastive": validation_mean["contrastive"],
            }
        )

        if args.print_every > 0 and (
            epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs
        ):
            print(
                f"Epoch {epoch:04d} | "
                f"train={train_mean['total']:.6f} | "
                f"val={validation_mean['total']:.6f} | "
                f"recon={validation_mean['reconstruction']:.6f} | "
                f"contrast={validation_mean['contrastive']:.6f}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "embedding_dim": embedding_dim,
            "encoder_dims": [512, embedding_dim],
            "decoder_dims": [embedding_dim, 512],
            "out_sig": False,
        },
        output_dir / "model_final.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "epoch_loss.csv", index=False)

    compound_embeddings = generate_compound_embeddings(
        model=model,
        compound_features=compound_features,
        batch_size=args.batch_size,
        device=device,
    )
    save_pickle(compound_embeddings, output_dir / "compound_embeddings.pkl")
    save_pickle(compound_names, output_dir / "compound_names.pkl")
    save_pickle(training_compound_names, output_dir / "training_compound_names.pkl")
    feature_mapping.to_csv(output_dir / "compound_feature_mapping.csv", index=False)

    selection_df = pd.DataFrame(
        {
            "compound": compound_names,
            "training_eligible": training_keep_mask.astype(bool),
            "positive_herb_count": np.sum(herb_labels_all == 1, axis=1),
            "negative_herb_count": np.sum(herb_labels_all == 0, axis=1),
        }
    )
    selection_df.to_csv(output_dir / "training_compound_selection.csv", index=False)

    config = vars(args).copy()
    config.update(
        {
            "resolved_device": str(device),
            "canonical_compound_count": len(canonical_compound_list),
            "mol2vec_supported_compound_count": len(compound_names),
            "training_compound_count": len(training_compound_names),
            "num_herbs": len(herb_list),
            "input_dim": input_dim,
            "embedding_dim": embedding_dim,
            "feature_metadata": feature_metadata,
            "compound_embedding_shape": list(compound_embeddings.shape),
            "training_label_counts": {
                "positive": int(np.sum(training_labels == 1)),
                "negative": int(np.sum(training_labels == 0)),
                "excluded": int(np.sum(training_labels == -1)),
            },
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config, fp, ensure_ascii=False, indent=2)

    print("\nSaved:")
    for name in [
        "model_final.pt",
        "epoch_loss.csv",
        "compound_embeddings.pkl",
        "compound_names.pkl",
        "training_compound_names.pkl",
        "compound_feature_mapping.csv",
        "training_compound_selection.csv",
        "config.json",
    ]:
        print(output_dir / name)


if __name__ == "__main__":
    main()
