#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train target-protein embeddings aligned to a TCM herb-embedding space using documented herb-target associations.
"""

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, TensorDataset

from model import Multimodal_AE, ContrastiveLoss


# ======================================================================================
# 1. Arguments and utilities
# ======================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train protein embeddings aligned to a TCM herb-embedding space."
    )

    parser.add_argument(
        "--embedding-dir",
        required=True,
        help="Directory containing individual_herb_embeddings.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the trained model, losses, and protein embeddings.",
    )

    # Shared project data.
    parser.add_argument(
        "--herb-embedding-file",
        default="individual_herb_embeddings.pkl",
    )
    parser.add_argument(
        "--herb-list",
        default="core/standard_TCM_entities/herb_list.pkl",
    )
    parser.add_argument(
        "--protein-list",
        default="data/herb_targets/protein_list.pkl",
    )
    parser.add_argument(
        "--herb-targets",
        default="data/herb_targets/herb_targets.pkl",
    )

    # Training settings.
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print one summary every N epochs.",
    )

    return parser.parse_args()


def load_pickle(path: Path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return torch.device(device_name)


# ======================================================================================
# 2. Data preparation
# ======================================================================================

def build_herb_labels(
    protein_list: List[str],
    herb_list: List[str],
    herb_targets: Dict[str, Iterable[str]],
) -> np.ndarray:
    """
    Build the protein × herb association matrix.

    Values
    ------
     1: associated herb
     0: no documented herb-target association
    -1: herb has no target annotation and is excluded from negative sampling
    """
    herb_target_sets = {
        herb: set(herb_targets[herb])
        for herb in herb_list
    }

    labels = np.empty(
        (len(protein_list), len(herb_list)),
        dtype=np.int8,
    )

    for herb_idx, herb in enumerate(herb_list):
        targets = herb_target_sets[herb]

        if not targets:
            labels[:, herb_idx] = -1
            continue

        labels[:, herb_idx] = np.fromiter(
            (1 if protein in targets else 0 for protein in protein_list),
            dtype=np.int8,
            count=len(protein_list),
        )

    positive_counts = np.sum(labels == 1, axis=1)
    negative_counts = np.sum(labels == 0, axis=1)

    if np.any(positive_counts == 0):
        bad = np.where(positive_counts == 0)[0][:10]
        raise ValueError(
            f"Proteins without a positive herb association: {bad.tolist()}"
        )

    if np.any(negative_counts == 0):
        bad = np.where(negative_counts == 0)[0][:10]
        raise ValueError(
            f"Proteins without a valid negative herb: {bad.tolist()}"
        )

    return labels


def prepare_dataset(
    protein_list: List[str],
    herb_labels: np.ndarray,
) -> TensorDataset:
    num_proteins = len(protein_list)

    # Protein identity is represented by a one-hot input.
    protein_inputs = torch.eye(
        num_proteins,
        dtype=torch.float32,
    )

    return TensorDataset(
        protein_inputs,
        torch.from_numpy(herb_labels),
    )


# ======================================================================================
# 3. Loss and training
# ======================================================================================

def sample_positive_negative_herbs(
    herb_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample one positive and one valid negative herb per protein."""
    positive_ids = []
    negative_ids = []

    for label_row in herb_labels:
        positive_candidates = torch.where(label_row == 1)[0]
        negative_candidates = torch.where(label_row == 0)[0]

        positive_ids.append(
            positive_candidates[
                torch.randint(
                    len(positive_candidates),
                    (1,),
                    device=label_row.device,
                )
            ]
        )
        negative_ids.append(
            negative_candidates[
                torch.randint(
                    len(negative_candidates),
                    (1,),
                    device=label_row.device,
                )
            ]
        )

    return torch.cat(positive_ids), torch.cat(negative_ids)


def compute_losses(
    reconstructed: torch.Tensor,
    protein_inputs: torch.Tensor,
    protein_embeddings: torch.Tensor,
    herb_labels: torch.Tensor,
    herb_embeddings: torch.Tensor,
    contrastive_loss: ContrastiveLoss,
    cross_entropy_loss: nn.CrossEntropyLoss,
    beta: float,
) -> Dict[str, torch.Tensor]:
    """Cross-entropy reconstruction plus herb–protein contrastive alignment."""
    protein_indices = torch.argmax(protein_inputs, dim=1)
    reconstruction = cross_entropy_loss(
        reconstructed,
        protein_indices,
    )

    positive_ids, negative_ids = sample_positive_negative_herbs(
        herb_labels
    )
    positive_herbs = herb_embeddings[positive_ids]
    negative_herbs = herb_embeddings[negative_ids]

    contrastive = torch.mean(
        contrastive_loss(
            protein_embeddings,
            positive_herbs,
            0,
        )
        + contrastive_loss(
            protein_embeddings,
            negative_herbs,
            1,
        )
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
    cross_entropy_loss: nn.CrossEntropyLoss,
    beta: float,
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "total": 0.0,
        "reconstruction": 0.0,
        "contrastive": 0.0,
        "top1_accuracy": 0.0,
    }
    sample_count = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for protein_inputs, herb_labels in loader:
            protein_inputs = protein_inputs.to(device)
            herb_labels = herb_labels.to(device)

            reconstructed, protein_embeddings = model(protein_inputs)

            losses = compute_losses(
                reconstructed=reconstructed,
                protein_inputs=protein_inputs,
                protein_embeddings=protein_embeddings,
                herb_labels=herb_labels,
                herb_embeddings=herb_embeddings,
                contrastive_loss=contrastive_loss,
                cross_entropy_loss=cross_entropy_loss,
                beta=beta,
            )

            if training:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

            batch_size = protein_inputs.shape[0]
            sample_count += batch_size

            for name in ("total", "reconstruction", "contrastive"):
                totals[name] += losses[name].item() * batch_size

            predicted = torch.argmax(reconstructed, dim=1)
            expected = torch.argmax(protein_inputs, dim=1)
            totals["top1_accuracy"] += (
                (predicted == expected).float().sum().item()
            )

    return {
        name: value / sample_count
        for name, value in totals.items()
    }


def average_fold_metrics(
    fold_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    return {
        key: float(np.mean([metrics[key] for metrics in fold_metrics]))
        for key in fold_metrics[0]
    }


# ======================================================================================
# 4. Protein-embedding export
# ======================================================================================

def generate_protein_embeddings(
    model: nn.Module,
    num_proteins: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    embedding_batches = []

    with torch.no_grad():
        for start in range(0, num_proteins, batch_size):
            stop = min(start + batch_size, num_proteins)
            protein_ids = torch.arange(start, stop)

            protein_inputs = F.one_hot(
                protein_ids,
                num_classes=num_proteins,
            ).float().to(device)

            _, embeddings = model(protein_inputs)
            embedding_batches.append(
                embeddings.cpu().numpy()
            )

    return np.concatenate(
        embedding_batches,
        axis=0,
    )


# ======================================================================================
# 5. Main workflow
# ======================================================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    herb_embeddings_np = np.asarray(
        load_pickle(
            embedding_dir / args.herb_embedding_file
        ),
        dtype=np.float32,
    )
    herb_list = list(
        load_pickle(Path(args.herb_list))
    )
    protein_list = list(
        load_pickle(Path(args.protein_list))
    )
    herb_targets = load_pickle(
        Path(args.herb_targets)
    )

    # Ensure herb labels and embedding rows use exactly the same order.
    if len(herb_targets) != len(herb_list):
        raise ValueError(
            f"herb_targets has {len(herb_targets)} herbs, "
            f"but herb_list has {len(herb_list)}."
        )

    if list(herb_targets.keys()) != herb_list:
        raise ValueError(
            "The order of herb_targets keys does not match herb_list."
        )

    if herb_embeddings_np.shape[0] != len(herb_list):
        raise ValueError(
            f"Herb embedding rows ({herb_embeddings_np.shape[0]}) "
            f"do not match herb_list ({len(herb_list)})."
        )

    num_proteins = len(protein_list)
    embedding_dim = herb_embeddings_np.shape[1]

    print(f"Device: {device}")
    print(f"Proteins: {num_proteins}")
    print(f"Herbs: {len(herb_list)}")
    print(f"Embedding dimension: {embedding_dim}")

    herb_labels = build_herb_labels(
        protein_list=protein_list,
        herb_list=herb_list,
        herb_targets=herb_targets,
    )
    dataset = prepare_dataset(
        protein_list=protein_list,
        herb_labels=herb_labels,
    )

    model = Multimodal_AE(
        input_dim=num_proteins,
        en_dims=[512, embedding_dim],
        de_dims=[embedding_dim, 512],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )
    contrastive_loss = ContrastiveLoss()
    cross_entropy_loss = nn.CrossEntropyLoss()

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
    fold_splits = list(
        kfold.split(np.arange(num_proteins))
    )

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

            train_metrics = run_loader(
                model=model,
                loader=train_loader,
                device=device,
                herb_embeddings=herb_embeddings,
                contrastive_loss=contrastive_loss,
                cross_entropy_loss=cross_entropy_loss,
                beta=args.beta,
                optimizer=optimizer,
            )
            validation_metrics = run_loader(
                model=model,
                loader=validation_loader,
                device=device,
                herb_embeddings=herb_embeddings,
                contrastive_loss=contrastive_loss,
                cross_entropy_loss=cross_entropy_loss,
                beta=args.beta,
                optimizer=None,
            )

            train_fold_metrics.append(train_metrics)
            validation_fold_metrics.append(validation_metrics)

        train_mean = average_fold_metrics(
            train_fold_metrics
        )
        validation_mean = average_fold_metrics(
            validation_fold_metrics
        )

        row = {"epoch": epoch}
        row.update({
            f"train_{key}": value
            for key, value in train_mean.items()
        })
        row.update({
            f"validation_{key}": value
            for key, value in validation_mean.items()
        })
        history.append(row)

        if (
            epoch == 1
            or epoch % args.print_every == 0
            or epoch == args.epochs
        ):
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train={train_mean['total']:.6f} | "
                f"val={validation_mean['total']:.6f} | "
                f"recon={validation_mean['reconstruction']:.6f} | "
                f"contrast={validation_mean['contrastive']:.6f} | "
                f"top1={validation_mean['top1_accuracy']:.4f}"
            )

    # Save final model.
    model_payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": num_proteins,
        "embedding_dim": embedding_dim,
        "encoder_dims": [512, embedding_dim],
        "decoder_dims": [embedding_dim, 512],
    }
    torch.save(
        model_payload,
        output_dir / "model_final.pt",
    )

    # Save fold-averaged training history.
    pd.DataFrame(history).to_csv(
        output_dir / "epoch_loss.csv",
        index=False,
    )

    # Generate and save protein embeddings in protein_list order.
    protein_embeddings = generate_protein_embeddings(
        model=model,
        num_proteins=num_proteins,
        batch_size=args.batch_size,
        device=device,
    )
    save_pickle(
        protein_embeddings,
        output_dir / "target_embeddings.pkl",
    )

    config = vars(args).copy()
    config.update({
        "resolved_device": str(device),
        "num_proteins": num_proteins,
        "num_herbs": len(herb_list),
        "embedding_dim": embedding_dim,
        "protein_embedding_shape": list(
            protein_embeddings.shape
        ),
    })

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            config,
            fp,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved:")
    print(output_dir / "model_final.pt")
    print(output_dir / "epoch_loss.csv")
    print(output_dir / "target_embeddings.pkl")
    print(output_dir / "config.json")


if __name__ == "__main__":
    main()
