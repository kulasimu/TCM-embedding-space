#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare Mol2Vec structural vectors for retained herbal compounds.

Compound order is defined by ``compound_list.pkl``. SMILES are matched from the
compound table by ``Common Name``; when a name occurs more than once, the first
row is used to preserve the historical matching behavior. Mol2Vec vectors are
generated with DeepChem ``Mol2VecFingerprint`` and saved once for downstream
compound-alignment training.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


FEATURE_METHOD = "DeepChem Mol2VecFingerprint"


class Mol2VecResult:
    def __init__(self, features: np.ndarray, kept_indices: list[int], failures: pd.DataFrame):
        self.features = features
        self.kept_indices = kept_indices
        self.failures = failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Mol2Vec structural vectors for herbal compounds."
    )
    parser.add_argument(
        "--compound-list",
        default="data/herb_compounds/compound_list.pkl",
        help="Ordered list of retained herbal compounds.",
    )
    parser.add_argument(
        "--compound-table",
        default="data/herb_compounds/compound_SMILES.xlsx",
        help="Excel table containing compound names and SMILES.",
    )
    parser.add_argument(
        "--sheet-name",
        default="filtered",
        help="Excel sheet name or zero-based sheet index.",
    )
    parser.add_argument("--name-column", default="Common Name")
    parser.add_argument("--smiles-column", default="Smiles")
    parser.add_argument(
        "--output-file",
        default="data/herb_compounds/compound_mol2vec_vectors.pkl",
        help="Output pickle containing Mol2Vec vectors and row metadata.",
    )
    parser.add_argument(
        "--mapping-csv",
        default="data/herb_compounds/compound_mol2vec_mapping.csv",
        help="Matched compound-to-SMILES rows.",
    )
    parser.add_argument(
        "--failure-csv",
        default="data/herb_compounds/compound_mol2vec_failures.csv",
        help="Compounds for which Mol2Vec did not yield a usable vector.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Progress interval during Mol2Vec generation.",
    )
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


def resolve_sheet_name(value: str):
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def match_compounds_to_smiles(
    compound_table: pd.DataFrame,
    compound_list: Sequence[str],
    name_column: str,
    smiles_column: str,
) -> pd.DataFrame:
    """Match SMILES in canonical compound-list order using the first duplicate."""
    if name_column not in compound_table.columns:
        raise KeyError(f"Missing compound-name column: {name_column}")
    if smiles_column not in compound_table.columns:
        raise KeyError(f"Missing SMILES column: {smiles_column}")

    positions: dict[object, list[int]] = {}
    for row_pos, name in enumerate(compound_table[name_column].tolist()):
        positions.setdefault(name, []).append(row_pos)

    rows = []
    missing = []
    missing_smiles = []
    for compound in compound_list:
        matches = positions.get(compound, [])
        if not matches:
            missing.append(compound)
            continue

        source_row = matches[0]
        smiles = compound_table.iloc[source_row][smiles_column]
        if not isinstance(smiles, str) or not smiles.strip():
            missing_smiles.append(compound)
            continue

        rows.append(
            {
                "compound": compound,
                "smiles": smiles.strip(),
                "source_row": int(source_row),
                "duplicate_name_count": int(len(matches)),
            }
        )

    if missing:
        raise ValueError(
            f"{len(missing)} compounds were not found in the compound table. "
            f"Examples: {missing[:10]}"
        )
    if missing_smiles:
        raise ValueError(
            f"{len(missing_smiles)} compounds have missing/empty SMILES. "
            f"Examples: {missing_smiles[:10]}"
        )

    matched = pd.DataFrame(rows)
    if len(matched) != len(compound_list):
        raise RuntimeError("Compound-to-SMILES matching did not preserve all compounds.")
    return matched


def create_mol2vec_featurizer():
    """Create DeepChem Mol2Vec lazily so --help works without DeepChem installed."""
    try:
        import deepchem as dc
    except ImportError as exc:
        raise ImportError(
            "DeepChem is required for Mol2Vec preprocessing. Install the "
            "DeepChem environment used for compound preprocessing."
        ) from exc
    return dc.feat.Mol2VecFingerprint()


def _coerce_mol2vec_vector(raw) -> tuple[Optional[np.ndarray], Optional[str]]:
    array = np.asarray(raw)
    if array.size == 0:
        return None, "empty_vector"

    try:
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None, "invalid_vector"

    if vector.size == 0:
        return None, "empty_vector"
    if not np.isfinite(vector).all():
        return None, "non_finite_vector"
    return vector, None


def generate_mol2vec_features(
    compound_names: Sequence[str],
    smiles_list: Sequence[str],
    featurizer=None,
    print_every: int = 0,
) -> Mol2VecResult:
    """Generate Mol2Vec vectors, retaining compounds with usable vectors."""
    if len(compound_names) != len(smiles_list):
        raise ValueError("compound_names and smiles_list must have the same length.")
    if featurizer is None:
        featurizer = create_mol2vec_featurizer()

    vectors = []
    kept_indices: list[int] = []
    failures = []
    detected_dim: Optional[int] = None

    for i, (compound, smiles) in enumerate(zip(compound_names, smiles_list)):
        try:
            raw = featurizer.featurize(smiles)
        except Exception as exc:
            failures.append(
                {
                    "compound": compound,
                    "smiles": smiles,
                    "reason": "featurization_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        vector, reason = _coerce_mol2vec_vector(raw)
        if vector is None:
            failures.append(
                {
                    "compound": compound,
                    "smiles": smiles,
                    "reason": reason,
                    "detail": "",
                }
            )
            continue

        if detected_dim is None:
            detected_dim = int(vector.size)
        elif vector.size != detected_dim:
            raise ValueError(
                f"Inconsistent Mol2Vec dimension for {compound!r}: "
                f"{vector.size} vs detected {detected_dim}."
            )

        vectors.append(vector)
        kept_indices.append(i)

        if print_every > 0 and (
            (i + 1) % print_every == 0 or i + 1 == len(smiles_list)
        ):
            print(f"Mol2Vec: {i + 1}/{len(smiles_list)}", flush=True)

    if not vectors:
        raise ValueError("Mol2Vec did not produce any usable compound vectors.")

    failure_columns = ["compound", "smiles", "reason", "detail"]
    failure_df = pd.DataFrame(failures, columns=failure_columns)
    return Mol2VecResult(
        features=np.stack(vectors, axis=0).astype(np.float32, copy=False),
        kept_indices=kept_indices,
        failures=failure_df,
    )


def build_feature_payload(
    mapping: pd.DataFrame,
    result: Mol2VecResult,
) -> dict:
    """Build an aligned feature payload for downstream compound training."""
    features = np.asarray(result.features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("Mol2Vec feature matrix must be 2-D.")
    if features.shape[0] != len(result.kept_indices):
        raise ValueError("Mol2Vec features and kept indices are not aligned.")

    kept = mapping.iloc[result.kept_indices].reset_index(drop=True)
    if len(kept) != features.shape[0]:
        raise ValueError("Matched compound metadata and Mol2Vec vectors are not aligned.")

    dropped = result.failures["compound"].tolist() if len(result.failures) else []
    return {
        "feature_method": FEATURE_METHOD,
        "feature_dim": int(features.shape[1]),
        "compound_names": kept["compound"].tolist(),
        "smiles": kept["smiles"].tolist(),
        "features": features,
        "source_rows": kept["source_row"].astype(int).tolist(),
        "duplicate_name_counts": kept["duplicate_name_count"].astype(int).tolist(),
        "canonical_compound_count": int(len(mapping)),
        "supported_compound_count": int(len(kept)),
        "dropped_compounds": dropped,
    }


def main() -> None:
    args = parse_args()

    compound_list = list(load_pickle(Path(args.compound_list)))
    if len(compound_list) != len(set(compound_list)):
        raise ValueError("compound_list contains duplicate names.")

    compound_table = pd.read_excel(
        args.compound_table,
        sheet_name=resolve_sheet_name(args.sheet_name),
    )
    mapping = match_compounds_to_smiles(
        compound_table=compound_table,
        compound_list=compound_list,
        name_column=args.name_column,
        smiles_column=args.smiles_column,
    )

    result = generate_mol2vec_features(
        compound_names=mapping["compound"].tolist(),
        smiles_list=mapping["smiles"].tolist(),
        print_every=args.print_every,
    )
    feature_dim = int(result.features.shape[1])

    print(f"Canonical compounds: {len(compound_list)}")
    print(f"Mol2Vec-supported compounds: {result.features.shape[0]}")
    print(f"Detected Mol2Vec feature dimension: {feature_dim}")
    payload = build_feature_payload(mapping, result)
    payload.update(
        {
            "compound_table": str(args.compound_table),
            "sheet_name": args.sheet_name,
            "name_column": args.name_column,
            "smiles_column": args.smiles_column,
        }
    )

    output_file = Path(args.output_file)
    mapping_csv = Path(args.mapping_csv)
    failure_csv = Path(args.failure_csv)

    save_pickle(payload, output_file)
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_csv, index=False)
    failure_csv.parent.mkdir(parents=True, exist_ok=True)
    result.failures.to_csv(failure_csv, index=False)

    duplicate_count = int((mapping["duplicate_name_count"] > 1).sum())
    print(f"Compound names with duplicate source rows: {duplicate_count}")
    print(f"Mol2Vec failures: {len(result.failures)}")
    print(f"Saved Mol2Vec features: {output_file}")
    print(f"Saved compound mapping: {mapping_csv}")
    print(f"Saved Mol2Vec failure table: {failure_csv}")


if __name__ == "__main__":
    main()
