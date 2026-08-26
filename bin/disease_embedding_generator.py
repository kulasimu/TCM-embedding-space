#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate disease embeddings in the fixed TCM-ES using disease-associated symptoms."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from TCM_embedding_generator import TCMEmbeddingGenerator


DEFAULT_MAIN_MODEL = Path("core/trained_model/model_epoch_60.pkl")
DEFAULT_REPEAT_MODEL_ROOT = Path("core/trained_model_repeated")
DEFAULT_MODEL_PATTERN = "model_epoch_*.pkl"
DEFAULT_OUTPUT_ROOT = Path("results/embeddings/disease_embeddings")

DEFAULT_DISEASE_TABLE = Path("data/disease/Disease_class_gene(science2015).xlsx")
DEFAULT_DISEASE_MMSYM = Path("data/disease/disease_mmsym.pkl")
DEFAULT_MM_SYMPTOM_LIST = Path("data/disease/mm_symptom_list.pkl")
DEFAULT_MM_SYMPTOM_SEMANTICS = Path("data/disease/mm_symptom_list_semantics.pkl")

DEFAULT_SYMPTOM_LIST = Path("core/standard_TCM_entities/symptom_list.pkl")
DEFAULT_SYMPTOM_SEMANTICS = Path("core/standard_TCM_entities/symptom_semantic_encodings.pkl")
DEFAULT_HERB_LIST = Path("core/standard_TCM_entities/herb_list.pkl")

MAX_DISEASE_SYMPTOMS = 40


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def normalize_symptom_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [str(item) for item in value if pd.notna(item)]
    if pd.isna(value):
        return []
    return [str(value)]


def build_disease_semantic_inputs(
    disease_table: Path,
    disease_mmsym: Path,
    mm_symptom_list: Path,
    mm_symptom_semantics: Path,
    max_symptoms: int = MAX_DISEASE_SYMPTOMS,
) -> Tuple[List[str], List[List[str]], np.ndarray, pd.DataFrame]:
    """Construct row-aligned semantic inputs for diseases in table order."""
    disease_df = pd.read_excel(disease_table)
    disease_df.columns = disease_df.columns.astype(str).str.strip()
    if "Disease" not in disease_df.columns:
        raise ValueError(f"Disease column not found in: {disease_table}")

    disease_names = disease_df["Disease"].astype(str).tolist()
    disease_to_symptoms = load_pickle(disease_mmsym)
    if not isinstance(disease_to_symptoms, dict):
        raise TypeError("disease_mmsym.pkl must contain a disease-to-symptom mapping.")

    symptom_names = list(load_pickle(mm_symptom_list))
    symptom_semantics = np.asarray(load_pickle(mm_symptom_semantics), dtype=np.float32)

    if symptom_semantics.ndim != 2:
        raise ValueError("Modern-medicine symptom semantics must be a 2-D array.")

    expected_rows = len(symptom_names) + 3
    if symptom_semantics.shape[0] != expected_rows:
        raise ValueError(
            "Semantic rows do not match the modern-medicine symptom vocabulary: "
            f"{symptom_semantics.shape[0]} vs {len(symptom_names)} + 3."
        )

    missing_diseases = [name for name in disease_names if name not in disease_to_symptoms]
    if missing_diseases:
        preview = ", ".join(missing_diseases[:10])
        raise ValueError(f"Diseases missing from disease_mmsym: {preview}")

    symptom_to_id = {str(symptom): idx for idx, symptom in enumerate(symptom_names)}
    disease_symptoms = [
        normalize_symptom_list(disease_to_symptoms[name])
        for name in disease_names
    ]

    missing_symptoms = sorted({
        symptom
        for symptoms in disease_symptoms
        for symptom in symptoms
        if symptom not in symptom_to_id
    })
    if missing_symptoms:
        preview = ", ".join(missing_symptoms[:20])
        raise ValueError(
            f"{len(missing_symptoms)} disease-associated symptoms are absent from "
            f"mm_symptom_list, for example: {preview}"
        )

    pad_id = len(symptom_names)
    start_id = len(symptom_names) + 1
    sequence_length = max_symptoms + 1

    semantic_inputs = []
    summary_rows = []

    for disease, symptoms in zip(disease_names, disease_symptoms):
        symptom_ids = [symptom_to_id[symptom] for symptom in symptoms]
        token_ids = [start_id] + symptom_ids[:max_symptoms]
        token_ids += [pad_id] * (sequence_length - len(token_ids))

        semantic_inputs.append(symptom_semantics[token_ids])
        summary_rows.append({
            "disease": disease,
            "raw_symptom_count": len(symptoms),
            "encoded_symptom_count": min(len(symptoms), max_symptoms),
            "nonempty_mmsym": len(symptoms) > 0,
            "truncated": len(symptoms) > max_symptoms,
        })

    semantic_inputs_array = np.asarray(semantic_inputs, dtype=np.float32)
    expected_shape = (
        len(disease_names),
        sequence_length,
        symptom_semantics.shape[1],
    )
    if semantic_inputs_array.shape != expected_shape:
        raise ValueError(
            f"Expected disease semantic input shape {expected_shape}, "
            f"found {semantic_inputs_array.shape}."
        )

    return (
        disease_names,
        disease_symptoms,
        semantic_inputs_array,
        pd.DataFrame(summary_rows),
    )


def generate_disease_embedding_from_semantics(
    generator: TCMEmbeddingGenerator,
    symptom_semantic_input: np.ndarray,
    has_symptoms: bool,
) -> np.ndarray:
    """Generate one disease embedding from a prepared symptom-semantic sequence."""
    symptom_semantic_input = np.asarray(symptom_semantic_input, dtype=np.float32)
    if symptom_semantic_input.shape[0] != generator.MAX_SYM_LEN:
        raise ValueError(
            "Disease semantic input must have "
            f"{generator.MAX_SYM_LEN} sequence positions, "
            f"found {symptom_semantic_input.shape}."
        )

    symptom_tensor = torch.FloatTensor(
        symptom_semantic_input[None, :, :]
    ).to(generator.device)

    herb_start_id = len(generator.herb_list) + 1
    herb_end_id = len(generator.herb_list) + 2
    herb_input_ids = [herb_start_id]
    disease_embedding = None

    n_steps = generator.MAX_HERB_LEN if has_symptoms else 1

    with torch.no_grad():
        for _ in range(n_steps):
            herb_tensor = torch.IntTensor([herb_input_ids]).to(generator.device)

            _, herb_prediction, z1, _, _ = generator.model(
                symptom_tensor,
                herb_tensor,
                mask_input="herb",
            )

            disease_embedding = (
                z1[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if not has_symptoms:
                break

            prediction_scores = (
                herb_prediction[0, -1]
                .detach()
                .cpu()
                .numpy()
            )

            next_herb_id = None
            for candidate in np.argsort(-prediction_scores):
                candidate = int(candidate)
                if candidate not in herb_input_ids:
                    next_herb_id = candidate
                    break

            if next_herb_id is None:
                break

            herb_input_ids.append(next_herb_id)
            if next_herb_id == herb_end_id:
                break

    if disease_embedding is None:
        raise RuntimeError("Disease embedding generation failed.")

    return disease_embedding


def discover_repeat_models(
    repeat_model_root: Path,
    model_pattern: str,
) -> List[Tuple[str, Path]]:
    """Find one checkpoint under each repeat_* directory."""
    repeat_dirs = sorted(
        [path for path in repeat_model_root.glob("repeat_*") if path.is_dir()],
        key=lambda path: path.name,
    )
    if not repeat_dirs:
        raise FileNotFoundError(f"No repeat_* folders found under: {repeat_model_root}")

    models: List[Tuple[str, Path]] = []
    for repeat_dir in repeat_dirs:
        model_files = sorted(repeat_dir.glob(model_pattern))
        if len(model_files) != 1:
            raise ValueError(
                f"Expected exactly one model checkpoint in {repeat_dir} matching "
                f"{model_pattern!r}, found {len(model_files)}: "
                f"{[path.name for path in model_files]}"
            )
        models.append((repeat_dir.name, model_files[0]))
    return models


def build_model_jobs(
    mode: str,
    main_model: Path,
    repeat_model_root: Path,
    model_pattern: str,
    output_root: Path,
) -> List[Tuple[str, Path, Path]]:
    """Resolve model checkpoints and output directories for one run mode."""
    if mode == "main":
        if not main_model.exists():
            raise FileNotFoundError(f"Main model not found: {main_model}")
        return [("main", main_model, output_root / "main")]

    if mode == "repeat":
        return [
            (name, model_path, output_root / name)
            for name, model_path in discover_repeat_models(
                repeat_model_root,
                model_pattern,
            )
        ]

    raise ValueError("mode must be 'main' or 'repeat'.")


def generate_embeddings_for_model(
    model_name: str,
    model_path: Path,
    output_dir: Path,
    disease_names: Sequence[str],
    disease_symptoms: Sequence[Sequence[str]],
    disease_semantic_inputs: np.ndarray,
    input_summary: pd.DataFrame,
    device: torch.device,
    symptom_list: Path,
    symptom_semantics: Path,
    herb_list: Path,
    overwrite: bool,
) -> None:
    """Generate and save row-aligned disease embeddings for one TCM-ES model."""
    output_file = output_dir / "disease_embedding.pkl"
    if output_file.exists() and not overwrite:
        print(f"[{model_name}] disease embeddings already exist; skipped: {output_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{model_name}] model: {model_path}")

    generator = TCMEmbeddingGenerator(
        model_dir=str(model_path),
        device=device,
        symptom_list_dir=str(symptom_list),
        symptom_semantic_dir=str(symptom_semantics),
        herb_list_dir=str(herb_list),
    )

    embedding_rows = []
    n_diseases = len(disease_names)

    for index, (disease, symptoms, semantic_input) in enumerate(
        zip(disease_names, disease_symptoms, disease_semantic_inputs),
        start=1,
    ):
        if index == 1 or index % 20 == 0 or index == n_diseases:
            print(f"[{model_name}] disease {index}/{n_diseases}: {disease}", flush=True)

        embedding_rows.append(
            generate_disease_embedding_from_semantics(
                generator,
                semantic_input,
                has_symptoms=len(symptoms) > 0,
            )
        )

    disease_embeddings = np.asarray(embedding_rows, dtype=np.float32)
    expected_shape = (len(disease_names), generator.EMBEDDING_DIM)
    if disease_embeddings.shape != expected_shape:
        raise ValueError(
            f"[{model_name}] expected embedding shape {expected_shape}, "
            f"found {disease_embeddings.shape}."
        )
    if not np.isfinite(disease_embeddings).all():
        raise ValueError(f"[{model_name}] disease embeddings contain NaN or infinity.")

    nonempty_ids = np.asarray(
        [index for index, symptoms in enumerate(disease_symptoms) if len(symptoms) > 0],
        dtype=int,
    )
    if len(nonempty_ids) > 1:
        unique_nonempty = np.unique(disease_embeddings[nonempty_ids], axis=0).shape[0]
        if unique_nonempty <= 1:
            raise RuntimeError(
                f"[{model_name}] all non-empty diseases produced the same embedding. "
                "Check disease semantic inputs and model inference."
            )

    save_pickle(disease_embeddings, output_file)
    save_pickle(list(disease_names), output_dir / "disease_names.pkl")
    input_summary.to_csv(
        output_dir / "disease_embedding_input_summary.csv",
        index=False,
        encoding="utf_8_sig",
    )

    print(f"[{model_name}] saved: {disease_embeddings.shape}")
    print(f"[{model_name}] output: {output_file}")

    del generator
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate disease embeddings from disease-associated symptom semantic "
            "representations using the main or repeated TCM-ES models."
        )
    )
    parser.add_argument("--mode", choices=("main", "repeat"), required=True)

    parser.add_argument("--main-model", type=Path, default=DEFAULT_MAIN_MODEL)
    parser.add_argument(
        "--repeat-model-root",
        type=Path,
        default=DEFAULT_REPEAT_MODEL_ROOT,
    )
    parser.add_argument("--model-pattern", default=DEFAULT_MODEL_PATTERN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--disease-table", type=Path, default=DEFAULT_DISEASE_TABLE)
    parser.add_argument("--disease-mmsym", type=Path, default=DEFAULT_DISEASE_MMSYM)
    parser.add_argument("--mm-symptom-list", type=Path, default=DEFAULT_MM_SYMPTOM_LIST)
    parser.add_argument(
        "--mm-symptom-semantics",
        type=Path,
        default=DEFAULT_MM_SYMPTOM_SEMANTICS,
    )

    parser.add_argument("--symptom-list", type=Path, default=DEFAULT_SYMPTOM_LIST)
    parser.add_argument(
        "--symptom-semantics",
        type=Path,
        default=DEFAULT_SYMPTOM_SEMANTICS,
    )
    parser.add_argument("--herb-list", type=Path, default=DEFAULT_HERB_LIST)

    parser.add_argument("--max-symptoms", type=int, default=MAX_DISEASE_SYMPTOMS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_symptoms < 1:
        raise ValueError("--max-symptoms must be at least 1.")

    required_inputs = [
        args.disease_table,
        args.disease_mmsym,
        args.mm_symptom_list,
        args.mm_symptom_semantics,
        args.symptom_list,
        args.symptom_semantics,
        args.herb_list,
    ]
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n" + "\n".join(missing))

    disease_names, disease_symptoms, disease_semantic_inputs, input_summary = (
        build_disease_semantic_inputs(
            disease_table=args.disease_table,
            disease_mmsym=args.disease_mmsym,
            mm_symptom_list=args.mm_symptom_list,
            mm_symptom_semantics=args.mm_symptom_semantics,
            max_symptoms=args.max_symptoms,
        )
    )

    jobs = build_model_jobs(
        mode=args.mode,
        main_model=args.main_model,
        repeat_model_root=args.repeat_model_root,
        model_pattern=args.model_pattern,
        output_root=args.output_root,
    )

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Diseases: {len(disease_names)}")
    print(f"Disease semantic input shape: {disease_semantic_inputs.shape}")
    print(f"Models to process: {len(jobs)}")

    for model_name, model_path, output_dir in jobs:
        generate_embeddings_for_model(
            model_name=model_name,
            model_path=model_path,
            output_dir=output_dir,
            disease_names=disease_names,
            disease_symptoms=disease_symptoms,
            disease_semantic_inputs=disease_semantic_inputs,
            input_summary=input_summary,
            device=device,
            symptom_list=args.symptom_list,
            symptom_semantics=args.symptom_semantics,
            herb_list=args.herb_list,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
