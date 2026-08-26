#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate TCM-ES embeddings for the COVID-19 clinical cohort.

This script is intentionally parallel to bin/TCM_embedding_generator.py, but
uses the COVID-19 symptom vocabulary and symptom semantic encodings.

Input data are fixed under:
    data/COVID_19_data/
        COVID_data.pkl
        COVID_formula_composition.pkl
        COVID_symptom_list.pkl
        COVID_symptom_semantics.pkl

Output embeddings are saved under:
    results/embeddings/COVID_19_cases/original/

The saved files are aligned row-by-row with:
    data/COVID_19_data/COVID_19_data_for_embeddings.pkl
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class COVIDEmbeddingGenerator:
    def __init__(
        self,
        model_dir="core/trained_model/model_epoch_60.pkl",
        save_dir="results/embeddings/COVID_19_cases/original",
        device="cpu",
        herb_list_dir="core/standard_TCM_entities/herb_list.pkl",
        covid_symptom_list_dir="data/COVID_19_data/COVID_symptom_list.pkl",
        covid_symptom_semantic_dir="data/COVID_19_data/COVID_symptom_semantics.pkl",
    ):
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.device = device
        self.herb_list_dir = herb_list_dir
        self.covid_symptom_list_dir = covid_symptom_list_dir
        self.covid_symptom_semantic_dir = covid_symptom_semantic_dir

        self.n_herb_seq = 30
        self.n_sym_seq = 40
        self.MAX_HERB_LEN = self.n_herb_seq + 1
        self.MAX_SYM_LEN = self.n_sym_seq + 1
        self.EMBEDDING_DIM = 256

        os.makedirs(self.save_dir, exist_ok=True)
        self._load_trained_model()
        self._load_terms()

    def _load_pickle(self, path):
        with open(path, "rb") as fp:
            return pickle.load(fp)

    def _load_terms(self):
        self.herb_list = self._load_pickle(self.herb_list_dir)
        self.covid_symptom_list = self._load_pickle(self.covid_symptom_list_dir)
        self.covid_symptom_semantics = np.asarray(self._load_pickle(self.covid_symptom_semantic_dir), dtype=np.float32,)

        if self.covid_symptom_semantics.shape[0] != len(self.covid_symptom_list) + 3:
            raise ValueError(
                "COVID_symptom_semantics must contain COVID symptoms + <pad> + <S> + <E>. "
                f"Got {self.covid_symptom_semantics.shape[0]} rows for "
                f"{len(self.covid_symptom_list)} COVID symptoms."
            )

        self.herb2id = {h: i for i, h in enumerate(self.herb_list)}
        self.covid_symptom2id = {s: i for i, s in enumerate(self.covid_symptom_list)}

        self.covid_pad_sym_idx = len(self.covid_symptom_list)
        self.covid_start_sym_idx = len(self.covid_symptom_list) + 1
        self.covid_end_sym_idx = len(self.covid_symptom_list) + 2

    def _load_trained_model(self):
        self.model = torch.load(
            self.model_dir,
            map_location=self.device,
            weights_only=False,
        )
        self.model.to(self.device)
        self.model.eval()

    def _generate_herb_data(self, herb_id_seq, inOrOut="input"):
        if len(herb_id_seq) > self.n_herb_seq:
            herb_id_seq = herb_id_seq[:self.n_herb_seq]

        seq = (
            [len(self.herb_list) + 1] + herb_id_seq
            if inOrOut == "input"
            else herb_id_seq + [len(self.herb_list) + 2]
        )
        return seq + [len(self.herb_list)] * (self.MAX_HERB_LEN - len(seq))

    def _generate_empty_covid_sym_data(self):
        """
        Generate a dummy COVID symptom input for herb-only embedding.

        The model requires both symptom and herb inputs. When generating herb/formula
        embeddings, the symptom branch receives only COVID <S> and <pad> tokens.
        """
        seq = [self.covid_start_sym_idx]
        seq = seq + [self.covid_pad_sym_idx] * (self.MAX_SYM_LEN - len(seq))
        return [self.covid_symptom_semantics[s] for s in seq]

    def _generate_covid_sym_data_from_semantics(self, sym_semantic_seq):
        if len(sym_semantic_seq) > self.n_sym_seq:
            sym_semantic_seq = sym_semantic_seq[:self.n_sym_seq]

        semantics = [self.covid_symptom_semantics[self.covid_start_sym_idx]]
        semantics += list(sym_semantic_seq)
        semantics += [self.covid_symptom_semantics[self.covid_pad_sym_idx]] * (
            self.MAX_SYM_LEN - len(semantics)
        )
        return semantics

    def covid_symptoms_to_semantics(self, symptoms):
        return [
            self.covid_symptom_semantics[self.covid_symptom2id[s]]
            for s in symptoms
        ]

    def generate_covid_symptom_embedding(self, case_symptoms):
        herb_inputs, symptom_inputs = [], []

        for symptoms in case_symptoms:
            symptom_semantic_seq = self.covid_symptoms_to_semantics(symptoms)
            symptom_inputs.append(
                self._generate_covid_sym_data_from_semantics(symptom_semantic_seq)
            )
            herb_inputs.append(self._generate_herb_data([], "input"))

        herb_tensor = torch.IntTensor(np.asarray(herb_inputs)).to(self.device)
        sym_tensor = torch.FloatTensor(np.asarray(symptom_inputs)).to(self.device)

        with torch.no_grad():
            _, _, z1, _, _ = self.model(sym_tensor, herb_tensor)

        return z1.cpu().numpy()

    def generate_herb_embedding(self, case_herbs):
        herb_inputs, symptom_inputs = [], []

        for herb_seq in case_herbs:
            herb_id_seq = [self.herb2id[h] for h in herb_seq if h in self.herb2id]
            herb_inputs.append(self._generate_herb_data(herb_id_seq, "input"))
            symptom_inputs.append(self._generate_empty_covid_sym_data())

        herb_tensor = torch.IntTensor(np.asarray(herb_inputs)).to(self.device)
        sym_tensor = torch.FloatTensor(np.asarray(symptom_inputs)).to(self.device)

        with torch.no_grad():
            _, _, _, z2, _ = self.model(sym_tensor, herb_tensor)

        return z2.cpu().numpy()

    def save_embeddings(self, embeddings_dict):
        for key, value in embeddings_dict.items():
            with open(os.path.join(self.save_dir, f"{key}.pkl"), "wb") as fp:
                pickle.dump(value, fp)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def unique_list(x):
    out = []
    seen = set()
    for item in x:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def to_float_score(x):
    if pd.isna(x):
        return 0.0
    return float(x)


def build_covid_clinical_dataframe(
    covid_df,
    covid_formula_composition,
    covid_symptom_list,
    generator,
):
    rows = []

    case_initial_symptoms_for_embedding = []
    case_improved_symptoms_for_embedding = []
    case_unimproved_symptoms_for_embedding = []

    def sort_and_trim(symptoms):
        symptoms = sorted(
            symptoms,
            key=lambda s: covid_symptom_list.index(s),
        )
        return symptoms[:generator.n_sym_seq]

    for original_row_idx, row in covid_df.iterrows():
        case_id = row["患者編號"]
        formula_name = row["處方(processed)"]

        if formula_name not in covid_formula_composition:
            raise KeyError(
                f"Formula {formula_name!r} is not found in COVID_formula_composition. "
                f"Original row index: {original_row_idx}"
            )

        formula_herbs = unique_list(covid_formula_composition[formula_name])

        initial_symptoms = []
        improved_symptoms = []
        unimproved_symptoms = []
        initial_score_vector = []
        followup_score_vector = []

        for symptom in covid_symptom_list:
            init_score = to_float_score(row[f"{symptom}-initial"])
            follow_score = to_float_score(row[f"{symptom}-followup"])

            initial_score_vector.append(init_score)
            followup_score_vector.append(follow_score)

            if init_score > 0:
                initial_symptoms.append(symptom)

                if follow_score < init_score:
                    improved_symptoms.append(symptom)
                else:
                    unimproved_symptoms.append(symptom)

        initial_symptoms_for_embedding = sort_and_trim(initial_symptoms)
        improved_symptoms_for_embedding = sort_and_trim(improved_symptoms)
        unimproved_symptoms_for_embedding = sort_and_trim(unimproved_symptoms)

        row_out = {
            "Original_row_idx": int(original_row_idx),
            "Cases_id": case_id,
            "Initial_symptoms": initial_symptoms_for_embedding,
            "Formula": formula_herbs,
            "Alleviated_symptoms": improved_symptoms_for_embedding,
            "Unalleviated_symptoms": unimproved_symptoms_for_embedding,
            "Formula_name": formula_name,
            "n_initial_symptoms": len(initial_symptoms),
            "n_alleviated_symptoms": len(improved_symptoms),
            "n_unimproved_symptoms": len(unimproved_symptoms),

            # Fixed-order 60-dimensional symptom score vectors.
            # The order is exactly the same as COVID_symptom_list.
            "Initial_score_vector": initial_score_vector,
            "Followup_score_vector": followup_score_vector,
        }

        # Optional metadata pass-through. This makes downstream row-based
        # filtering easier if these columns exist in COVID_data.pkl.
        for optional_col in [
            "初诊至复诊天数",
            "初診至復診天數",
            "初診至复診天數",
            "初诊至復诊天数",
            "case_intervalDays",
            "intervalDays",
            "感染至初诊天数",
            "感染至初診天數",
        ]:
            if optional_col in covid_df.columns:
                row_out[optional_col] = row[optional_col]

        rows.append(row_out)

        case_initial_symptoms_for_embedding.append(initial_symptoms_for_embedding)
        case_improved_symptoms_for_embedding.append(improved_symptoms_for_embedding)
        case_unimproved_symptoms_for_embedding.append(unimproved_symptoms_for_embedding)

    clinical_df = pd.DataFrame(rows).reset_index(drop=True)

    if len(clinical_df) != len(covid_df):
        raise ValueError(
            f"Unexpected row mismatch: clinical_df has {len(clinical_df)} rows, "
            f"but raw covid_df has {len(covid_df)} rows."
        )

    return (
        clinical_df,
        case_initial_symptoms_for_embedding,
        case_improved_symptoms_for_embedding,
        case_unimproved_symptoms_for_embedding,
    )


def generate_embeddings_in_batches(generator, inputs, mode, batch_size=1000):
    embeddings = np.zeros((0, generator.EMBEDDING_DIM), dtype=np.float32)

    for start in range(0, len(inputs), batch_size):
        end = min(start + batch_size, len(inputs))
        print(mode, start, end)

        batch = inputs[start:end]
        if mode == "covid_symptom":
            emb = generator.generate_covid_symptom_embedding(batch)
        elif mode == "herb":
            emb = generator.generate_herb_embedding(batch)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        embeddings = np.concatenate((embeddings, emb.astype(np.float32)), axis=0)

    return embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TCM-ES embeddings for COVID-19 clinical cases."
    )
    parser.add_argument(
        "--data-dir",
        default="data/COVID_19_data",
        help="Directory containing COVID_data.pkl and COVID metadata files.",
    )

    parser.add_argument(
        "--covid-data-file",
        default="COVID_data.pkl",
        help="COVID-19 case dataframe filename inside --data-dir.",
    )

    parser.add_argument(
        "--model-dir",
        default="core/trained_model/model_epoch_60.pkl",
        help="Trained TCM-ES model checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/embeddings/COVID_19_cases/original",
        help="Output embedding directory.",
    )
    parser.add_argument(
        "--clinical-out",
        default="data/COVID_19_data/COVID_19_data_for_embeddings.pkl",
        help="Output clinical dataframe aligned with generated embeddings.",
    )



    parser.add_argument("--batch-size", type=int, default=1000)

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.clinical_out))

    covid_data_path = os.path.join(args.data_dir, args.covid_data_file)
    covid_formula_path = os.path.join(args.data_dir, "COVID_formula_composition.pkl")
    covid_symptom_list_path = os.path.join(args.data_dir, "COVID_symptom_list.pkl")
    covid_symptom_semantic_path = os.path.join(args.data_dir, "COVID_symptom_semantics.pkl")

    covid_df = pd.read_pickle(covid_data_path).reset_index(drop=True)
    covid_formula_composition = load_pickle(covid_formula_path)
    covid_symptom_list = load_pickle(covid_symptom_list_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = COVIDEmbeddingGenerator(
        model_dir=args.model_dir,
        save_dir=args.out_dir,
        device=device,
        covid_symptom_list_dir=covid_symptom_list_path,
        covid_symptom_semantic_dir=covid_symptom_semantic_path,
    )

    (
        clinical_df,
        case_initial_symptoms_for_embedding,
        case_improved_symptoms_for_embedding,
        case_unimproved_symptoms_for_embedding,
    ) = build_covid_clinical_dataframe(
        covid_df=covid_df,
        covid_formula_composition=covid_formula_composition,
        covid_symptom_list=covid_symptom_list,
        generator=generator,
    )

    if len(clinical_df) != len(covid_df):
        raise ValueError(
            f"Clinical dataframe rows do not match raw COVID rows: "
            f"{len(clinical_df)} vs {len(covid_df)}"
        )

    if not np.array_equal(
            clinical_df["Original_row_idx"].values,
            np.arange(len(covid_df)),
    ):
        raise ValueError("Original_row_idx is not aligned with raw COVID row order.")

    clinical_df.to_pickle(args.clinical_out)
    clinical_df.to_csv(os.path.join(args.out_dir, "COVID_19_data_for_embeddings.csv"), index=False, encoding="utf_8_sig",)


    case_herb_inputs = clinical_df["Formula"].tolist()

    print("COVID-19 clinical cases:", len(clinical_df))
    print("Prescription counts:")
    print(clinical_df["Formula_name"].value_counts())

    case_symptom_embeddings = generate_embeddings_in_batches(
        generator,
        case_initial_symptoms_for_embedding,
        mode="covid_symptom",
        batch_size=args.batch_size,
    )

    case_herb_embeddings = generate_embeddings_in_batches(
        generator,
        case_herb_inputs,
        mode="herb",
        batch_size=args.batch_size,
    )

    case_improved_symptom_embeddings = generate_embeddings_in_batches(
        generator,
        case_improved_symptoms_for_embedding,
        mode="covid_symptom",
        batch_size=args.batch_size,
    )

    case_unimproved_symptom_embeddings = generate_embeddings_in_batches(
        generator,
        case_unimproved_symptoms_for_embedding,
        mode="covid_symptom",
        batch_size=args.batch_size,
    )

    individual_symptom_embeddings = generate_embeddings_in_batches(
        generator,
        [[s] for s in covid_symptom_list],
        mode="covid_symptom",
        batch_size=args.batch_size,
    )


    unique_formula_names = list(covid_formula_composition.keys())
    formula_embedding_dict = {}
    for name in unique_formula_names:
        formula_embedding_dict[name] = generator.generate_herb_embedding(
            [unique_list(covid_formula_composition[name])]
        )[0]

    generator.save_embeddings({
        "case_symptom_embeddings": case_symptom_embeddings,
        "case_herb_embeddings": case_herb_embeddings,
        "case_improved_symptom_embeddings": case_improved_symptom_embeddings,
        "case_unimproved_symptom_embeddings": case_unimproved_symptom_embeddings,
        "individual_symptom_embeddings": individual_symptom_embeddings,
        "COVID_formula_embeddings": formula_embedding_dict,
    })

    print("Saved clinical dataframe:", args.clinical_out)
    print("Saved embeddings to:", args.out_dir)
    print("case_symptom_embeddings:", case_symptom_embeddings.shape)
    print("case_herb_embeddings:", case_herb_embeddings.shape)
    print("individual_symptom_embeddings:", individual_symptom_embeddings.shape)


if __name__ == "__main__":
    main()
