#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare co-occurrence-based baseline embeddings for comparison with TCM-ES.

The baselines are constructed from the training formula records only. Symptoms
and herbs are represented in one joint item space, and the same weighted
co-occurrence matrix is used for both the SVD and graph baselines.
"""

import argparse
import ast
import os
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def save_pickle(obj, path):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def parse_list_cell(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return list(x)
    if pd.isna(x):
        return []

    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return []

        try:
            y = ast.literal_eval(x)
            if isinstance(y, (list, tuple)):
                return list(y)
        except Exception:
            pass

        for sep in [";", "；", ",", "，"]:
            if sep in x:
                return [s.strip() for s in x.split(sep) if s.strip()]

        return [x]

    return []


def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"Cannot find required column from {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def terms_to_ids(term_list, term2id):
    return [term2id[x] for x in term_list if x in term2id]


def row_l2_normalize(X, eps=1e-8):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norm, eps)


# -----------------------------------------------------------------------------
# Joint symptom-herb co-occurrence matrix
# -----------------------------------------------------------------------------

def build_joint_binary_matrix(
    symptom_ids_by_record,
    herb_ids_by_record,
    n_symptoms,
    n_herbs,
):
    """Build the record-by-item binary matrix for symptoms and herbs."""
    n_records = len(symptom_ids_by_record)
    n_items = n_symptoms + n_herbs
    X = np.zeros((n_records, n_items), dtype=np.float32)

    for row_id, (symptom_ids, herb_ids) in enumerate(
        zip(symptom_ids_by_record, herb_ids_by_record)
    ):
        symptom_ids = sorted(set(symptom_ids))
        herb_ids = sorted(set(herb_ids))

        if symptom_ids:
            X[row_id, symptom_ids] = 1.0
        if herb_ids:
            X[row_id, [n_symptoms + h for h in herb_ids]] = 1.0

    return X


def compute_ppmi(X, counts, alpha=1e-8):
    """Compute positive pointwise mutual information from joint counts."""
    n_records = X.shape[0]
    frequency = X.sum(axis=0).astype(np.float64)
    counts64 = counts.astype(np.float64)

    numerator = (counts64 + alpha) * n_records
    denominator = (
        (frequency[:, None] + alpha)
        * (frequency[None, :] + alpha)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(numerator / denominator)

    ppmi = np.maximum(pmi, 0.0)
    ppmi[counts64 == 0] = 0.0
    np.fill_diagonal(ppmi, 0.0)
    return ppmi.astype(np.float32)


def build_weighted_cooccurrence_matrix(X, weighting="ppmi"):
    """
    Build the joint item-item co-occurrence matrix used by both baselines.

    weighting:
        count  : raw co-occurrence counts
        log1p  : log(1 + co-occurrence count)
        ppmi   : positive pointwise mutual information
    """
    counts = (X.T @ X).astype(np.float32)

    if weighting == "cooc":
        weighted = counts.copy()
        np.fill_diagonal(weighted, 0.0)

    elif weighting == "log1p":
        weighted = counts.copy()
        np.fill_diagonal(weighted, 0.0)
        weighted = np.log1p(weighted).astype(np.float32)

    elif weighting == "ppmi":
        weighted = compute_ppmi(X, counts)

    else:
        raise ValueError(
            "Unknown co-occurrence weighting: "
            f"{weighting}. Choose from count, log1p, or ppmi."
        )

    return counts, weighted.astype(np.float32)


# -----------------------------------------------------------------------------
# Co-occurrence SVD baseline
# -----------------------------------------------------------------------------

def build_svd_item_embeddings(
    weighted_matrix,
    dim,
    seed,
    normalize_items=True,
):
    """Generate shared symptom-herb embeddings using truncated SVD."""
    dim = min(dim, weighted_matrix.shape[0] - 1)

    svd = TruncatedSVD(
        n_components=dim,
        random_state=seed,
    )
    item_embeddings = svd.fit_transform(weighted_matrix).astype(np.float32)

    if normalize_items:
        item_embeddings = normalize(
            item_embeddings,
            norm="l2",
            axis=1,
        ).astype(np.float32)

    summary = pd.DataFrame({
        "component": np.arange(1, dim + 1),
        "singular_value": svd.singular_values_,
        "explained_variance_ratio": svd.explained_variance_ratio_,
        "explained_variance_ratio_cumulative": np.cumsum(
            svd.explained_variance_ratio_
        ),
    })

    return item_embeddings, summary


# -----------------------------------------------------------------------------
# Co-occurrence graph baseline
# -----------------------------------------------------------------------------

def build_sparse_cooccurrence_graph(
    weighted_matrix,
    counts,
    min_count=10,
    topk_per_node=100,
):
    """
    Construct a sparse weighted graph from the selected co-occurrence matrix.

    Edge weights are taken directly from `weighted_matrix`. Raw co-occurrence
    counts are used only for the minimum-count filter.
    """
    n_items = weighted_matrix.shape[0]
    all_idx = np.arange(n_items)
    rows, cols, values = [], [], []

    for i in range(n_items):
        keep = (
            (all_idx != i)
            & (counts[i] >= min_count)
            & (weighted_matrix[i] > 0)
        )
        js = np.where(keep)[0]

        if topk_per_node is not None and len(js) > topk_per_node:
            js = js[
                np.argsort(-weighted_matrix[i, js])[:topk_per_node]
            ]

        if len(js) == 0:
            continue

        rows.extend([i] * len(js))
        cols.extend(js.tolist())
        values.extend(weighted_matrix[i, js].astype(np.float32).tolist())

    W = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(n_items, n_items),
        dtype=np.float32,
    ).tocsr()

    W = W.maximum(W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def sparse_to_networkx_graph(W):
    """Convert the sparse weighted adjacency matrix to a NetworkX graph."""
    W = W.tocoo()
    graph = nx.Graph()

    for i in range(W.shape[0]):
        graph.add_node(str(i))

    for i, j, weight in zip(W.row, W.col, W.data):
        if i < j and weight > 0:
            graph.add_edge(str(i), str(j), weight=float(weight))

    return graph


def build_graph_item_embeddings(
    W,
    dim,
    seed,
    walk_length=30,
    num_walks=10,
    window=10,
    epochs=5,
    workers=4,
    p=1.0,
    q=1.0,
    normalize_items=True,
):
    """Generate graph embeddings using weighted Node2Vec random walks."""
    try:
        from node2vec import Node2Vec
    except ImportError as exc:
        raise ImportError(
            "The graph baseline requires the 'node2vec' package."
        ) from exc

    graph = sparse_to_networkx_graph(W)

    node2vec = Node2Vec(
        graph,
        dimensions=dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        weight_key="weight",
        workers=workers,
        seed=seed,
    )

    model = node2vec.fit(
        window=window,
        min_count=1,
        batch_words=256,
        epochs=epochs,
        seed=seed,
    )

    item_embeddings = np.zeros((W.shape[0], dim), dtype=np.float32)
    for i in range(W.shape[0]):
        key = str(i)
        if key in model.wv:
            item_embeddings[i] = model.wv[key]

    if normalize_items:
        item_embeddings = normalize(
            item_embeddings,
            norm="l2",
            axis=1,
        ).astype(np.float32)

    summary = pd.DataFrame({
        "parameter": [
            "method",
            "dim",
            "walk_length",
            "num_walks",
            "window",
            "epochs",
            "p",
            "q",
            "workers",
            "n_nodes",
            "n_edges",
        ],
        "value": [
            "weighted_Node2Vec",
            dim,
            walk_length,
            num_walks,
            window,
            epochs,
            p,
            q,
            workers,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        ],
    })

    return item_embeddings, summary


# -----------------------------------------------------------------------------
# Record-level embeddings
# -----------------------------------------------------------------------------

def average_item_embeddings(
    item_ids_by_record,
    item_embeddings,
    offset=0,
    normalize_records=False,
):
    """Pool constituent item embeddings to obtain record-level embeddings."""
    dim = item_embeddings.shape[1]
    output = np.zeros((len(item_ids_by_record), dim), dtype=np.float32)

    for row_id, item_ids in enumerate(item_ids_by_record):
        item_ids = sorted(set(item_ids))
        if not item_ids:
            continue

        joint_ids = [offset + i for i in item_ids]
        output[row_id] = item_embeddings[joint_ids].mean(axis=0)

    if normalize_records:
        output = row_l2_normalize(output).astype(np.float32)

    return output


def export_baseline_embeddings(
    out_dir,
    selected_formula_ids,
    individual_symptom_embeddings,
    individual_herb_embeddings,
    selected_formula_symptom_embeddings,
    selected_formula_herb_embeddings,
):
    ensure_dir(out_dir)

    save_pickle(
        individual_symptom_embeddings.astype(np.float32),
        os.path.join(out_dir, "individual_symptom_embeddings.pkl"),
    )
    save_pickle(
        individual_herb_embeddings.astype(np.float32),
        os.path.join(out_dir, "individual_herb_embeddings.pkl"),
    )
    save_pickle(
        selected_formula_symptom_embeddings.astype(np.float32),
        os.path.join(out_dir, "selected_formula_symptom_embeddings.pkl"),
    )
    save_pickle(
        selected_formula_herb_embeddings.astype(np.float32),
        os.path.join(out_dir, "selected_formula_herb_embeddings.pkl"),
    )
    save_pickle(
        list(selected_formula_ids),
        os.path.join(out_dir, "selected_formula_ids(line_number).pkl"),
    )


def export_symptom_frequency_metadata(X_train, symptom_list, out_dirs):
    """Save training-set marginal symptom frequencies for downstream clinical analyses."""
    frequency_in_train = np.asarray(
        X_train[:, :len(symptom_list)].sum(axis=0)
    ).ravel().astype(int)

    frequency_df = pd.DataFrame({
        "local_entity_id": np.arange(len(symptom_list), dtype=int),
        "item_name": list(symptom_list),
        "frequency_in_train": frequency_in_train,
    })

    for out_dir in out_dirs:
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, "symptom_frequency_in_train.csv")
        frequency_df.to_csv(
            out_path,
            index=False,
            encoding="utf_8_sig",
        )
        print("Saved training symptom frequencies to:", out_path)

def export_general_clinical_embeddings(
    out_dir,
    case_ids,
    symptom_embeddings,
    formula_embeddings,
):
    ensure_dir(out_dir)
    save_pickle(
        symptom_embeddings.astype(np.float32),
        os.path.join(out_dir, "case_symptom_embeddings.pkl"),
    )
    save_pickle(
        formula_embeddings.astype(np.float32),
        os.path.join(out_dir, "case_herb_embeddings.pkl"),
    )
    save_pickle(
        list(case_ids),
        os.path.join(out_dir, "case_ids.pkl"),
    )


# -----------------------------------------------------------------------------
# Optional COVID-19 case embeddings
# -----------------------------------------------------------------------------

def build_semantic_matched_symptom_embeddings(
    target_symptom_list,
    target_symptom_semantics,
    source_symptom_list,
    source_symptom_semantics,
    source_symptom_embeddings,
    topk=1,
):
    """Map external symptom terms to the nearest standardized TCM symptoms."""
    target_semantics = np.asarray(
        target_symptom_semantics[:len(target_symptom_list)],
        dtype=np.float32,
    )
    source_semantics = np.asarray(
        source_symptom_semantics[:len(source_symptom_list)],
        dtype=np.float32,
    )

    target_semantics = row_l2_normalize(target_semantics)
    source_semantics = row_l2_normalize(source_semantics)
    similarity = target_semantics @ source_semantics.T

    topk = int(min(topk, similarity.shape[1]))
    topk_idx = np.argsort(-similarity, axis=1)[:, :topk]
    topk_similarity = np.take_along_axis(similarity, topk_idx, axis=1)

    if topk == 1:
        target_embeddings = source_symptom_embeddings[topk_idx[:, 0]].copy()
    else:
        weights = topk_similarity - topk_similarity.max(axis=1, keepdims=True)
        weights = np.exp(weights)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        target_embeddings = np.sum(
            source_symptom_embeddings[topk_idx] * weights[:, :, None],
            axis=1,
        )

    target_embeddings = row_l2_normalize(target_embeddings).astype(np.float32)

    mapping_rows = []
    for i, symptom in enumerate(target_symptom_list):
        mapping_rows.append({
            "target_symptom": symptom,
            "best_source_symptom": source_symptom_list[int(topk_idx[i, 0])],
            "best_source_index": int(topk_idx[i, 0]),
            "best_cosine_similarity": float(topk_similarity[i, 0]),
        })

    return target_embeddings, pd.DataFrame(mapping_rows)


def export_covid_case_embeddings(
    out_dir,
    case_ids,
    individual_symptom_embeddings,
    case_symptom_embeddings,
    case_formula_embeddings,
    symptom_mapping_df,
):
    ensure_dir(out_dir)
    save_pickle(
        individual_symptom_embeddings.astype(np.float32),
        os.path.join(out_dir, "individual_symptom_embeddings.pkl"),
    )
    save_pickle(
        case_symptom_embeddings.astype(np.float32),
        os.path.join(out_dir, "case_symptom_embeddings.pkl"),
    )
    save_pickle(
        case_formula_embeddings.astype(np.float32),
        os.path.join(out_dir, "case_herb_embeddings.pkl"),
    )
    save_pickle(list(case_ids), os.path.join(out_dir, "case_ids.pkl"))
    symptom_mapping_df.to_csv(
        os.path.join(out_dir, "COVID_to_TCM_symptom_semantic_matching.csv"),
        index=False,
        encoding="utf_8_sig",
    )


def prepare_covid_inputs(args, symptom_list, symptom_item_embeddings, herb2id):
    covid_data = pd.read_pickle(args.covid_clinical_data).reset_index(drop=True)
    covid_symptom_list = load_pickle(args.covid_symptom_list)
    covid_symptom_semantics = np.asarray(
        load_pickle(args.covid_symptom_semantics),
        dtype=np.float32,
    )
    tcm_symptom_semantics = np.asarray(
        load_pickle(args.tcm_symptom_semantics),
        dtype=np.float32,
    )

    covid_sym2id = {s: i for i, s in enumerate(covid_symptom_list)}
    case_ids = list(covid_data["Cases_id"])
    case_symptom_ids = [
        terms_to_ids(parse_list_cell(x), covid_sym2id)
        for x in covid_data["Initial_symptoms"].tolist()
    ]
    case_herb_ids = [
        terms_to_ids(parse_list_cell(x), herb2id)
        for x in covid_data["Formula"].tolist()
    ]

    mapped_symptom_embeddings, mapping_df = (
        build_semantic_matched_symptom_embeddings(
            target_symptom_list=covid_symptom_list,
            target_symptom_semantics=covid_symptom_semantics,
            source_symptom_list=symptom_list,
            source_symptom_semantics=tcm_symptom_semantics,
            source_symptom_embeddings=symptom_item_embeddings,
            topk=args.semantic_match_topk,
        )
    )

    return (
        case_ids,
        case_symptom_ids,
        case_herb_ids,
        mapped_symptom_embeddings,
        mapping_df,
    )


# -----------------------------------------------------------------------------
# Data loading and export workflow
# -----------------------------------------------------------------------------

def load_formula_inputs(args):
    formula_data = pd.read_pickle(args.formula_data).reset_index(drop=True)
    symptom_list = load_pickle(args.symptom_list)
    herb_list = load_pickle(args.herb_list)

    symptom_col = find_column(
        formula_data,
        ["Symptoms", "symptoms", "Symptom", "symptom"],
    )
    herb_col = find_column(
        formula_data,
        ["Herbs", "herbs", "Formula", "formula"],
    )

    sym2id = {s: i for i, s in enumerate(symptom_list)}
    herb2id = {h: i for i, h in enumerate(herb_list)}

    selected_ids_path = os.path.join(
        args.learned_emb_dir,
        "selected_formula_ids(line_number).pkl",
    )
    selected_formula_ids = [int(x) for x in load_pickle(selected_ids_path)]

    learned_symptom_embeddings = load_pickle(
        os.path.join(
            args.learned_emb_dir,
            "selected_formula_symptom_embeddings.pkl",
        )
    )
    learned_formula_embeddings = load_pickle(
        os.path.join(
            args.learned_emb_dir,
            "selected_formula_herb_embeddings.pkl",
        )
    )

    if not (
        len(selected_formula_ids)
        == len(learned_symptom_embeddings)
        == len(learned_formula_embeddings)
    ):
        raise ValueError(
            "Selected formula IDs and learned TCM-ES embeddings are not aligned."
        )

    selected_data = formula_data.iloc[selected_formula_ids].reset_index(drop=True)
    selected_symptom_ids = [
        terms_to_ids(parse_list_cell(x), sym2id)
        for x in selected_data[symptom_col].tolist()
    ]
    selected_herb_ids = [
        terms_to_ids(parse_list_cell(x), herb2id)
        for x in selected_data[herb_col].tolist()
    ]

    train_idx = [int(x) for x in load_pickle(args.train_idx)]
    train_data = formula_data.iloc[train_idx].reset_index(drop=True)
    train_symptom_ids = [
        terms_to_ids(parse_list_cell(x), sym2id)
        for x in train_data[symptom_col].tolist()
    ]
    train_herb_ids = [
        terms_to_ids(parse_list_cell(x), herb2id)
        for x in train_data[herb_col].tolist()
    ]

    return {
        "formula_data": formula_data,
        "symptom_list": symptom_list,
        "herb_list": herb_list,
        "sym2id": sym2id,
        "herb2id": herb2id,
        "selected_formula_ids": selected_formula_ids,
        "selected_symptom_ids": selected_symptom_ids,
        "selected_herb_ids": selected_herb_ids,
        "train_symptom_ids": train_symptom_ids,
        "train_herb_ids": train_herb_ids,
    }


def load_general_clinical_inputs(args, sym2id, herb2id):
    clinical_data = pd.read_pickle(args.general_clinical_data).reset_index(drop=True)

    case_ids = list(clinical_data["Cases_id"])
    case_symptom_ids = [
        terms_to_ids(parse_list_cell(x), sym2id)
        for x in clinical_data["Initial_symptoms"].tolist()
    ]
    case_herb_ids = [
        terms_to_ids(parse_list_cell(x), herb2id)
        for x in clinical_data["Formula"].tolist()
    ]

    return case_ids, case_symptom_ids, case_herb_ids


def export_one_baseline(
    name,
    item_embeddings,
    out_dir,
    inputs,
    clinical_inputs,
    args,
):
    n_symptoms = len(inputs["symptom_list"])
    n_herbs = len(inputs["herb_list"])

    symptom_item_embeddings = item_embeddings[:n_symptoms]
    herb_item_embeddings = item_embeddings[n_symptoms:n_symptoms + n_herbs]

    formula_symptom_embeddings = average_item_embeddings(
        inputs["selected_symptom_ids"],
        item_embeddings,
        offset=0,
        normalize_records=args.normalize_record_embeddings,
    )
    formula_herb_embeddings = average_item_embeddings(
        inputs["selected_herb_ids"],
        item_embeddings,
        offset=n_symptoms,
        normalize_records=args.normalize_record_embeddings,
    )

    export_baseline_embeddings(
        out_dir=out_dir,
        selected_formula_ids=inputs["selected_formula_ids"],
        individual_symptom_embeddings=symptom_item_embeddings,
        individual_herb_embeddings=herb_item_embeddings,
        selected_formula_symptom_embeddings=formula_symptom_embeddings,
        selected_formula_herb_embeddings=formula_herb_embeddings,
    )

    case_ids, case_symptom_ids, case_herb_ids = clinical_inputs
    clinical_symptom_embeddings = average_item_embeddings(
        case_symptom_ids,
        item_embeddings,
        offset=0,
        normalize_records=args.normalize_record_embeddings,
    )
    clinical_formula_embeddings = average_item_embeddings(
        case_herb_ids,
        item_embeddings,
        offset=n_symptoms,
        normalize_records=args.normalize_record_embeddings,
    )

    export_general_clinical_embeddings(
        out_dir=os.path.join(out_dir, "general_clinical_cases"),
        case_ids=case_ids,
        symptom_embeddings=clinical_symptom_embeddings,
        formula_embeddings=clinical_formula_embeddings,
    )

    if args.include_covid:
        (
            covid_case_ids,
            covid_case_symptom_ids,
            covid_case_herb_ids,
            covid_symptom_embeddings,
            mapping_df,
        ) = prepare_covid_inputs(
            args,
            inputs["symptom_list"],
            symptom_item_embeddings,
            inputs["herb2id"],
        )

        covid_case_symptom_embeddings = average_item_embeddings(
            covid_case_symptom_ids,
            covid_symptom_embeddings,
            offset=0,
            normalize_records=args.normalize_record_embeddings,
        )
        covid_case_formula_embeddings = average_item_embeddings(
            covid_case_herb_ids,
            item_embeddings,
            offset=n_symptoms,
            normalize_records=args.normalize_record_embeddings,
        )

        export_covid_case_embeddings(
            out_dir=os.path.join(out_dir, "COVID_19_cases"),
            case_ids=covid_case_ids,
            individual_symptom_embeddings=covid_symptom_embeddings,
            case_symptom_embeddings=covid_case_symptom_embeddings,
            case_formula_embeddings=covid_case_formula_embeddings,
            symptom_mapping_df=mapping_df,
        )

    print(f"{name} embeddings saved to: {out_dir}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare co-occurrence SVD and graph baseline embeddings for TCM-ES."
        )
    )

    parser.add_argument(
        "--formula-data",
        default="data/TCM_formulas/TCM_formula_data_example_2000.pkl",
    )
    parser.add_argument(
        "--learned-emb-dir",
        default="results/embeddings/TCM_embeddings",
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
        "--train-idx",
        default="data/training/train_idx.pkl",
    )
    parser.add_argument(
        "--general-clinical-data",
        default=(
            "data/general_TCM_clinical_cases/"
            "TCM_general_cases_data_example_500_model_eligible.pkl"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="results/embedding_baseline_comparison",
    )

    parser.add_argument(
        "--cooc-weighting",
        choices=["cooc", "log1p", "ppmi"],
        default="cooc",
        help=(
            "Weighting of the joint symptom-herb co-occurrence matrix. "
            "The same matrix is used by the SVD and graph baselines."
        ),
    )
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--graph-min-count", type=int, default=10)
    parser.add_argument("--graph-topk-per-node", type=int, default=100)
    parser.add_argument("--graph-workers", type=int, default=4)

    parser.add_argument(
        "--no-normalize-item-embeddings",
        action="store_true",
        help="Disable L2 normalization of individual item embeddings.",
    )
    parser.add_argument(
        "--normalize-record-embeddings",
        action="store_true",
        help="L2-normalize pooled record-level embeddings.",
    )
    parser.add_argument(
        "--save-cooccurrence-matrices",
        action="store_true",
        help="Save the raw count and weighted co-occurrence matrices.",
    )

    parser.add_argument(
        "--include-covid",
        action="store_true",
        help="Also export baseline embeddings for the COVID-19 clinical dataset.",
    )
    parser.add_argument(
        "--covid-clinical-data",
        default="data/COVID_19_data/COVID_19_data_for_embeddings.pkl",
    )
    parser.add_argument(
        "--covid-symptom-list",
        default="data/COVID_19_data/COVID_symptom_list.pkl",
    )
    parser.add_argument(
        "--covid-symptom-semantics",
        default="data/COVID_19_data/COVID_symptom_semantics.pkl",
    )
    parser.add_argument(
        "--tcm-symptom-semantics",
        default="core/standard_TCM_entities/symptom_semantic_encodings.pkl",
    )
    parser.add_argument("--semantic-match-topk", type=int, default=1)

    return parser


def main():
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)

    inputs = load_formula_inputs(args)
    clinical_inputs = load_general_clinical_inputs(
        args,
        inputs["sym2id"],
        inputs["herb2id"],
    )

    n_symptoms = len(inputs["symptom_list"])
    n_herbs = len(inputs["herb_list"])

    X_train = build_joint_binary_matrix(
        symptom_ids_by_record=inputs["train_symptom_ids"],
        herb_ids_by_record=inputs["train_herb_ids"],
        n_symptoms=n_symptoms,
        n_herbs=n_herbs,
    )

    counts, weighted_matrix = build_weighted_cooccurrence_matrix(
        X_train,
        weighting=args.cooc_weighting,
    )

    print("Training records:", X_train.shape[0])
    print("Joint item space:", X_train.shape[1])
    print("Co-occurrence weighting:", args.cooc_weighting)

    if args.save_cooccurrence_matrices:
        np.save(
            os.path.join(args.out_dir, "joint_cooccurrence_counts.npy"),
            counts,
        )
        np.save(
            os.path.join(
                args.out_dir,
                f"joint_cooccurrence_{args.cooc_weighting}.npy",
            ),
            weighted_matrix,
        )

    normalize_items = not args.no_normalize_item_embeddings

    # Co-occurrence SVD baseline
    svd_dir = os.path.join(
        args.out_dir,
        f"baseline_{args.cooc_weighting}_svd",
    )
    svd_item_embeddings, svd_summary = build_svd_item_embeddings(
        weighted_matrix=weighted_matrix,
        dim=args.dim,
        seed=args.seed,
        normalize_items=normalize_items,
    )
    ensure_dir(svd_dir)
    export_symptom_frequency_metadata(
        X_train=X_train,
        symptom_list=inputs["symptom_list"],
        out_dirs=[svd_dir],
    )
    svd_summary.to_csv(
        os.path.join(svd_dir, "svd_component_summary.csv"),
        index=False,
    )
    export_one_baseline(
        name="SVD baseline",
        item_embeddings=svd_item_embeddings,
        out_dir=svd_dir,
        inputs=inputs,
        clinical_inputs=clinical_inputs,
        args=args,
    )

    # Co-occurrence graph baseline
    graph_dir = os.path.join(
        args.out_dir,
        f"baseline_{args.cooc_weighting}_graph",
    )
    graph_adjacency = build_sparse_cooccurrence_graph(
        weighted_matrix=weighted_matrix,
        counts=counts,
        min_count=args.graph_min_count,
        topk_per_node=args.graph_topk_per_node,
    )
    ensure_dir(graph_dir)
    export_symptom_frequency_metadata(
        X_train=X_train,
        symptom_list=inputs["symptom_list"],
        out_dirs=[graph_dir],
    )
    sparse.save_npz(
        os.path.join(graph_dir, "graph_sparse_adjacency.npz"),
        graph_adjacency,
    )

    graph_item_embeddings, graph_summary = build_graph_item_embeddings(
        W=graph_adjacency,
        dim=args.dim,
        seed=args.seed,
        workers=args.graph_workers,
        normalize_items=normalize_items,
    )
    graph_summary.to_csv(
        os.path.join(graph_dir, "graph_embedding_summary.csv"),
        index=False,
    )
    export_one_baseline(
        name="Graph baseline",
        item_embeddings=graph_item_embeddings,
        out_dir=graph_dir,
        inputs=inputs,
        clinical_inputs=clinical_inputs,
        args=args,
    )

    print("Done.")
    print("Outputs saved to:", args.out_dir)


if __name__ == "__main__":
    main()
