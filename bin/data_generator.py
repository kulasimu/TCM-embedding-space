import numpy as np
import torch
import random as Random
import pickle
import sys
import torch.utils.data as Data
import pandas as pd
import os
import argparse
import math
import re
from collections import defaultdict


parser = argparse.ArgumentParser()

parser.add_argument(
    "out_dir",
    nargs="?",
    default="data/training",
    help="Model input data saving directory."
)
parser.add_argument(
    "--formula-data",
    type=str,
    default="data/TCM_formulas/TCM_formula_data_example_2000.pkl",
    help="TCM formula-record dataframe used for model-data preparation."
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--same-title-jaccard", type=float, default=0.6)
parser.add_argument("--cross-title-jaccard", type=float, default=0.9)
parser.add_argument("--num-repeats", type=int, default=1)
parser.add_argument(
    "--make-dataloader",
    action="store_true",
    help="Generate dataloaders after loading an existing split or creating a new split if needed."
)
parser.add_argument(
    "--regenerate-split",
    action="store_true",
    help="Generate a new split before dataloader preparation instead of using existing split files."
)
parser.add_argument("--make-null-dataloader", action="store_true")
parser.add_argument("--show-data", action="store_true")

# External TCM-PD all-valid formula-record dataset preparation.
# This branch parses, checks, and length-filters the external benchmark data.
# Similarity filtering for the external test set is performed in the downstream prediction analysis.
parser.add_argument("--make-external-pd-all-valid", action="store_true")
parser.add_argument("--external-xlsx", type=str, default="data/TCM_PD_external/TCM_PD_combined_30527.xlsx")
parser.add_argument("--external-out-dir", type=str, default="data/TCM_PD_external")
parser.add_argument("--external-split-col", type=str, default="Train_test_split")
parser.add_argument("--external-symptom-col", type=str, default="standard_symptom_names")
parser.add_argument("--external-herb-col", type=str, default="standard_herb_names")

args = parser.parse_args()
out_dir = args.out_dir
seed = args.seed

if args.num_repeats < 1:
    raise ValueError("--num-repeats must be at least 1.")

if args.make_null_dataloader and args.num_repeats > 1:
    raise ValueError("--make-null-dataloader is only supported when --num-repeats is 1.")


######### Configuration
MAX_HERB_LEN = 31 # Max number of herbs and a start/ending sign
MAX_SYM_LEN = 41 # Max number of symptoms and a start/ending sign
batch_size = 256


######### Load standardized TCM entities

# Standardized herb terms, 780 most frequently-prescribed herbs are included
with open('core/standard_TCM_entities/herb_list.pkl', 'rb') as fp:
    herb_list = pickle.load(fp)
num_herbs = len(herb_list)

# Standardized symptom terms, 1436 symptoms are included
with open('core/standard_TCM_entities/symptom_list.pkl', 'rb') as fp:
    symptom_list = pickle.load(fp)
num_symptoms = len(symptom_list)

# Semantic encodings of 1436 symptoms and special tokens
with open('core/standard_TCM_entities/symptom_semantic_encodings.pkl','rb') as fp:
    symptom_semantics = pickle.load(fp)

########################################################################################################################
# Functions used to generate dataloader for model input and output
########################################################################################################################

def generate_herb_data(herb_id_seq,inOrOut):
    herb_data = [num_herbs + 1] + herb_id_seq if inOrOut == 'input' else herb_id_seq + [num_herbs + 2] # Add 'Start' or 'End'
    herb_data = herb_data + (MAX_HERB_LEN-len(herb_data))*[num_herbs] # Padding
    return herb_data

def generate_sym_data(sym_id_seq,inOrOut):
    sym_idx = [num_symptoms + 1] + sym_id_seq if inOrOut == 'input' else sym_id_seq + [num_symptoms + 2] # Add 'Start' or 'End'
    sym_idx = sym_idx + (MAX_SYM_LEN-len(sym_idx))*[num_symptoms] # Padding
    sym_data = [symptom_semantics[s] for s in sym_idx]
    return sym_data if inOrOut == 'input' else sym_idx

def prepare_torch_data(data_idx, data_pd):
    formula_IDs = list(data_pd['ID'])
    formula_symptoms = list(data_pd['Symptoms'])
    formula_herbs = list(data_pd['Herbs'])
    # formula_symptom_vectors = np.array(list(example_data_pd['Symptom vectors']))
    # formula_herb_vectors = np.array(list(example_data_pd['Herb vectors']))

    x1 = []  # Model input 1: Symptoms GUSE embedding input start with 'Start'
    x2 = []  # Model input 2: Herbs id input start with 'Start'
    y1 = []  # Model output 1: Symptoms id output end with 'End'
    y2 = []  # Model output 2: Herbs id output end with 'End'
    data_label = []  # Symptom and herb labels of data, used to calculate similarity between data records

    for i in data_idx:
        symptom_id_seq = [symptom_list.index(s) for s in formula_symptoms[i]]
        herb_id_seq = [herb_list.index(h) for h in formula_herbs[i]]
        n_augment = int(max(len(symptom_id_seq), len(herb_id_seq)))

        # print('ID: ', formula_IDs[i])
        # print('Herbs: ', " ".join(formula_herbs[i]))
        # print('Symptoms: ', " ".join(formula_symptoms[i]))

        # A quick check
        n_s = len(formula_symptoms[i])
        n_h = len(formula_herbs[i])
        if not (n_s >= 3 and n_h >= 3 and n_s < MAX_SYM_LEN and n_h < MAX_HERB_LEN):
            raise ValueError(
                f"Formula record {formula_IDs[i]} does not meet the required conditions: "
                f"n_symptoms={n_s}, n_herbs={n_h}"
            )

        formula_symptom_vector = np.zeros(num_symptoms)
        for s in symptom_id_seq:
            formula_symptom_vector[s] = 1
        formula_herb_vector = np.zeros(num_herbs)
        for h in herb_id_seq:
            formula_herb_vector[h] = 1

        for _ in range(n_augment):
            symptom_id_seq_temp = symptom_id_seq.copy()
            Random.shuffle(symptom_id_seq_temp)
            x1.append(generate_sym_data(symptom_id_seq_temp, 'input'))
            y1.append(generate_sym_data(symptom_id_seq_temp, 'output'))

            herb_id_seq_tmp = herb_id_seq.copy()
            Random.shuffle(herb_id_seq_tmp)
            x2.append(generate_herb_data(herb_id_seq_tmp, 'input'))
            y2.append(generate_herb_data(herb_id_seq_tmp, 'output'))

            data_label.append(np.concatenate((formula_symptom_vector, formula_herb_vector)))
            # print(data_label[-1].shape)

    x1, x2, y1, y2, data_label = np.array(x1), np.array(x2), np.array(y1), np.array(y2), np.array(data_label)

    torch_dataset = Data.TensorDataset(torch.FloatTensor(x1), torch.LongTensor(x2), torch.LongTensor(y1), torch.LongTensor(y2), torch.IntTensor(data_label))
    data_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def jaccard_similarity(set_a, set_b):
    set_a = set(set_a)
    set_b = set(set_b)

    union = set_a | set_b
    if len(union) == 0:
        return 0.0

    return len(set_a & set_b) / len(union)


def make_deranged_indices(indices, seed=42):
    """
    Return a deranged permutation of indices.
    No index is paired with itself.
    """
    rng = np.random.default_rng(seed)
    indices = np.array(list(indices))

    if len(indices) <= 1:
        return indices.copy()

    for _ in range(10000):
        permuted = rng.permutation(indices)
        if np.all(permuted != indices):
            return permuted

    raise RuntimeError("Failed to generate a deranged permutation.")


def make_splitwise_shuffled_formula_dataframe(
        formula_data_pd,
        train_idx,
        val_idx,
        test_idx,
        seed=42
):
    """
    Create shuffled-pairing null dataset within each split.
    This avoids train/val/test cross-contamination.
    """
    formula_data_pd = formula_data_pd.reset_index(drop=True)
    formula_data_null_pd = formula_data_pd.copy(deep=True)

    formula_data_null_pd["Original_row_idx"] = np.arange(len(formula_data_pd))
    formula_data_null_pd["Herb_shuffled_row_idx"] = np.arange(len(formula_data_pd))
    formula_data_null_pd["Shuffle_split"] = "none"

    split_dict = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }

    for k, (split_name, split_indices) in enumerate(split_dict.items()):
        split_indices = list(split_indices)
        herb_shuffled_indices = make_deranged_indices(
            split_indices,
            seed=seed + k
        )

        for original_indices, shuffled_indices in zip(split_indices, herb_shuffled_indices):
            formula_data_null_pd.at[original_indices, "Herbs"] = list(formula_data_pd.at[shuffled_indices, "Herbs"])
            formula_data_null_pd.at[original_indices, "Herb_shuffled_row_idx"] = shuffled_indices
            formula_data_null_pd.at[original_indices, "Shuffle_split"] = split_name

        similar_herb_sets = 0
        jaccard_threshold = 0.6
        jaccard_values = []

        for i in split_indices:
            j = formula_data_null_pd.at[i, "Herb_shuffled_row_idx"]

            herbs_i_original = set(formula_data_pd.at[i, "Herbs"])
            herbs_j_shuffled = set(formula_data_pd.at[j, "Herbs"])

            jac = jaccard_similarity(herbs_i_original, herbs_j_shuffled)
            jaccard_values.append(jac)

            if jac >= jaccard_threshold:
                similar_herb_sets += 1

        print(
            split_name,
            f"herb sets with Jaccard >= {jaccard_threshold}:",
            similar_herb_sets,
            f"out of {len(split_indices)}"
        )


    return formula_data_null_pd

def get_valid_indices(data_pd):
    """
    Get formula indices satisfying the same inclusion criteria used by prepare_torch_data.
    """
    data_idx = list(np.arange(len(data_pd)))
    formula_symptoms = list(data_pd["Symptoms"])
    formula_herbs = list(data_pd["Herbs"])

    data_idx_subset = []
    for idx in data_idx:
        n_s = len(formula_symptoms[idx])
        n_h = len(formula_herbs[idx])
        if n_s >= 3 and n_h >= 3 and n_s < MAX_SYM_LEN and n_h < MAX_HERB_LEN:
            data_idx_subset.append(idx)

    return data_idx_subset



def split_indices_622(
        data_pd,
        data_idx_subset,
        split_seed,
        same_title_jaccard_threshold=0.6,
        cross_title_jaccard_threshold=0.9
):
    """
    Split formula records into training, validation, and test sets using approximately 6:2:2.

    Formula-similarity grouping is applied before splitting when the corresponding
    Jaccard threshold is within [0, 1]. A threshold greater than 1 disables that
    grouping rule. When both thresholds are greater than 1, the split reduces to
    the record-level random split used for the primary model.
    """
    if same_title_jaccard_threshold > 1 and cross_title_jaccard_threshold > 1:
        rng = Random.Random(split_seed)
        data_idx_subset = sorted(data_idx_subset)

        total_size = len(data_idx_subset)
        train_size = int(total_size * 0.6)
        val_size = int(total_size * 0.2)

        train_idx = rng.sample(data_idx_subset, train_size)

        remaining_idx = sorted(set(data_idx_subset) - set(train_idx))
        val_idx = rng.sample(remaining_idx, val_size)

        test_idx = sorted(set(data_idx_subset) - set(train_idx) - set(val_idx))

        return train_idx, val_idx, test_idx, []

    idx = sorted(data_idx_subset)
    n = len(idx)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    titles = [
        "".join(str(data_pd.at[row_i, "Title"]).split())
        for row_i in idx
    ]

    herb_sets = [
        set(data_pd.at[row_i, "Herbs"])
        for row_i in idx
    ]

    similar_pairs = []

    ########################################################################
    # 1. Group same-title records with sufficiently similar herb compositions
    ########################################################################
    same_title_edges = 0

    if same_title_jaccard_threshold <= 1:
        title_map = defaultdict(list)

        for local_i, title in enumerate(titles):
            title_map[title].append(local_i)

        for members in title_map.values():
            if len(members) <= 1:
                continue

            for a_pos in range(len(members)):
                a = members[a_pos]
                hs_a = herb_sets[a]

                for b in members[a_pos + 1:]:
                    hs_b = herb_sets[b]
                    jac = len(hs_a & hs_b) / len(hs_a | hs_b)

                    if jac >= same_title_jaccard_threshold:
                        union(a, b)
                        same_title_edges += 1
                        similar_pairs.append({
                            "row_i": idx[a],
                            "row_j": idx[b],
                            "reason": "same_title_herb_similar",
                            "jaccard": jac
                        })

    ########################################################################
    # 2. Group cross-title records with highly similar herb compositions
    ########################################################################
    cross_title_edges = 0

    if cross_title_jaccard_threshold <= 1:
        herb_freq = defaultdict(int)

        for hs in herb_sets:
            for h in hs:
                herb_freq[h] += 1

        ordered_sets = [
            tuple(sorted(hs, key=lambda h: (herb_freq[h], h)))
            for hs in herb_sets
        ]

        inverted = defaultdict(list)
        threshold = cross_title_jaccard_threshold

        for i, hs_i in enumerate(herb_sets):
            len_i = len(hs_i)

            prefix_len = max(
                1,
                len_i - math.ceil(threshold * len_i) + 1
            )

            candidates = set()

            for h in ordered_sets[i][:prefix_len]:
                candidates.update(inverted[h])

            for j in candidates:
                # Same-title pairs are controlled by the same-title threshold.
                if titles[i] == titles[j]:
                    continue

                len_j = len(herb_sets[j])

                # Length filter.
                if min(len_i, len_j) / max(len_i, len_j) < threshold:
                    continue

                jac = len(hs_i & herb_sets[j]) / len(hs_i | herb_sets[j])

                if jac >= threshold:
                    union(i, j)
                    cross_title_edges += 1

                    similar_pairs.append({
                        "row_i": idx[i],
                        "row_j": idx[j],
                        "reason": "cross_title_herb_similar",
                        "jaccard": jac
                    })

            for h in ordered_sets[i][:prefix_len]:
                inverted[h].append(i)

    ########################################################################
    # 3. Collect groups of formula-similar records
    ########################################################################
    groups = defaultdict(list)

    for local_i, row_i in enumerate(idx):
        groups[find(local_i)].append(row_i)

    group_list = list(groups.values())

    ########################################################################
    # 4. Split groups as indivisible units
    ########################################################################
    rng = Random.Random(split_seed)
    rng.shuffle(group_list)
    group_list.sort(key=len, reverse=True)

    targets = {
        "train": n * 0.6,
        "val": n * 0.2,
        "test": n * 0.2
    }

    split = {
        "train": [],
        "val": [],
        "test": []
    }

    for g in group_list:
        best_split = min(
            split,
            key=lambda k: sum(
                abs(
                    len(split[s])
                    + (len(g) if s == k else 0)
                    - targets[s]
                )
                for s in split
            )
        )

        split[best_split].extend(g)

    group_sizes = sorted([len(g) for g in group_list], reverse=True)

    print(
        f"Split groups: {len(group_list)}; "
        f"largest groups: {group_sizes[:10]}; "
        f"same-title edges: {same_title_edges}; "
        f"cross-title edges: {cross_title_edges}; "
        f"same-title Jaccard threshold: {same_title_jaccard_threshold}; "
        f"cross-title Jaccard threshold: {cross_title_jaccard_threshold}"
    )

    return sorted(split["train"]), sorted(split["val"]), sorted(split["test"]), similar_pairs

def save_split_indices(split_dir, train_idx, val_idx, test_idx):
    os.makedirs(split_dir, exist_ok=True)

    with open(os.path.join(split_dir, "train_idx.pkl"), "wb") as fp:
        pickle.dump(train_idx, fp)
    with open(os.path.join(split_dir, "val_idx.pkl"), "wb") as fp:
        pickle.dump(val_idx, fp)
    with open(os.path.join(split_dir, "test_idx.pkl"), "wb") as fp:
        pickle.dump(test_idx, fp)


def load_split_indices(split_dir):
    with open(os.path.join(split_dir, "train_idx.pkl"), "rb") as fp:
        train_idx = pickle.load(fp)
    with open(os.path.join(split_dir, "val_idx.pkl"), "rb") as fp:
        val_idx = pickle.load(fp)
    with open(os.path.join(split_dir, "test_idx.pkl"), "rb") as fp:
        test_idx = pickle.load(fp)

    return train_idx, val_idx, test_idx


def save_train_idx_with_syndrome(split_dir, train_idx, data_pd):
    train_idx_with_syndrome = []

    for i in train_idx:
        syndrome_i = data_pd.iloc[i]["Syndromes"]
        if isinstance(syndrome_i, list) and len(syndrome_i) > 0:
            train_idx_with_syndrome.append(i)

    with open(os.path.join(split_dir, "train_idx_with_syndrome.pkl"), "wb") as fp:
        pickle.dump(train_idx_with_syndrome, fp)

    print(
        "Training records with Syndrome:",
        len(train_idx_with_syndrome),
        "out of",
        len(train_idx)
    )

    return train_idx_with_syndrome


def save_similar_formula_pairs(similar_pairs, data_pd, out_path):
    """
    Save formula pairs identified as similar during strict split construction.
    This file is for manual inspection of the leakage-control rule.
    """
    rows = []

    for pair in similar_pairs:
        i = pair["row_i"]
        j = pair["row_j"]

        herbs_i = list(data_pd.at[i, "Herbs"])
        herbs_j = list(data_pd.at[j, "Herbs"])

        set_i = set(herbs_i)
        set_j = set(herbs_j)

        rows.append({
            "reason": pair["reason"],
            "jaccard": pair["jaccard"],

            "row_i": i,
            "row_j": j,

            "id_i": data_pd.at[i, "ID"],
            "id_j": data_pd.at[j, "ID"],

            "title_i": data_pd.at[i, "Title"],
            "title_j": data_pd.at[j, "Title"],

            "n_herbs_i": len(set_i),
            "n_herbs_j": len(set_j),
            "n_shared_herbs": len(set_i & set_j),

            "shared_herbs": " ".join(sorted(set_i & set_j)),
            "herbs_i": " ".join(herbs_i),
            "herbs_j": " ".join(herbs_j),

            "symptoms_i": " ".join(list(data_pd.at[i, "Symptoms"])),
            "symptoms_j": " ".join(list(data_pd.at[j, "Symptoms"]))
        })

    similar_pairs_pd = pd.DataFrame(rows)

    if len(similar_pairs_pd) > 0:
        similar_pairs_pd = similar_pairs_pd.sort_values(
            by=["reason", "jaccard"],
            ascending=[True, False]
        )

    similar_pairs_pd.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(
        f"Identified similar formula pairs saved to {out_path}. "
        f"Number of pairs: {len(similar_pairs_pd)}"
    )


def save_dataloaders(split_dir, train_idx, val_idx, test_idx, data_pd, loader_seed):
    """
    Generate and save dataloaders for one specific split only.
    """
    Random.seed(loader_seed)
    np.random.seed(loader_seed)
    torch.manual_seed(loader_seed)

    train_loader = prepare_torch_data(train_idx, data_pd)
    val_loader = prepare_torch_data(val_idx, data_pd)
    test_loader = prepare_torch_data(test_idx, data_pd)

    with open(os.path.join(split_dir, "train_dataloader.pkl"), "wb") as fp:
        pickle.dump(train_loader, fp)
    with open(os.path.join(split_dir, "val_dataloader.pkl"), "wb") as fp:
        pickle.dump(val_loader, fp)
    with open(os.path.join(split_dir, "test_dataloader.pkl"), "wb") as fp:
        pickle.dump(test_loader, fp)

    print(
        f"Dataloaders saved to {split_dir}. "
        f"Training: {len(train_idx)}, Validation: {len(val_idx)}, Testing: {len(test_idx)}"
    )





########################################################################################################################
# Prepare external TCM-PD dataset for fair downstream formula-record prediction
########################################################################################################################

def parse_external_terms(value):
    """
    Parse standardized terms separated by common delimiters.
    """
    if pd.isna(value):
        return []

    terms = re.split(r"[；;,，、]+", str(value))
    terms = [t.strip() for t in terms if t.strip()]

    # Remove duplicated terms while preserving order.
    out = []
    seen = set()
    for t in terms:
        if t not in seen:
            out.append(t)
            seen.add(t)

    return out


def build_external_all_valid_dataframe():
    """
    Convert external TCM-PD Excel into a minimal all-valid formula-style dataframe.

    Cleaning rules:
        1. parse standardized symptom/herb terms;
        2. remove duplicated terms while preserving order;
        3. require all terms to be in standardized vocabularies;
        4. require non-empty Symptoms and Herbs;
        5. drop records exceeding model length limits.

    Required output columns:
        Train_test_split
        Symptoms
        Herbs
    """
    external_pd = pd.read_excel(args.external_xlsx)

    rows = []
    dropped_too_long_rows = []
    herb_set = set(herb_list)
    symptom_set = set(symptom_list)

    for i, row in external_pd.iterrows():
        split_label = str(row[args.external_split_col]).strip()

        symptoms = parse_external_terms(row[args.external_symptom_col])
        herbs = parse_external_terms(row[args.external_herb_col])

        missing_symptoms = [s for s in symptoms if s not in symptom_set]
        missing_herbs = [h for h in herbs if h not in herb_set]

        if len(missing_symptoms) > 0 or len(missing_herbs) > 0:
            raise ValueError(
                f"External row {i} contains terms outside standard vocab.\n"
                f"Missing symptoms: {missing_symptoms}\n"
                f"Missing herbs: {missing_herbs}"
            )

        if split_label not in ["train", "test"]:
            raise ValueError(
                f"External row {i} has invalid split label: {split_label}. "
                f"Expected 'train' or 'test'."
            )

        if len(symptoms) < 1 or len(herbs) < 1:
            raise ValueError(
                f"External row {i} has empty Symptoms or Herbs after parsing. "
                f"Symptoms={symptoms}, Herbs={herbs}"
            )

        if len(symptoms) >= MAX_SYM_LEN or len(herbs) >= MAX_HERB_LEN:
            dropped_too_long_rows.append({
                "external_row": i,
                "split": split_label,
                "n_symptoms": len(symptoms),
                "n_herbs": len(herbs),
                "Symptoms": "；".join(symptoms),
                "Herbs": "；".join(herbs)
            })
            continue

        rows.append({
            "Train_test_split": split_label,
            "Symptoms": symptoms,
            "Herbs": herbs
        })

    external_all_valid_pd = pd.DataFrame(rows).reset_index(drop=True)

    print("External all-valid formula records after length filtering:")
    print(external_all_valid_pd["Train_test_split"].value_counts())

    print("Dropped external records exceeding model length limit:", len(dropped_too_long_rows))

    if len(dropped_too_long_rows) > 0:
        os.makedirs(args.external_out_dir, exist_ok=True)
        pd.DataFrame(dropped_too_long_rows).to_csv(
            os.path.join(args.external_out_dir, "external_dropped_too_long_records.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    return external_all_valid_pd


def save_external_all_valid_dataset():
    """
    Save cleaned all-valid external TCM-PD dataframe.
    Similarity-based test filtering is handled in formula_prediction_external.py.
    """
    os.makedirs(args.external_out_dir, exist_ok=True)

    external_all_valid_pd = build_external_all_valid_dataframe()

    all_valid_path = os.path.join(
        args.external_out_dir,
        "TCM_PD_formula_data_external_all_valid.pkl"
    )

    external_all_valid_pd.to_pickle(all_valid_path)

    split_counts = external_all_valid_pd["Train_test_split"].value_counts().to_dict()

    metadata_path = os.path.join(
        args.external_out_dir,
        "external_all_valid_metadata.txt"
    )

    with open(metadata_path, "w", encoding="utf-8") as fp:
        fp.write("data_type: external_TCM_PD_all_valid\n")
        fp.write(f"external_xlsx: {args.external_xlsx}\n")
        fp.write(f"external_split_col: {args.external_split_col}\n")
        fp.write(f"external_symptom_col: {args.external_symptom_col}\n")
        fp.write(f"external_herb_col: {args.external_herb_col}\n")
        fp.write(f"max_symptoms_allowed: {MAX_SYM_LEN - 1}\n")
        fp.write(f"max_herbs_allowed: {MAX_HERB_LEN - 1}\n")
        fp.write(f"n_all_valid_records: {len(external_all_valid_pd)}\n")
        fp.write(f"split_counts: {split_counts}\n")
        fp.write("note: similarity filtering is performed in formula_prediction_external.py\n")

    print(
        f"External all-valid dataframe saved to {all_valid_path}. "
        f"Total: {len(external_all_valid_pd)}, split counts: {split_counts}"
    )



########################################################################################################################
# Run data preparation
########################################################################################################################

if args.make_external_pd_all_valid:
    save_external_all_valid_dataset()
    sys.exit(0)

# TCM formula records used for model training and evaluation
formula_data_pd = pd.read_pickle(args.formula_data)
print("Data columns: ", formula_data_pd.columns)
if args.show_data:
    print(formula_data_pd[['Title', 'Indication text', 'Symptoms', 'Herbs']])

# Retain records meeting the model input-length and minimum-content criteria.
data_idx_subset = get_valid_indices(formula_data_pd)
print("Valid formula records:", len(data_idx_subset), "out of", len(formula_data_pd))

os.makedirs(out_dir, exist_ok=True)


def split_files_exist(split_dir):
    required_files = [
        os.path.join(split_dir, "train_idx.pkl"),
        os.path.join(split_dir, "val_idx.pkl"),
        os.path.join(split_dir, "test_idx.pkl")
    ]
    return all(os.path.exists(path) for path in required_files)


def generate_one_split(split_dir, split_seed):
    train_idx, val_idx, test_idx, similar_pairs = split_indices_622(
        formula_data_pd,
        data_idx_subset,
        split_seed=split_seed,
        same_title_jaccard_threshold=args.same_title_jaccard,
        cross_title_jaccard_threshold=args.cross_title_jaccard
    )

    save_split_indices(split_dir, train_idx, val_idx, test_idx)
    save_train_idx_with_syndrome(split_dir, train_idx, formula_data_pd)

    if args.same_title_jaccard <= 1 or args.cross_title_jaccard <= 1:
        save_similar_formula_pairs(
            similar_pairs,
            formula_data_pd,
            os.path.join(split_dir, "identified_similar_formulas.csv")
        )

    with open(os.path.join(split_dir, "split_metadata.txt"), "w", encoding="utf-8") as fp:
        fp.write(f"seed: {split_seed}\n")
        fp.write(f"same_title_jaccard: {args.same_title_jaccard}\n")
        fp.write(f"cross_title_jaccard: {args.cross_title_jaccard}\n")
        fp.write(f"train_size: {len(train_idx)}\n")
        fp.write(f"val_size: {len(val_idx)}\n")
        fp.write(f"test_size: {len(test_idx)}\n")

    print(
        f"Data split saved to {split_dir}. "
        f"Training: {len(train_idx)}, Validation: {len(val_idx)}, Testing: {len(test_idx)}"
    )

    return train_idx, val_idx, test_idx


def get_split_for_dataloader(split_dir, split_seed):
    if split_files_exist(split_dir) and not args.regenerate_split:
        print(
            f"Using existing split from {split_dir}. "
            "Use --regenerate-split to generate a new split."
        )
        return load_split_indices(split_dir)

    return generate_one_split(split_dir, split_seed)


def prepare_one_dataset(split_dir, split_seed, loader_seed):
    if args.make_dataloader:
        train_idx, val_idx, test_idx = get_split_for_dataloader(split_dir, split_seed)
        save_dataloaders(
            split_dir=split_dir,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            data_pd=formula_data_pd,
            loader_seed=loader_seed
        )
    else:
        train_idx, val_idx, test_idx = generate_one_split(split_dir, split_seed)

    return train_idx, val_idx, test_idx


if args.num_repeats == 1:
    train_idx, val_idx, test_idx = prepare_one_dataset(
        split_dir=out_dir,
        split_seed=seed,
        loader_seed=seed
    )

    if args.make_null_dataloader:
        herb_shuffled_dir = os.path.join(out_dir, "herb_shuffled")
        os.makedirs(herb_shuffled_dir, exist_ok=True)

        formula_data_null_pd = make_splitwise_shuffled_formula_dataframe(
            formula_data_pd,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=seed
        )

        formula_data_null_pd.to_pickle(
            os.path.join(herb_shuffled_dir, "TCM_formula_data_shuffled_pairing.pkl")
        )
        save_split_indices(herb_shuffled_dir, train_idx, val_idx, test_idx)
        save_train_idx_with_syndrome(herb_shuffled_dir, train_idx, formula_data_null_pd)
        save_dataloaders(
            split_dir=herb_shuffled_dir,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            data_pd=formula_data_null_pd,
            loader_seed=seed
        )

else:
    repeated_split_dir = os.path.join(out_dir, "repeated_splits")
    os.makedirs(repeated_split_dir, exist_ok=True)

    for repeat_id in range(1, args.num_repeats + 1):
        repeat_seed = seed + repeat_id
        repeat_out_dir = os.path.join(repeated_split_dir, f"repeat_{repeat_id:02d}")

        prepare_one_dataset(
            split_dir=repeat_out_dir,
            split_seed=repeat_seed,
            loader_seed=seed + 1000 + repeat_id
        )
