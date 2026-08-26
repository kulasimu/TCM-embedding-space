import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
import torch
import sys
import subprocess
from pathlib import Path
import pandas as pd
import random
from bin.TCM_embedding_generator import TCMEmbeddingGenerator




########################################################################################################################
# Step 0: Prepare the data for model training, validation and testing (model input and output)
########################################################################################################################

#### Step 0.1: Prepare data split and dataloaders for main model training (internal data).

# Main split
subprocess.run(
    [
        sys.executable,
        "bin/data_generator.py",
        "data/training",
        "--formula-data", "data/TCM_formulas/TCM_formula_data_example_2000.pkl",
        "--make-dataloader",
    ],
    check=True,
)


# Repeat splits for robustness evaluation
subprocess.run(
    [
        sys.executable,
        "bin/data_generator.py",
        "data/training",
        "--formula-data",
        "data/TCM_formulas/TCM_formula_data_example_2000.pkl",
        "--num-repeats", "10",
    ],
    check=True,
)

#### Step 0.2: Prepare data for external benchmark (TCM-PD)

subprocess.run(
    [
        sys.executable,
        "bin/data_generator.py",
        "--make-external-pd-all-valid",
        "--external-xlsx",
        "data/TCM_PD_external/TCM_PD_combined_30527.xlsx",
        "--external-out-dir",
        "data/TCM_PD_external",
    ],
    check=True,
)



########################################################################################################################
# Step 1: Train the main model for learning TCM entity embeddings.
########################################################################################################################
#### Step 1.1: Train the main model for learning TCM entity embeddings.
# Before training, make sure the following preprocessed data files are placed in the directory:
# train_dataloader.pkl, val_dataloader.pkl, and test_dataloader.pkl.
# Place them in the "data/training/" folder.
# The trained model is saved in 'core/trained_model'

data_dir = 'data/training'
save_dir = 'core/trained_model'
show_loss = 'N'
beta = 1.5
train_mode = "full"
subprocess.run(
    [
    sys.executable,
    "bin/train_main_model.py",
    data_dir,
    save_dir,
    show_loss,
    str(beta),
    train_mode
])
# Select a well-trained model at certain epoch and save it to "core/trained_model" for further analysis


#### Step 1.2: Repeated model training

data_dir = "data/training"
repeat_save_base_dir = "core/trained_model_repeated"
show_loss = "N"
seed = 42
beta = 1.5
train_mode = "full"
num_repeats = 10


subprocess.run(
    [
        sys.executable,
        "bin/data_generator.py",
        "data/training",
        "--formula-data",
        "data/TCM_formulas/TCM_formula_data_example_2000.pkl",
        "--num-repeats", "10",
        "--make-dataloader",
    ],
    check=True,
)

for repeat_id in range(1, num_repeats + 1):
    repeat_data_dir = os.path.join(data_dir, "repeated_splits", f"repeat_{repeat_id:02d}")
    repeat_save_dir = os.path.join(repeat_save_base_dir, f"repeat_{repeat_id:02d}")

    subprocess.run([
        sys.executable,
        "bin/train_main_model.py",
        repeat_data_dir,
        repeat_save_dir,
        show_loss,
        str(beta),
        train_mode
    ])

    for fname in ["train_dataloader.pkl", "val_dataloader.pkl", "test_dataloader.pkl"]:
        fpath = os.path.join(repeat_data_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)




########################################################################################################################
# Step 2: Generate embeddings using trained models and 2 co-occurrence based models
########################################################################################################################

#### Step 2.1: Generate embeddings using original model and 10 repeat models

#### Main model
subprocess.run(
    [
        sys.executable,
        "bin/TCM_embedding_generator.py",
        "--model-dir",
        "core/trained_model/model_epoch_60.pkl",
        "--tasks", "formula", "individual", "external-tcm-pd", "general-clinical", "attention"
    ],
    check=True,
)

#### Repeat model
repeat_model_root = Path("core/trained_model_repeated")
for repeat_id in range(1, 11):
    repeat_name = f"repeat_{repeat_id:02d}"
    model_dir = repeat_model_root / repeat_name
    model_files = sorted(model_dir.glob("*.pkl"))

    if len(model_files) != 1:
        raise RuntimeError(f"{model_dir} should contain exactly one model checkpoint,found {len(model_files)}: {[p.name for p in model_files]}")
    model_path = model_files[0]
    subprocess.run(
        [
            sys.executable,
            "bin/TCM_embedding_generator.py",
            "--model-dir", str(model_path),
            "--formula-data", "data/TCM_formulas/TCM_formula_data_example_2000.pkl",
            "--TCM-out-dir", f"results/embeddings/TCM_embeddings_repeated/{repeat_name}",
        ],
        check=True,
    )


#### Step 2.2 (for demo only): Generate embeddings for arbitrary herb combinations or symptom patterns using original model

# Set up the generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = TCMEmbeddingGenerator(model_dir='core/trained_model/model_epoch_60.pkl',device=device)  # Load the selected model in last step

# Select one or more of the following symptoms or herbs (ranked by frequency of occurrence in TCM formula records) as input.
# print(herbs_list)
# print(symptom_list)

#### Decode herbs to treat the symptom pattern  (from classical TCM formula Si-Jun-Zi-Tang 四君子汤)
example_symptom_pattern = ['消瘦', '食少', '气短', '大便溏薄', '神疲']
symptom_pattern_embeddings = generator.generate_symptom_embedding([example_symptom_pattern])  # To support batch processing, always wrap the symptom sequence in a list of lists, even when submitting only one formula
symptom2herb_decodings = generator.herb_decoding(example_symptom_pattern, printout=True)
print('-----------------------\n')

#### Decode the symptoms that can be treated by the formula
example_formula = ["射干", "细辛", "山药", "枳实", "橘皮", "藿香"]
formula_embedding = generator.generate_herb_embedding([example_formula])  # To support batch processing, always wrap the herb sequence in a list of lists, e.g., [["麻黄", "甘草", "杏仁", "石膏"]], even when submitting only one formula
print(formula_embedding.shape)
formula2symptom_decodings = generator.symptom_decoding(example_formula, printout=True)
print('-----------------------\n')




#### Step 2.3: Prepare co-occurrence-based baseline embeddings

subprocess.run(
    [
        sys.executable,
        "bin/prepare_embedding_baselines.py",
        "--cooc-weighting", "ppmi"
    ],
    check=True,
)



########################################################################################################################
# Step 3: Analysing the embeddings geometry
########################################################################################################################
#### Step 3.0 (optional) predictive effect of attentions

subprocess.run([
    sys.executable,
    "bin/attention_perturbation_analysis.py",

    "--random-repeats", "10",
    "--bootstrap-repeats", "1000",
    "--seed", "42",
])


#### Step 3.1: Identify principal directions in the embedding space using PCA

source_name = "original"
embedding_dir = "results/embeddings/TCM_embeddings"
pca_embedding_type = "average"

out_dir = (f"results/TCM_embedding_analysis/{source_name}({pca_embedding_type})")
pca_model = (Path(out_dir) / "pca"/ "TCM_embedding_pca_model.pkl")


subprocess.run(
    [
        sys.executable,
        "bin/TCM_embedding_pca_projection.py",
        "--embedding-dir", embedding_dir,
        "--out-dir", out_dir,
        "--pca-components", "6",
        "--pca-embedding-type", pca_embedding_type,

        # Uncomment to reuse the fitted PCA.
        # "--pca-model", str(pca_model),
    ],
    check=True,
)



#### Step 3.2: The alignment between TCM embeddings and TCM principles (TCM syndromes, herb natures etc.)

source_name = "original"
pca_embedding_type = "average"
analysis_dir = (f"results/TCM_embedding_analysis/{source_name}({pca_embedding_type})")

subprocess.run(
    [
        sys.executable,
        "bin/TCM_principle_alignment_analysis.py",
        "--analysis-dir", analysis_dir,
        "--plot", "syndrome",
        "--embedding-side", "both",
    ],
    check=True,
)



########################################################################################################################
#### Step 4: Evaluate the robustness of the embedding structure on 10 repeated models
#### Importantly, this analysis should be performed on full dataset
########################################################################################################################
#### Global geometry stability and local variance

subprocess.run(
    [
        sys.executable,
        "bin/embedding_robustness_evaluation.py",

        "--mode", "all",

        "--repeat-model-root",
        "core/trained_model_repeated",

        "--primary-pca-analysis-dir",
        "results/TCM_embedding_analysis/original(average)",
    ],
    check=True,
)


#### Principal component alignment

subprocess.run(
    [
        sys.executable,
        "bin/TCM_embedding_pca_repeat_alignment.py",
        "--main-pca","results/TCM_embedding_analysis/original(average)/pca/TCM_embedding_pca_model.pkl",
        "--main-embedding-dir", "results/embeddings/TCM_embeddings",
        "--repeat-embedding-root", "results/embeddings/TCM_embeddings_repeated",
        "--out-dir", "results/TCM_embedding_pca_repeat_alignment",
        "--n-components", "6",
    ],
    check=True,
)



########################################################################################################################
# Step 5: Predictive / retrieval performance on testing / external benchmark / clinical datasets
########################################################################################################################

#### Step 5.1: Internal formula prediction performance on our own test set

subprocess.run(
    [
        sys.executable,
        "bin/formula_prediction_internal.py",

        # Demo data and frozen TCM-ES embeddings
        "--formula-data", "data/TCM_formulas/TCM_formula_data_example_2000.pkl",
        "--embedding-dir", "results/embeddings/TCM_embeddings",

        # Internal train / validation / test split
        "--train-idx", "data/training/train_idx.pkl",
        "--val-idx", "data/training/val_idx.pkl",
        "--test-idx", "data/training/test_idx.pkl",

        "--out-dir", "results/TCM_formula_prediction_internal",

        # MLP architecture
        "--hidden1", "1024",
        "--hidden2", "512",
        "--dropout", "0.2",

        "--seed", "42",
    ],
    check=True,
)


#### Step 5.3: Evaluating formula prediction performance on benchmark TCM-PD dataset and comparing with published models

subprocess.run(
    [
        sys.executable,
        "bin/formula_prediction_external.py",

        # External TCM-PD data and frozen symptom-pattern embeddings
        "--external-formula-data", "data/TCM_PD_external/TCM_PD_formula_data_external_all_valid.pkl",
        "--external-embedding-dir", "results/embeddings/TCM_PD_external/original",

        # Cached evaluation split
        "--external-split-dir", "results/TCM_formula_prediction_external/data_split",
        "--out-dir", "results/TCM_formula_prediction_external",

        # Internal reference corpus for overlap filtering
        "--reference-formula-data", "data/TCM_formulas/TCM_formula_data_example_2000.pkl",

        # Evaluation settings
        "--external-val-ratio", "0.1",
        "--external-symptom-jaccard", "0.8",
        "--external-herb-jaccard", "0.8",

        "--k-values", "5,10,20",
        "--ridge-lambda", "10",

        # MLP architecture
        "--hidden1", "1024",
        "--hidden2", "512",
        "--dropout", "0.2",

        "--seed", "42",
    ],
    check=True,
)


#### Step 5.3: Embedding distance based analysis (clinical cases)
# (within case alleviated-unalleviated comparison, formula-symptom-pattern retrieval)

subprocess.run(
    [
        sys.executable,
        "bin/clinical_distance_based_analysis.py",
        "--clinical-data", "data/general_TCM_clinical_cases/TCM_general_cases_data_example_500_model_eligible.pkl",

        "--symptom-list",
        "core/standard_TCM_entities/symptom_list.pkl",

        "--tcm-clinical-emb-dir",
        "results/embeddings/general_TCM_clinical_cases/original",

        "--tcm-emb-dir",
        "results/embeddings/TCM_embeddings",

        "--cooc-svd-emb-dir",
        "results/embedding_baseline_comparison/baseline_ppmi_svd",

        "--cooc-graph-emb-dir",
        "results/embedding_baseline_comparison/baseline_ppmi_graph",

        "--out-dir",
        "results/clinical_distance_based_analysis",

        "--set-metric",
        "euclidean",

        "--n-candidates",
        "50",

        "--n-repeats",
        "100",

        "--seed",
        "42",
    ],
    check=True,
)


#### Step 5.4: Clinical condition-improvement on COVID-19 data
## Generate embeddings for COVID-19 conditions/symptom patterns and prescribed formulas

subprocess.run(
    [
        sys.executable,
        "bin/COVID_19_embedding_generator.py",
        "--data-dir", "data/COVID_19_data",
        "--covid-data-file", "COVID_data_example_200.pkl",
        "--clinical-out", "data/COVID_19_data/COVID_19_data_for_embeddings_example_200.pkl",
        "--out-dir", "results/embeddings/COVID_19_cases/example_200",
    ],
    check=True,
)



## Evaluate the association between condition improvement and embedding proximity

subprocess.run(
    [
        sys.executable,
        "bin/COVID_condition_improvement_analysis.py",

        "--clinical-data", "data/COVID_19_data/COVID_19_data_for_embeddings_example_200.pkl",
        "--raw-covid-csv", "data/COVID_19_data/COVID_data_example_200.csv",
        "--covid-symptom-list", "data/COVID_19_data/COVID_symptom_list.pkl",
        "--tcm-covid-emb-dir", "results/embeddings/COVID_19_cases/example_200",
        "--include-formulas", "方D,方4,方B,方A",
        "--min-followup-days", "0",
        "--max-followup-days", "100",
        "--min-initial-symptoms", "1",
        "--min-initial-total-score", "0",
        "--max-initial-total-score", "100",
        "--improvement-outcome", "total_improvement_score",
        "--tcm-distance-metric", "euclidean",
        "--adjusted-min-n", "20",
        "--sensitivity-min-n", "10",
        "--adjusted-n-bins", "10",
        "--n-bootstrap", "1000",
        "--seed", "42",
        "--run-sensitivity-analyses",

        "--out-dir",
        "results/COVID_19_condition_improvement",


    ],
    check=True,
)



########################################################################################################################
# Step 6: Project biomedical entities (ingredient compounds of herbs, targets, diseases, etc.) into the TCM-ES through
# their associations with TCM herbs and symptoms
########################################################################################################################

######## Step 6.1 Train target alignment model for main TCM-ES and generate target embeddings

## Main model
subprocess.run([
    sys.executable,
    "bin/train_protein_embedding.py",
    "--embedding-dir","results/embeddings/TCM_embeddings",
    "--output-dir","results/protein_alignment_embeddings/main",
], check=True)

## 10 repeat models
for repeat_id in range(1, 11):
    repeat_name = f"repeat_{repeat_id:02d}"

    subprocess.run([
        sys.executable,
        "bin/train_protein_embedding.py",

        "--embedding-dir",
        f"results/embeddings/TCM_embeddings_repeated/{repeat_name}/",

        "--output-dir",
        f"results/protein_alignment_embeddings/{repeat_name}/",
    ])



#### Step 6.2: Train herbal compound alignment model for main TCM-ES and generate target embeddings

## Preprocess SMILES through pretrained Mol2Vec
subprocess.run(
    [
        sys.executable,
        "bin/prepare_compound_mol2vec_features.py",

        "--compound-list",
        "data/herb_compounds/compound_list.pkl",

        "--compound-table",
        "data/herb_compounds/compound_SMILES.xlsx",

        "--sheet-name",
        "filtered",

        "--output-file",
        "data/herb_compounds/compound_mol2vec_vectors.pkl",
    ],
    check=True,
)

## Train compound encoder
subprocess.run(
    [
        sys.executable,
        "bin/train_compound_embedding_mol2vec.py",
        "--embedding-dir", "results/embeddings/TCM_embeddings",
        "--output-dir", "results/compound_alignment_embeddings/main",
    ],
    check=True,
)





#### Step 6.3: Examine the concordance between embedding and PPI

subprocess.run(
    [
        sys.executable,
        "bin/PPI_concordance_analysis.py",

        "--seed", "42",
        "--consensus-method", "median",

        "--output-dir",
        "results/PPI_concordance_analysis",
    ],
    check=True,
)







