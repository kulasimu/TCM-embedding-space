# TCM Embedding Space (TCM-ES)

**Code and reproducibility resources for:**  
**_An Interpretable AI Framework Quantifying TCM Principles towards Integration with Modern Biomedicine_**

[arXiv:2507.11176](https://arxiv.org/abs/2507.11176) · [Online TCM-ES toolbox](http://47.239.86.39/) · [Pretrained checkpoints](https://github.com/kulasimu/TCM-embedding-space/releases)

## Overview

Traditional Chinese Medicine (TCM) uses a holistic diagnostic–therapeutic framework in which complex symptom patterns are summarized into latent clinical patterns and subsequently translated into individualized multi-herb therapies. The qualitative nature of these principles, however, makes them difficult to study quantitatively and to relate systematically to modern biomedical information.

This repository implements the **TCM Embedding Space (TCM-ES)** framework described in the accompanying paper. The core model is a Transformer-based autoencoder trained on matched symptom patterns and herbal formulas from ancient and classical TCM records. It combines bottleneck encoding, self-/cross-attention, reconstruction, and contrastive learning to map symptom patterns and formulas into a shared **256-dimensional embedding space**.

The core TCM-ES is trained only from matched symptom-pattern–formula records. Syndrome labels, herb-property annotations, clinical outcomes, and biomedical information are not used as training targets for the core model. These independent annotations and datasets are instead used for downstream interpretation, validation, and biomedical mapping.

After the core TCM-ES is established, the repository supports:

- embedding standardized symptoms, herbs, complete symptom patterns, and formulas;
- PCA-based characterization and alignment of independently trained TCM-ES models;
- evaluation of conventional TCM principle-associated organization;
- attention extraction and attention-guided token-removal perturbation;
- internal and external formula prediction;
- retrospective clinical distance-based analyses, including COVID-19 analyses;
- mapping symptom-defined diseases into the fixed TCM-ES;
- mapping target proteins through herb–target correspondence;
- mapping herbal compounds through Mol2Vec structural representations and herb–compound correspondence;
- evaluation of training-seed reproducibility and PPI-network concordance.

The framework is intended for quantitative representation, association analysis, and hypothesis generation. Embedding proximity or model attention should not by itself be interpreted as evidence of causality, therapeutic efficacy, or a validated biological mechanism.

---

## Repository structure

```text
TCM-embedding-space/
├── bin/
│   ├── attention_perturbation_analysis.py
│   ├── clinical_distance_based_analysis.py
│   ├── COVID_19_embedding_generator.py
│   ├── COVID_condition_improvement_analysis.py
│   ├── data_generator.py
│   ├── disease_embedding_generator.py
│   ├── embedding_robustness_evaluation.py
│   ├── formula_prediction_external.py
│   ├── formula_prediction_internal.py
│   ├── PPI_concordance_analysis.py
│   ├── prepare_embedding_baselines.py
│   ├── TCM_embedding_generator.py
│   ├── TCM_embedding_pca_projection.py
│   ├── TCM_embedding_pca_repeat_alignment.py
│   ├── TCM_principle_alignment_analysis.py
│   ├── train_compound_embedding.py
│   ├── train_main_model.py
│   └── train_protein_embedding.py
│
├── core/
│   ├── standard_TCM_entities/
│   ├── trained_model/                  # place the main checkpoint here
│   └── trained_model_repeated/         # place the 10 repeat checkpoints here
│
├── data/
│   ├── COVID_19_data/
│   ├── disease/
│   ├── general_TCM_clinical_cases/
│   ├── herb_compounds/
│   ├── herb_targets/
│   ├── PPI/
│   ├── TCM_formulas/
│   ├── TCM_PD_external/
│   └── training/
│
├── results/
├── demo.py
└── main.py
```

### Main scripts

| Script | Main purpose |
|---|---|
| `bin/data_generator.py` | Prepare train/validation/test splits, model dataloaders, repeated splits, non-leakage splits, and external TCM-PD inputs |
| `bin/train_main_model.py` | Train the core Transformer-based TCM-ES model |
| `bin/TCM_embedding_generator.py` | Generate symptom, herb, symptom-pattern, formula, clinical-case, and held-out attention outputs |
| `bin/TCM_embedding_pca_projection.py` | Fit/apply PCA to TCM-ES embeddings |
| `bin/TCM_embedding_pca_repeat_alignment.py` | Match independently fitted PCA axes across repeat models |
| `bin/TCM_principle_alignment_analysis.py` | Analyze organization associated with conventional TCM principles |
| `bin/embedding_robustness_evaluation.py` | Evaluate global and local reproducibility across independent training runs |
| `bin/attention_perturbation_analysis.py` | Test predictive relevance of cross-attention rankings by token removal |
| `bin/formula_prediction_internal.py` | Internal formula-prediction analyses |
| `bin/formula_prediction_external.py` | External TCM-PD formula-prediction analyses |
| `bin/clinical_distance_based_analysis.py` | General TCM clinical distance and retrieval analyses |
| `bin/COVID_19_embedding_generator.py` | Generate COVID-19 clinical embeddings using the study-specific symptom representation |
| `bin/COVID_condition_improvement_analysis.py` | Analyze condition–formula relationships and subsequent improvement in COVID-19 cases |
| `bin/disease_embedding_generator.py` | Map symptom-defined diseases into the fixed TCM-ES |
| `bin/train_protein_embedding.py` | Train the herb–target alignment autoencoder and generate target-protein embeddings |
| `bin/train_compound_embedding.py` | Train the herb–compound alignment autoencoder and generate herbal-compound embeddings |
| `bin/PPI_concordance_analysis.py` | Evaluate concordance between integrated TCM-ES geometry and the human PPI network |
| `bin/prepare_embedding_baselines.py` | Prepare co-occurrence-based comparison embeddings |

`main.py` provides the ordered analysis workflow used for the repository. `demo.py` provides a lighter-weight entry point for demonstration with the public example data.

---

## Installation

### 1. Obtain the repository

Either download the repository as a ZIP file from GitHub, or clone it:

```bash
git clone https://github.com/kulasimu/TCM-embedding-space.git
cd TCM-embedding-space
```

### 2. Create a Python environment

A separate environment is recommended:

```bash
conda create -n tcmes python=3.10 -y
conda activate tcmes
```

Install PyTorch using the build appropriate for your operating system and CUDA version, then install the scientific Python dependencies used by the analysis scripts. A typical environment includes:

```bash
pip install numpy pandas scipy scikit-learn matplotlib statsmodels \
            openpyxl tqdm networkx torchmetrics fastai
```

For herbal-compound structural processing/alignment:

```bash
pip install rdkit deepchem
```

A CUDA-enabled PyTorch installation is recommended for training the core model and the alignment models, although many embedding-generation and analysis steps can also run on CPU.

Some optional preprocessing workflows may require additional packages. If an individual script reports a missing dependency, install the package required by that module.

---

## Download the pretrained checkpoints

The large pretrained model files are distributed through **GitHub Releases** rather than stored directly in the Git repository.

Go to:

**https://github.com/kulasimu/TCM-embedding-space/releases**

The checkpoint release contains:

- **1 primary TCM-ES checkpoint**;
- **10 independently trained repeat-model checkpoints** used for training-seed robustness analyses.

### Required directory layout

After downloading, place the primary model at:

```text
core/trained_model/model_epoch_60.pkl
```

Place the ten repeat checkpoints in the corresponding repeat directories:

```text
core/
├── trained_model/
│   └── model_epoch_60.pkl
│
└── trained_model_repeated/
    ├── repeat_01/
    │   └── model_epoch_*.pkl
    ├── repeat_02/
    │   └── model_epoch_*.pkl
    ├── repeat_03/
    │   └── model_epoch_*.pkl
    ├── repeat_04/
    │   └── model_epoch_*.pkl
    ├── repeat_05/
    │   └── model_epoch_*.pkl
    ├── repeat_06/
    │   └── model_epoch_*.pkl
    ├── repeat_07/
    │   └── model_epoch_*.pkl
    ├── repeat_08/
    │   └── model_epoch_*.pkl
    ├── repeat_09/
    │   └── model_epoch_*.pkl
    └── repeat_10/
        └── model_epoch_*.pkl
```

The optimal epoch is not necessarily the same for all repeat models. Each `repeat_XX` directory should contain the corresponding optimal checkpoint downloaded from the Release.

The primary model checkpoint is loaded by default by several scripts from:

```text
core/trained_model/model_epoch_60.pkl
```

The repeat-model workflows scan the corresponding `repeat_XX` directories for their model checkpoint.

---

## Quick start

### 1. Run the lightweight demonstration

After placing the primary checkpoint in `core/trained_model/`, run:

```bash
python demo.py
```

The demonstration uses the public example datasets included in the repository and is intended to show the data and model workflow without requiring the full manuscript-scale analyses.

### 2. Generate core TCM-ES embeddings

For the public example formula records:

```bash
python bin/TCM_embedding_generator.py \
    --model-dir core/trained_model/model_epoch_60.pkl \
    --formula-data data/TCM_formulas/TCM_formula_data_example_2000.pkl \
    --tasks formula individual \
    --TCM-out-dir results/embeddings/TCM_embeddings
```

This generates, among other outputs:

```text
results/embeddings/TCM_embeddings/
├── individual_symptom_embeddings.pkl
├── individual_herb_embeddings.pkl
├── selected_formula_symptom_embeddings.pkl
├── selected_formula_herb_embeddings.pkl
└── selected_formula_ids(line_number).pkl
```

### 3. Generate held-out cross-attention for perturbation analysis

The attention analysis uses cross-attention extracted from matched held-out symptom-pattern–formula records.

```bash
python bin/TCM_embedding_generator.py \
    --model-dir core/trained_model/model_epoch_60.pkl \
    --formula-data data/TCM_formulas/TCM_formula_data.pkl \
    --tasks attention \
    --test-idx data/training/test_idx.pkl \
    --attention-out-dir results/attention/original/test \
    --batch-size 256
```

Then run:

```bash
python bin/attention_perturbation_analysis.py \
    --attention-file results/attention/original/test/cross_attention_test_all.pkl \
    --model-file core/trained_model/model_epoch_60.pkl \
    --test-idx data/training/test_idx.pkl \
    --mask-counts 1,2,3 \
    --random-repeats 10 \
    --bootstrap-repeats 1000 \
    --seed 42 \
    --out-dir results/attention/original/test/perturbation_analysis
```

`--formula-data` and `--test-idx` must refer to the same formula-record row coordinate system.

### 4. Run the complete analysis workflow

`main.py` contains the ordered calls for the manuscript analysis pipeline:

```bash
python main.py
```

The full workflow is computationally more demanding and assumes that the required datasets and all pretrained/repeat checkpoints have been placed in the expected directories. Individual modules can also be run separately; use:

```bash
python bin/<script_name>.py --help
```

to inspect the available command-line arguments.

---

## Training the core TCM-ES from scratch

The released checkpoints are recommended for reproducing the downstream analyses. To retrain the model, first prepare the split files and dataloaders.

### Primary record-level split

```bash
python bin/data_generator.py --make-main-split
python bin/data_generator.py --make-main-dataloader
```

The primary model uses a 60%/20%/20% train/validation/test split of the model-eligible historical formula records.

Additional data-generation modes are available for repeated independent splits, the stricter non-leakage split, shuffled-pairing null data, and external TCM-PD preprocessing. Inspect all options with:

```bash
python bin/data_generator.py --help
```

Then inspect the model-training options with:

```bash
python bin/train_main_model.py --help
```

The primary released model corresponds to the selected epoch-60 checkpoint used for the main TCM-ES analyses.

---

## Repeated-training models

Ten independently trained TCM-ES models are provided for reproducibility analyses. The repeat models use independent train/validation/test splits and random initializations, and the checkpoint with optimal validation performance was selected separately for each run.

Their embeddings are used for global pairwise-geometry reproducibility, local nearest-neighbour reproducibility, repeat-specific PCA followed by one-to-one PC matching and sign alignment, and repeat-aware biomedical/PPI analyses.

The repeat checkpoints should be stored under:

```text
core/trained_model_repeated/repeat_01/
...
core/trained_model_repeated/repeat_10/
```

The corresponding embedding outputs are conventionally stored under:

```text
results/embeddings/TCM_embeddings_repeated/repeat_01/
...
results/embeddings/TCM_embeddings_repeated/repeat_10/
```

---

## Biomedical entity mapping

Biomedical entities are mapped **after** construction of the core TCM-ES. The pretrained core TCM-ES is kept fixed during these mapping procedures.

### Disease embeddings

Diseases are represented from their associated symptom patterns and mapped using the trained symptom encoder:

```bash
python bin/disease_embedding_generator.py \
    --mode main \
    --main-model core/trained_model/model_epoch_60.pkl \
    --output-root results/embeddings/disease_embeddings \
    --disease-table "data/disease/Disease_class_gene(science2015).xlsx" \
    --disease-mmsym data/disease/disease_mmsym.pkl \
    --mm-symptom-list data/disease/mm_symptom_list.pkl \
    --mm-symptom-semantics data/disease/mm_symptom_list_semantics.pkl \
    --max-symptoms 40 \
    --device auto
```

Repeat-model disease embeddings can be generated with `--mode repeat`.

### Target-protein alignment

Target proteins are mapped using documented herb–target associations and the fixed TCM-ES herb embeddings:

```bash
python bin/train_protein_embedding.py \
    --embedding-dir results/embeddings/TCM_embeddings \
    --output-dir results/protein_alignment_embeddings/main
```

The same procedure can be repeated using each repeat model's herb embeddings.

### Herbal-compound alignment

Herbal compounds are represented from SMILES-derived **Mol2Vec structural features** and aligned to their documented source herbs. The current Mol2Vec preprocessing produces 300-dimensional input features; the alignment script can automatically detect the feature dimension.

```bash
python bin/train_compound_embedding.py \
    --embedding-dir results/embeddings/TCM_embeddings \
    --output-dir results/compound_alignment_embeddings/main
```

If desired, the detected structural input dimensionality can be checked explicitly:

```bash
python bin/train_compound_embedding.py \
    --embedding-dir results/embeddings/TCM_embeddings \
    --output-dir results/compound_alignment_embeddings/main \
    --input-dim 300
```

The compound alignment model uses structural reconstruction together with compound–herb contrastive alignment. The 256-dimensional bottleneck representation is used as the compound embedding in TCM-ES.

---

## Major downstream analyses

The main downstream modules are available as independent scripts in `bin/`:

```text
TCM_embedding_pca_projection.py
TCM_embedding_pca_repeat_alignment.py
TCM_principle_alignment_analysis.py
embedding_robustness_evaluation.py
formula_prediction_internal.py
formula_prediction_external.py
clinical_distance_based_analysis.py
COVID_19_embedding_generator.py
COVID_condition_improvement_analysis.py
PPI_concordance_analysis.py
```

Because several analyses depend on outputs from earlier steps, the recommended execution order is the one shown in `main.py`.

For any individual analysis:

```bash
python bin/<script_name>.py --help
```

shows its required paths and optional settings.

---

## Data and reproducibility notes

The repository contains standardized TCM vocabularies, processed/public demonstration datasets, split files, analysis scripts, and reproducibility resources used by the project.

The public clinical example datasets are intended to demonstrate the analysis workflow and data format. Results obtained from the example subsets are **not expected to reproduce the exact clinical effect estimates reported for the full study cohorts**.

Raw patient-level clinical datasets are not deposited publicly because of privacy and data-use restrictions. Availability of de-identified clinical data is subject to the conditions described in the accompanying manuscript.

The `results/` directory contains generated or precomputed analysis outputs where provided. Most outputs can be regenerated from the corresponding script once the required data and checkpoints are available.

---

## Online toolbox

An interactive implementation of the TCM-ES framework is available at:

**http://47.239.86.39/**

The toolbox supports browsing and visualizing TCM-ES entities, generating customized symptom-pattern or formula representations, bidirectional symptom/formula generation, relation retrieval, attention inspection, and exploratory analysis of mapped biomedical entities.

A demonstration video for the earlier version of the system is available at:

https://www.youtube.com/watch?v=nRLaDf87m7k

---

## Citation

If you use this repository, the TCM-ES model, or the associated data resources, please cite:

```bibtex
@article{Li2025TCMES,
  title   = {An Interpretable AI Framework Quantifying TCM Principles towards Integration with Modern Biomedicine},
  author  = {Haoran Li and Xingye Cheng and Ziyang Huang and Jingyuan Luo and Qianqian Xu and Qiguang Zhao and Tianchen Guo and Yumeng Zhang and Linda Li-Dan Zhong and Zhaoxiang Bian and Leihan Tang and Aiping Lyu and Liang Tian},
  journal = {arXiv preprint arXiv:2507.11176},
  year    = {2025}
}
```

---

## Contact

For questions about the code or reproducibility workflow, please open a GitHub Issue. For data-access questions subject to institutional or data-use restrictions, please refer to the contact information provided in the accompanying paper.
