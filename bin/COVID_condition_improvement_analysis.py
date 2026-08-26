#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COVID-19 condition-improvement analysis.

Scope
-----
This script only performs the article-style COVID condition-improvement analyses:
1. Case-level association:
   condition/formula embedding absolute distance vs selected improvement outcome
2. Symptom-level within-case comparison:
   alleviated vs unalleviated initial symptoms by formula-symptom distance.
It does not run supervised prediction, AUROC/AUPRC, ROC/PR curves, or
clinical_prediction-style models.

Data assumptions
----------------
The COVID clinical embedding dataframe and the raw COVID CSV are row-aligned.

Required clinical dataframe columns:
    Cases_id
    Initial_symptoms
    Formula
    Alleviated_symptoms
    Initial_score_vector
    Followup_score_vector

Required raw CSV columns:
    初诊至复诊天数
    處方(processed)

Filtering is applied only in this analysis script, not during embedding generation.

Initial severity filtering is range-based:
    --min-initial-total-score 18 --max-initial-total-score 24
means keeping cases with 18 <= initial total symptom score <= 24.
"""

import argparse
import json
import ast
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from matplotlib import colors as mcolors

# Fixed raw CSV columns. These names are known from the COVID_data.csv file.
# Do not infer alternative names here; fail fast if the expected columns are absent.
INTERVAL_COL = "初诊至复诊天数"
FORMULA_COL = "處方(processed)"
AGE_COL = "年齡"
SEX_COL = "性別"

FORMULA_DISPLAY_LABELS = {
    "方D": "Formula D",
    "方4": "Formula 4",
    "方B": "Formula B",
    "方A": "Formula A",
}

OUTCOME_LABELS = {
    "total_improvement_score": "Total symptom-score reduction",
    "fractional_improvement": "Fractional symptom-score reduction",
    "mean_improvement_per_initial_symptom": "Mean reduction per initial symptom",
    "mean_symptom_improvement_fraction": "Mean per-symptom fractional improvement",
    "alleviated_symptom_fraction": "Fraction of initial symptoms alleviated",
}

# Distance scale retained for the COVID formula-to-symptom comparison.
FORMULA_SYMPTOM_DISTANCE_NORMALIZATION = 1.7645632746717566 * 2.0


def make_group_gradient(cmap_name, n, vmin=0.35, vmax=0.85):
    """Generate n colors sampled from a Matplotlib colormap.

    Parameters
    ----------
    cmap_name : str
        Name of matplotlib colormap, e.g. "BuPu", "YlOrRd", "YlGn".
    n : int
        Number of colors needed.
    vmin, vmax : float
        Sampling range inside the colormap. Avoids overly pale or overly dark ends.
    """
    cmap = plt.get_cmap(cmap_name)

    if n == 1:
        vals = [0.65]
    elif n == 2:
        vals = [0.45, 0.80]
    elif n == 3:
        vals = [0.40, 0.62, 0.84]
    elif n == 4:
        vals = [0.35, 0.52, 0.69, 0.86]
    else:
        vals = np.linspace(vmin, vmax, n)

    return [cmap(v) for v in vals]

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def read_csv_robust(path):
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "big5"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


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
        if not x:
            return []
        try:
            y = ast.literal_eval(x)
            if isinstance(y, (list, tuple, np.ndarray)):
                return list(y)
        except Exception:
            pass
        for sep in [";", "；", ",", "，", "/"]:
            if sep in x:
                return [s.strip() for s in x.split(sep) if s.strip()]
        return [x]

    return []


def unique_list(x):
    out = []
    seen = set()
    for item in parse_list_cell(x):
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def safe_vector(x):
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, list):
        return np.asarray(x, dtype=float)
    if isinstance(x, tuple):
        return np.asarray(list(x), dtype=float)
    if isinstance(x, str):
        try:
            return np.asarray(ast.literal_eval(x), dtype=float)
        except Exception:
            return np.asarray([], dtype=float)
    return np.asarray([], dtype=float)


def parse_formula_filter(x):
    """Convert a comma-separated formula filter into a list.

    Examples
    --------
    "all" or None  -> no formula filtering
    "方D"          -> ["方D"]
    "方D,方4"      -> ["方D", "方4"]
    """
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() in {"all", "none"}:
        return None
    return [v.strip() for v in x.split(",") if v.strip()]


def compute_improvement_outcomes(init_vec, follow_vec, n_symptoms):
    """Compute outcome variables from initial and follow-up symptom score vectors.

    All outcomes are saved to the case-level output table. The specific y-axis
    used for the correlation plot is selected by --improvement-outcome.

    Definitions
    -----------
    total_improvement_score:
        sum(initial_score - followup_score)

    fractional_improvement:
        sum(initial_score - followup_score) / sum(initial_score)
        This is weighted by baseline symptom severity.

    mean_symptom_improvement_fraction:
        mean((initial_score_i - followup_score_i) / initial_score_i)
        across symptoms with initial_score_i > 0.
        This gives each baseline-positive symptom equal weight.
    """
    init_vec = np.asarray(init_vec[:n_symptoms], dtype=float)
    follow_vec = np.asarray(follow_vec[:n_symptoms], dtype=float)

    change_vec = init_vec - follow_vec
    initial_mask = init_vec > 0

    total_improvement_score = float(np.sum(change_vec))
    initial_total_score = float(np.sum(init_vec))
    n_initial_score_symptoms = int(np.sum(initial_mask))

    if initial_total_score > 0:
        fractional_improvement = float(total_improvement_score / initial_total_score)
    else:
        fractional_improvement = np.nan

    if n_initial_score_symptoms > 0:
        initial_positive_scores = init_vec[initial_mask]
        follow_positive_scores = follow_vec[initial_mask]
        change_positive_scores = change_vec[initial_mask]

        mean_improvement_per_initial_symptom = float(
            np.mean(change_positive_scores)
        )

        mean_symptom_improvement_fraction = float(
            np.mean(change_positive_scores / initial_positive_scores)
        )

        alleviated_symptom_fraction = float(
            np.mean(follow_positive_scores < initial_positive_scores)
        )
    else:
        mean_improvement_per_initial_symptom = np.nan
        mean_symptom_improvement_fraction = np.nan
        alleviated_symptom_fraction = np.nan

    return {
        "total_improvement_score": total_improvement_score,
        "initial_total_score": initial_total_score,
        "fractional_improvement": fractional_improvement,
        "mean_improvement_per_initial_symptom": mean_improvement_per_initial_symptom,
        "mean_symptom_improvement_fraction": mean_symptom_improvement_fraction,
        "alleviated_symptom_fraction": alleviated_symptom_fraction,
        "n_initial_score_symptoms": n_initial_score_symptoms,
    }


def compute_distance(a, B, metric):
    a = np.asarray(a, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    if metric == "euclidean":
        return np.linalg.norm(B - a[None, :], axis=1)

    if metric == "cosine":
        eps = 1e-8
        a_norm = a / max(np.linalg.norm(a), eps)
        B_norm = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), eps)
        return 1.0 - (B_norm @ a_norm)

    raise ValueError(f"Unknown distance metric: {metric}")


def compute_single_distance(a, b, metric):
    return float(compute_distance(a, np.asarray(b, dtype=np.float32)[None, :], metric)[0])


def bootstrap_mean_ci(values, n_boot=1000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean_obs = float(np.mean(values))
    if n_boot <= 0:
        return mean_obs, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        boot[i] = np.mean(values[idx])

    return mean_obs, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def load_inputs(args):
    clinical_df = pd.read_pickle(args.clinical_data).reset_index(drop=True)
    raw_csv = read_csv_robust(args.raw_covid_csv).reset_index(drop=True)
    covid_symptom_list = load_pickle(args.covid_symptom_list)

    required_clinical_cols = [
        "Cases_id",
        "Initial_symptoms",
        "Formula",
        "Alleviated_symptoms",
        "Initial_score_vector",
        "Followup_score_vector",
    ]
    missing = [c for c in required_clinical_cols if c not in clinical_df.columns]
    if missing:
        raise ValueError(f"Missing required clinical-data columns: {missing}")

    required_csv_cols = [INTERVAL_COL, FORMULA_COL]
    missing = [c for c in required_csv_cols if c not in raw_csv.columns]
    if missing:
        raise ValueError(
            f"Missing required raw CSV columns: {missing}. "
            f"Available columns: {list(raw_csv.columns)}"
        )

    if len(clinical_df) != len(raw_csv):
        raise ValueError(
            "clinical-data and raw-covid-csv must be row-aligned, but row counts differ: "
            f"clinical={len(clinical_df)}, raw_csv={len(raw_csv)}"
        )

    return clinical_df, raw_csv, covid_symptom_list


def prepare_case_dataframe(clinical_df, raw_csv):
    """Attach row-aligned metadata used for filtering.

    The embedding dataframe is the analysis table. The raw CSV contributes only
    fixed metadata columns that were not stored in the embedding dataframe:
    follow-up interval and original formula label.
    """
    df = clinical_df.copy()

    # Preserve original row index so the same selected rows can be taken from
    # all embedding matrices after filtering.
    df["_original_row_idx"] = np.arange(len(df), dtype=int)

    # Row-aligned metadata from COVID_data.csv.
    df["followup_interval_days"] = pd.to_numeric(raw_csv[INTERVAL_COL], errors="coerce")
    df["formula_name_for_filter"] = raw_csv[FORMULA_COL].astype(str)

    # Optional demographic metadata used by adjusted condition-formula models.
    # These columns are required by the adjusted condition-formula analysis.
    if AGE_COL in raw_csv.columns:
        df["age"] = pd.to_numeric(raw_csv[AGE_COL], errors="coerce")
    if SEX_COL in raw_csv.columns:
        sex_text = raw_csv[SEX_COL].astype(str).str.strip()
        df["sex_male"] = np.where(
            sex_text == "男",
            1.0,
            np.where(sex_text == "女", 0.0, np.nan),
        )

    # Recompute simple counts from list columns for transparent filtering reports.
    df["n_initial_symptoms_recomputed"] = df["Initial_symptoms"].apply(
        lambda x: len(unique_list(x))
    )
    df["n_alleviated_symptoms_recomputed"] = df["Alleviated_symptoms"].apply(
        lambda x: len(unique_list(x))
    )
    df["n_unalleviated_symptoms_recomputed"] = (
        df["n_initial_symptoms_recomputed"] - df["n_alleviated_symptoms_recomputed"]
    )

    df["initial_total_score_for_filter"] = df["Initial_score_vector"].apply(
        lambda x: float(np.sum(safe_vector(x))) if len(safe_vector(x)) else np.nan
    )

    return df



def make_initial_total_score_mask(df, args):
    """Keep cases whose initial total symptom score falls inside the requested range.

    The bounds are inclusive. If a bound is omitted, that side is left open.
    Examples:
        low severity:    --max-initial-total-score 17
        medium severity: --min-initial-total-score 18 --max-initial-total-score 24
        high severity:   --min-initial-total-score 25
    """
    score = df["initial_total_score_for_filter"]
    mask = score.notna()

    if args.min_initial_total_score is not None:
        mask = mask & (score >= args.min_initial_total_score)

    if args.max_initial_total_score is not None:
        mask = mask & (score <= args.max_initial_total_score)

    return mask


def filter_cases(df, args):
    formula_filter = parse_formula_filter(args.include_formulas)

    mask_valid_interval = df["followup_interval_days"].notna()
    mask_interval = (
        mask_valid_interval
        & (df["followup_interval_days"] >= args.min_followup_days)
        & (df["followup_interval_days"] <= args.max_followup_days)
    )

    mask_initial_symptoms = df["n_initial_symptoms_recomputed"] >= args.min_initial_symptoms

    if formula_filter is None:
        mask_formula = pd.Series(True, index=df.index)
        formula_label = "all"
    else:
        mask_formula = df["formula_name_for_filter"].isin(formula_filter)
        formula_label = ",".join(formula_filter)

    # Optional baseline-severity restriction based on the initial total symptom score.
    # This is useful for sensitivity analyses that compare patients with similar
    # baseline symptom burden.
    mask_initial_total_score = make_initial_total_score_mask(df, args)

    keep_mask = mask_interval & mask_initial_symptoms & mask_formula & mask_initial_total_score
    filtered_df = df.loc[keep_mask].reset_index(drop=True)

    summary = pd.DataFrame([
        {"step": "raw_loaded_cases", "n_cases": int(len(df))},
        {"step": f"valid_{INTERVAL_COL}", "n_cases": int(mask_valid_interval.sum())},
        {
            "step": f"interval_{args.min_followup_days:g}_{args.max_followup_days:g}_days",
            "n_cases": int(mask_interval.sum()),
        },
        {
            "step": f"at_least_{args.min_initial_symptoms}_initial_symptoms",
            "n_cases": int((mask_interval & mask_initial_symptoms).sum()),
        },
        {
            "step": f"formula_filter={formula_label}",
            "n_cases": int((mask_interval & mask_initial_symptoms & mask_formula).sum()),
        },
        {
            "step": (
                "initial_total_score_range="
                f"{args.min_initial_total_score if args.min_initial_total_score is not None else '-inf'}_to_"
                f"{args.max_initial_total_score if args.max_initial_total_score is not None else 'inf'}"
            ),
            "n_cases": int((mask_interval & mask_initial_symptoms & mask_formula & mask_initial_total_score).sum()),
        },
        {"step": "final_selected_cases", "n_cases": int(len(filtered_df))},
        {
            "step": "final_cases_with_both_alleviated_and_unalleviated",
            "n_cases": int(
                (
                    (filtered_df["n_alleviated_symptoms_recomputed"] >= 1)
                    & (filtered_df["n_unalleviated_symptoms_recomputed"] >= 1)
                ).sum()
            ),
        },
    ])

    return filtered_df, summary


def load_tcm_es_embeddings(args, n_cases):
    """Load COVID-19 TCM-ES embeddings and verify case-row alignment."""
    source = {
        "case_symptom_embeddings": np.asarray(
            load_pickle(os.path.join(args.tcm_covid_emb_dir, "case_symptom_embeddings.pkl")),
            dtype=np.float32,
        ),
        "case_herb_embeddings": np.asarray(
            load_pickle(os.path.join(args.tcm_covid_emb_dir, "case_herb_embeddings.pkl")),
            dtype=np.float32,
        ),
        "individual_symptom_embeddings": np.asarray(
            load_pickle(os.path.join(args.tcm_covid_emb_dir, "individual_symptom_embeddings.pkl")),
            dtype=np.float32,
        ),
        "distance_metric": args.tcm_distance_metric,
    }

    for key in ["case_symptom_embeddings", "case_herb_embeddings"]:
        if source[key].shape[0] != n_cases:
            raise ValueError(
                f"{key} has {source[key].shape[0]} rows, "
                f"but clinical data has {n_cases} rows."
            )

    return source


# -----------------------------------------------------------------------------
# Adjusted condition-formula association and sensitivity analyses
# -----------------------------------------------------------------------------

def set_publication_plot_style():
    """Compact publication-style Matplotlib settings for effect-estimate plots."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.2,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,

        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.6,
        "ytick.major.size": 0.0,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


def display_formula_name(x):
    return FORMULA_DISPLAY_LABELS.get(str(x), str(x))


def parse_formula_list(x):
    vals = parse_formula_filter(x)
    if vals is None:
        return None
    return vals


def parse_time_windows(x):
    """Parse comma-separated windows such as '5-9,10-14,all'."""
    if x is None or str(x).strip() == "":
        x = "5-9,6-10,7-11,8-12,9-13,10-14,11-15,12-16,13-17,5-14,7-14,10-17,all"

    out = []
    for token in str(x).split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in {"all", "all_observed"}:
            out.append(("all observed", None, None))
            continue
        if "-" not in token:
            raise ValueError(f"Invalid time-window token: {token}. Expected e.g. 5-9 or all.")
        left, right = token.split("-", 1)
        lo = float(left)
        hi = float(right)
        label = f"{int(lo) if lo.is_integer() else lo:g}-{int(hi) if hi.is_integer() else hi:g}"
        out.append((label, lo, hi))
    return out


def build_adjusted_analysis_case_dataframe(full_df, covid_symptom_list, tcm_source, args):
    """Build the one-row-per-case table for adjusted condition-formula analyses.

    This uses only TCM-ES condition/formula distances. In addition to the
    regression variables, the exported case table retains the baseline and
    follow-up score vectors and their total scores so the adjusted association
    plot can be reproduced from a single patient-level table.
    """
    required = ["age", "sex_male"]
    missing = [c for c in required if c not in full_df.columns]
    if missing:
        raise ValueError(
            f"Adjusted condition-formula analysis requires demographic columns {missing}. "
            f"Expected raw CSV columns: {AGE_COL}, {SEX_COL}."
        )

    metric = tcm_source["distance_metric"]
    rows = []
    n_score_terms = len(covid_symptom_list)

    for row_idx, row in full_df.iterrows():
        initial_score_vector = safe_vector(row["Initial_score_vector"])[:n_score_terms]
        followup_score_vector = safe_vector(row["Followup_score_vector"])[:n_score_terms]

        outcomes = compute_improvement_outcomes(
            init_vec=initial_score_vector,
            follow_vec=followup_score_vector,
            n_symptoms=n_score_terms,
        )

        followup_total_score = float(np.sum(followup_score_vector))

        distance = compute_single_distance(
            tcm_source["case_symptom_embeddings"][row_idx],
            tcm_source["case_herb_embeddings"][row_idx],
            metric=metric,
        )

        rows.append({
            "original_row_idx": int(row["_original_row_idx"]),
            "case_id": row["Cases_id"],
            "formula_name": row["formula_name_for_filter"],
            "formula_label": display_formula_name(row["formula_name_for_filter"]),
            "initial_symptoms": json.dumps(unique_list(row["Initial_symptoms"]), ensure_ascii=False),
            "followup_interval_days": float(row["followup_interval_days"]),
            "age": float(row["age"]) if pd.notna(row["age"]) else np.nan,
            "sex_male": float(row["sex_male"]) if pd.notna(row["sex_male"]) else np.nan,
            "sex_label": (
                "male" if row.get("sex_male") == 1.0
                else ("female" if row.get("sex_male") == 0.0 else np.nan)
            ),
            # JSON strings preserve the complete row-level score information in CSV.
            "initial_score_vector": json.dumps(
                initial_score_vector.astype(float).tolist(), ensure_ascii=False
            ),
            "followup_score_vector": json.dumps(
                followup_score_vector.astype(float).tolist(), ensure_ascii=False
            ),
            "followup_total_score": followup_total_score,
            "condition_formula_distance": float(distance),

            **outcomes,
        })

    out = pd.DataFrame(rows)

    formulas = parse_formula_list(args.include_formulas)
    if formulas is not None:
        out = out[out["formula_name"].isin(formulas)].copy()

    out = out[
        (out["n_initial_score_symptoms"] >= args.min_initial_symptoms)
        & (out["followup_interval_days"] >= args.min_followup_days)
        & (out["followup_interval_days"] <= args.max_followup_days)
        & (out["initial_total_score"] >= args.min_initial_total_score)
        & (out["initial_total_score"] <= args.max_initial_total_score)
    ].copy()

    return out.reset_index(drop=True)

def add_within_formula_z_distance(df, formula_col="formula_name"):
    """Add z-distance within each formula inside the current analysis subset."""
    out = df.copy()
    out["within_formula_z_distance"] = out.groupby(formula_col)["condition_formula_distance"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 1e-12 else np.nan
    )
    return out


def fit_adjusted_distance_model(
    df,
    outcome_col,
    include_formula_fixed_effects=True,
    adjust_initial_total_score=True,
    min_n=30,
):
    """Fit adjusted linear model and return the coefficient of z-distance."""
    covariates = ["within_formula_z_distance", outcome_col]
    candidate_terms = []

    if adjust_initial_total_score:
        candidate_terms.append("initial_total_score")
    candidate_terms.extend([
        "n_initial_score_symptoms",
        "followup_interval_days",
        "age",
        "sex_male",
    ])

    covariates.extend(candidate_terms)
    d = df.dropna(subset=list(dict.fromkeys(covariates))).copy()
    if len(d) < min_n or d["within_formula_z_distance"].std(ddof=0) <= 1e-12:
        return None

    terms = ["within_formula_z_distance"]
    for term in candidate_terms:
        if d[term].nunique(dropna=True) > 1:
            terms.append(term)

    if include_formula_fixed_effects and d["formula_name"].nunique(dropna=True) > 1:
        terms.append("C(formula_name)")

    formula = outcome_col + " ~ " + " + ".join(terms)
    try:
        model = smf.ols(formula, data=d).fit(cov_type="HC3")
    except Exception as exc:
        print(f"Adjusted model failed: {formula}\n{exc}")
        return None

    term = "within_formula_z_distance"
    ci = model.conf_int().loc[term]
    beta = float(model.params[term])
    ci_low = float(ci[0])
    ci_high = float(ci[1])
    p_value = float(model.pvalues[term])

    if not np.all(np.isfinite([beta, ci_low, ci_high, p_value])):
        return None

    return {
        "n": int(len(d)),
        "n_formulas": int(d["formula_name"].nunique(dropna=True)),
        "beta": beta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p": p_value,
        "model_formula": formula,
    }


def format_p_value(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def rounded_xlim(lo, hi):
    """Return a clean x-axis range that includes zero."""
    lo = min(float(lo), 0.0)
    hi = max(float(hi), 0.0)

    span = hi - lo
    if span <= 0:
        return -0.05, 0.05

    pad = 0.10 * span
    lo = lo - pad
    hi = hi + pad

    step = 0.01
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step
    return lo, hi


def save_forest_plot(
    df,
    label_col,
    output_path_prefix,
    xlabel="Adjusted β per within-formula z-score distance",
    title=None,
    xlim=None,
    group_col=None,
    color_map=None,
    marker="s",
):
    """Save grouped publication-style forest plot.

    Parameters
    ----------
    df : DataFrame
        Must contain beta, ci_low, ci_high, n, and label_col.
    label_col : str
        Column for row labels.
    output_path_prefix : str
        Save path prefix.
    xlabel : str
        X-axis label.
    title : str or None
        Figure title.
    xlim : tuple or None
        Optional x limits.
    group_col : str or None
        Optional group/domain column (e.g. severity, age, sex).
    color_map : dict or None
        Optional mapping from group name to base color.
    marker : str
        Matplotlib marker symbol.
    """
    plot_df = df.dropna(subset=["beta", "ci_low", "ci_high"]).copy()
    if len(plot_df) == 0:
        return

    # Preserve original order
    plot_df = plot_df.reset_index(drop=True)

    if group_col is None:
        plot_df["_group"] = "all"
        group_col = "_group"

    # Default group colors
    default_group_colors = {
        "severity": "BuPu",  # 蓝-紫
        "age": "YlOrRd",  # 黄-橙-红（偏红橙）
        "sex": "YlGn",  # 黄-绿

        "formula": "PuBu",  # 可选：方剂图用蓝紫
        "rolling 5-day windows": "Blues",
        "broader windows": "Oranges",
        "all observed": "Greens",

        "window": "Blues",
        "all": "Blues",
    }

    if color_map is None:
        color_map = default_group_colors

    # Build row positions with one dedicated header row per group
    display_rows = []
    group_headers = []
    y = 0.0

    for group_name, subdf in plot_df.groupby(group_col, sort=False):
        n_rows = len(subdf)
        colors = make_group_gradient(color_map.get(str(group_name), "Blues"), n_rows)

        # Header row for group name, e.g. severity / age / sex
        header_y = y
        y += 0.85

        group_start_y = y
        for i, (_, row) in enumerate(subdf.iterrows()):
            row_dict = row.to_dict()
            row_dict["_plot_y"] = y
            row_dict["_plot_color"] = colors[i]
            display_rows.append(row_dict)
            y += 1.0

        group_end_y = y - 1.0

        group_headers.append({
            "group": group_name,
            "header_y": header_y,
            "group_start_y": group_start_y,
            "group_end_y": group_end_y,
            "separator_y": group_start_y - 0.55,
        })

        y += 0.55

    display_df = pd.DataFrame(display_rows)

    fig_h = max(2.6, 0.34 * len(display_df) + 1.0)
    fig, ax = plt.subplots(figsize=(5.2, fig_h))
    ax.set_facecolor("white")

    # Row shading
    # Alternating grey-white stripes for actual data rows only
    stripe_colors = ["#F2F2F2", "#FFFFFF"]
    for idx, (_, row) in enumerate(display_df.iterrows()):
        ax.axhspan(
            row["_plot_y"] - 0.5,
            row["_plot_y"] + 0.5,
            color=stripe_colors[idx % 2],
            zorder=-3,
        )

    # Zero reference line
    ax.axvline(
        0,
        linestyle=(0, (3, 2)),
        linewidth=1.0,
        color="#8A8A8A",
        zorder=-1,
    )

    # Draw CI + marker
    cap_half_height = 0.10
    for _, row in display_df.iterrows():
        beta = float(row["beta"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        yy = float(row["_plot_y"])
        cc = row["_plot_color"]

        ax.plot([lo, hi], [yy, yy], color=cc, linewidth=1.2, solid_capstyle="butt", zorder=2)
        ax.plot([lo, lo], [yy - cap_half_height, yy + cap_half_height], color=cc, linewidth=1.2, zorder=2)
        ax.plot([hi, hi], [yy - cap_half_height, yy + cap_half_height], color=cc, linewidth=1.2, zorder=2)

        ax.scatter(
            beta,
            yy,
            s=26,
            facecolor=cc,
            edgecolor=cc,
            linewidth=0.8,
            marker=marker,
            zorder=3,
        )

    # Left labels
    ax.set_yticks(display_df["_plot_y"].values)
    ax.set_yticklabels([f"{row[label_col]} (n={int(row['n'])})" for _, row in display_df.iterrows()])

    # Group headers + separators
    for gh in group_headers:
        if str(gh["group"]) != "all":
            ax.text(
                -0.02,
                gh["header_y"],
                str(gh["group"]),
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8.2,
                fontweight="bold",
            )

        ax.hlines(
            gh["separator_y"],
            xmin=0,
            xmax=1,
            transform=ax.get_yaxis_transform(),
            colors="#BFBFBF",
            linewidth=0.6,
            zorder=-2,
        )


    # X limits
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        lo = float(np.nanmin(display_df["ci_low"]))
        hi = float(np.nanmax(display_df["ci_high"]))
        span = hi - lo if hi > lo else 0.1
        pad = 0.12 * span
        ax.set_xlim(lo - pad, hi + pad)

    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, pad=6, fontweight="bold")


    # Clean frame
    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_linewidth(0.8)
    ax.spines["top"].set_color("#666666")
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.8, width=0.7, pad=2)
    ax.grid(False)

    # Invert y so first group stays on top
    ax.set_ylim(y - 0.35, -0.55)
    ax.invert_yaxis()

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path_prefix + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(output_path_prefix + ".svg", bbox_inches="tight")
    fig.savefig(output_path_prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)


def make_formula_forest(forest_df, args, out_dir, window_label, min_days=None, max_days=None):
    formulas = parse_formula_list(args.include_formulas)
    if formulas is None:
        formulas = sorted(forest_df["formula_name"].dropna().unique())

    df0 = forest_df[forest_df["formula_name"].isin(formulas)].copy()
    if min_days is not None:
        df0 = df0[(df0["followup_interval_days"] >= min_days) & (df0["followup_interval_days"] <= max_days)].copy()

    rows = []
    for formula_name in formulas:
        d = df0[df0["formula_name"] == formula_name].copy()
        d = add_within_formula_z_distance(d)
        result = fit_adjusted_distance_model(
            d,
            outcome_col=args.improvement_outcome,
            include_formula_fixed_effects=False,
            adjust_initial_total_score=True,
            min_n=args.sensitivity_min_n,
        )
        row = {
            "analysis": "formula_forest",
            "window": window_label,
            "formula_name": formula_name,
            "label": display_formula_name(formula_name),
        }
        if result is None:
            row.update({"n": int(len(d)), "n_formulas": 1, "beta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "model_formula": ""})
        else:
            row.update(result)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, f"formula_forest_{window_label}.csv"), index=False, encoding="utf_8_sig")
    save_forest_plot(
        out,
        label_col="label",
        output_path_prefix=os.path.join(out_dir, f"formula_forest_{window_label}"),
        xlabel="Adjusted β per within-formula z-score distance",
        title=f"Formula-stratified analysis ({window_label.replace('_', '-')})",
        group_col=None,
    )
    return out


def make_time_window_forest(forest_df, args, out_dir):
    formulas = parse_formula_list(args.include_formulas)
    df0 = forest_df.copy()
    if formulas is not None:
        df0 = df0[df0["formula_name"].isin(formulas)].copy()

    rows = []
    for label, lo, hi in parse_time_windows(args.sensitivity_time_windows):
        d = df0.copy()
        if lo is not None:
            d = d[(d["followup_interval_days"] >= lo) & (d["followup_interval_days"] <= hi)].copy()
        d = add_within_formula_z_distance(d)
        result = fit_adjusted_distance_model(
            d,
            outcome_col=args.improvement_outcome,
            include_formula_fixed_effects=True,
            adjust_initial_total_score=True,
            min_n=args.sensitivity_min_n,
        )
        row = {"analysis": "time_window_forest", "window": label, "label": label}
        if result is None:
            row.update({"n": int(len(d)), "n_formulas": int(d["formula_name"].nunique()), "beta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "model_formula": ""})
        else:
            row.update(result)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "time_window_rolling_forest.csv"), index=False, encoding="utf_8_sig")
    save_forest_plot(
        out,
        label_col="label",
        output_path_prefix=os.path.join(out_dir, "time_window_rolling_forest"),
        xlabel="Adjusted β per within-formula z-score distance",
        title="Rolling time-window analysis",
        group_col=None,
    )
    return out

def save_adjusted_case_formula_pca_projection(
    d,
    tcm_source,
    args,
    out_dir,
):
    """
    Project exactly the cases used in the adjusted all-observed
    association plot, together with the formulas appearing among
    those cases, into the existing main-model PCA space.

    No PCA fitting is performed.
    """

    pca_out_dir = Path(out_dir) / "adjusted_condition_formula_pca"
    pca_out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Load existing main-model PCA and visualization scale
    # ------------------------------------------------------------
    with open(args.main_pca_model, "rb") as fp:
        main_pca = pickle.load(fp)

    pca_scale = np.asarray(
        np.load(args.main_pca_scale),
        dtype=float,
    )

    if len(pca_scale) < 3:
        raise ValueError(
            f"Main PCA scale contains only {len(pca_scale)} components."
        )

    # ------------------------------------------------------------
    # Exact cases used in the adjusted association panel
    # ------------------------------------------------------------
    original_indices = (
        d["original_row_idx"]
        .astype(int)
        .to_numpy()
    )

    case_embeddings = np.asarray(
        tcm_source["case_symptom_embeddings"][
            original_indices
        ],
        dtype=float,
    )

    expected_dim = main_pca.components_.shape[1]

    if case_embeddings.shape[1] != expected_dim:
        raise ValueError(
            "Case embedding dimension does not match main PCA: "
            f"{case_embeddings.shape[1]} versus {expected_dim}."
        )

    case_pca_raw = main_pca.transform(
        case_embeddings
    )[:, :3]

    case_pca_scaled = (
        case_pca_raw / pca_scale[:3]
    )

    preferred_columns = [
        "original_row_idx",
        "case_id",
        "formula_name",
        "formula_label",
        "initial_symptoms",
        "followup_interval_days",
        "initial_total_score",
        "n_initial_score_symptoms",
        "condition_formula_distance",
        "within_formula_z_distance",
        args.improvement_outcome,
    ]

    case_columns = [
        column
        for column in preferred_columns
        if column in d.columns
    ]

    case_pca_df = d[case_columns].copy()

    case_pca_df["PC1_raw"] = case_pca_raw[:, 0]
    case_pca_df["PC2_raw"] = case_pca_raw[:, 1]
    case_pca_df["PC3_raw"] = case_pca_raw[:, 2]

    case_pca_df["PC1_scaled"] = case_pca_scaled[:, 0]
    case_pca_df["PC2_scaled"] = case_pca_scaled[:, 1]
    case_pca_df["PC3_scaled"] = case_pca_scaled[:, 2]

    case_pca_df.to_pickle(
        pca_out_dir /
        "adjusted_association_case_initial_symptom_pca.pkl"
    )

    case_pca_df.to_csv(
        pca_out_dir /
        "adjusted_association_case_initial_symptom_pca.csv",
        index=False,
        encoding="utf_8_sig",
    )

    # ------------------------------------------------------------
    # Formulas appearing among these exact cases
    # ------------------------------------------------------------
    formula_embedding_path = (
        Path(args.tcm_covid_emb_dir) /
        "COVID_formula_embeddings.pkl"
    )

    with open(formula_embedding_path, "rb") as fp:
        formula_embedding_dict = pickle.load(fp)

    observed_formula_names = (
        d["formula_name"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    missing_formulas = [
        name
        for name in observed_formula_names
        if name not in formula_embedding_dict
    ]

    if missing_formulas:
        raise KeyError(
            "Formulas are missing from "
            "COVID_formula_embeddings.pkl: "
            f"{missing_formulas}"
        )

    formula_embeddings = np.vstack([
        np.asarray(
            formula_embedding_dict[name],
            dtype=float,
        ).reshape(-1)
        for name in observed_formula_names
    ])

    if formula_embeddings.shape[1] != expected_dim:
        raise ValueError(
            "Formula embedding dimension does not match main PCA: "
            f"{formula_embeddings.shape[1]} versus {expected_dim}."
        )

    formula_pca_raw = main_pca.transform(
        formula_embeddings
    )[:, :3]

    formula_pca_scaled = (
        formula_pca_raw / pca_scale[:3]
    )

    formula_counts = (
        d["formula_name"]
        .astype(str)
        .value_counts()
    )

    formula_pca_df = pd.DataFrame({
        "formula_name": observed_formula_names,
        "formula_label": [
            display_formula_name(name)
            for name in observed_formula_names
        ],
        "n_cases": [
            int(formula_counts.get(name, 0))
            for name in observed_formula_names
        ],
        "PC1_raw": formula_pca_raw[:, 0],
        "PC2_raw": formula_pca_raw[:, 1],
        "PC3_raw": formula_pca_raw[:, 2],
        "PC1_scaled": formula_pca_scaled[:, 0],
        "PC2_scaled": formula_pca_scaled[:, 1],
        "PC3_scaled": formula_pca_scaled[:, 2],
    })

    formula_pca_df.to_pickle(
        pca_out_dir /
        "adjusted_association_observed_formula_pca.pkl"
    )

    formula_pca_df.to_csv(
        pca_out_dir /
        "adjusted_association_observed_formula_pca.csv",
        index=False,
        encoding="utf_8_sig",
    )

    plot_adjusted_case_formula_pca_3d(
        case_pca_df=case_pca_df,
        formula_pca_df=formula_pca_df,
        output_prefix=(
            pca_out_dir /
            "adjusted_association_initial_symptoms_formulas_scaled"
        ),
        scaled=True,
    )

    plot_adjusted_case_formula_pca_3d(
        case_pca_df=case_pca_df,
        formula_pca_df=formula_pca_df,
        output_prefix=(
            pca_out_dir /
            "adjusted_association_initial_symptoms_formulas_raw"
        ),
        scaled=False,
    )

    print(
        "Saved PCA projections for adjusted association cases:",
        len(case_pca_df),
    )
    print(
        "Observed formulas:",
        observed_formula_names,
    )

    return case_pca_df, formula_pca_df


def plot_adjusted_case_formula_pca_3d(
    case_pca_df,
    formula_pca_df,
    output_prefix,
    scaled=True,
):
    suffix = "scaled" if scaled else "raw"

    pc_columns = [
        f"PC1_{suffix}",
        f"PC2_{suffix}",
        f"PC3_{suffix}",
    ]

    formula_names = (
        formula_pca_df["formula_name"]
        .astype(str)
        .tolist()
    )

    cmap = plt.get_cmap("tab10")
    formula_colors = {
        name: cmap(i % 10)
        for i, name in enumerate(formula_names)
    }

    fig = plt.figure(figsize=(6.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    for formula_name in formula_names:
        color = formula_colors[formula_name]

        case_subset = case_pca_df[
            case_pca_df["formula_name"].astype(str)
            == formula_name
        ]

        # Initial symptom-pattern embeddings
        ax.scatter(
            case_subset[pc_columns[0]],
            case_subset[pc_columns[1]],
            case_subset[pc_columns[2]],
            s=10,
            alpha=0.23,
            color=color,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
        )

        # Formula embedding
        formula_row = formula_pca_df[
            formula_pca_df["formula_name"].astype(str)
            == formula_name
        ].iloc[0]

        x = float(formula_row[pc_columns[0]])
        y = float(formula_row[pc_columns[1]])
        z = float(formula_row[pc_columns[2]])

        ax.scatter(
            [x],
            [y],
            [z],
            marker="*",
            s=230,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            depthshade=False,
            zorder=10,
        )

        ax.text(
            x,
            y,
            z,
            "  " + str(formula_row["formula_label"]),
            fontsize=8,
        )

    if scaled:
        ax.set_xlabel("Scaled PC1")
        ax.set_ylabel("Scaled PC2")
        ax.set_zlabel("Scaled PC3")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    ax.set_title(
        "Initial symptom patterns and prescribed formulas "
        f"(n={len(case_pca_df)})",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )

    ax.view_init(elev=20, azim=-62)
    ax.set_box_aspect((1.0, 1.0, 0.85))
    ax.grid(False)

    fig.tight_layout()

    output_prefix = str(output_prefix)

    fig.savefig(
        output_prefix + ".png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        output_prefix + ".pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_prefix + ".svg",
        bbox_inches="tight",
    )

    plt.close(fig)

def run_adjusted_condition_formula_association(
    analysis_df,
    tcm_source,
    args,
    out_dir,
    n_bins=15,
):
    """Plot patient-level covariate-adjusted improvement by distance bin.

    Patient-level adjusted improvement is derived from the full adjusted model:

        adjusted_y_i = observed_y_i - nuisance_i + mean(nuisance)

    where nuisance_i is the fitted contribution of all model terms except
    within-formula standardized patient-formula distance. This removes the
    patient-specific contribution of baseline severity, symptom count,
    follow-up interval, age, sex, and formula fixed effects while retaining the
    observed residual variation and the distance-associated component. Adding
    back the mean nuisance contribution keeps the adjusted outcome on the
    original clinical-outcome scale.

    Orange points show the mean patient-level adjusted improvement in
    equal-frequency distance bins. Error bars show the within-bin standard
    error of the mean (SE), not a model-based confidence interval. The blue line
    and ribbon show the adjusted linear trend and its model-based 95% CI.
    """
    formulas = parse_formula_list(args.include_formulas)

    d = analysis_df.copy()
    if formulas is not None:
        d = d[d["formula_name"].isin(formulas)].copy()

    d = add_within_formula_z_distance(d)
    outcome_col = args.improvement_outcome

    required_cols = [
        outcome_col,
        "within_formula_z_distance",
        "initial_total_score",
        "n_initial_score_symptoms",
        "followup_interval_days",
        "age",
        "sex_male",
        "formula_name",
    ]
    d = d.dropna(subset=required_cols).copy().reset_index(drop=True)

    if len(d) < args.adjusted_min_n:
        print(
            "Adjusted all-observed association plot skipped: "
            f"only {len(d)} complete cases."
        )
        return

    if args.save_pca_projection:
        save_adjusted_case_formula_pca_projection(
            d=d,
            tcm_source=tcm_source,
            args=args,
            out_dir=out_dir,
        )


    term = "within_formula_z_distance"
    linear_formula = (
        f"{outcome_col} ~ {term}"
        " + initial_total_score"
        " + n_initial_score_symptoms"
        " + followup_interval_days"
        " + age"
        " + sex_male"
        " + C(formula_name)"
    )
    linear_model = smf.ols(linear_formula, data=d).fit(cov_type="HC3")

    beta = float(linear_model.params[term])
    ci_low, ci_high = linear_model.conf_int().loc[term].astype(float)
    p_value = float(linear_model.pvalues[term])

    # ------------------------------------------------------------------
    # Patient-level adjusted improvement.
    # ------------------------------------------------------------------
    observed = d[outcome_col].astype(float).to_numpy()
    fitted = np.asarray(linear_model.fittedvalues, dtype=float)
    residual = observed - fitted
    distance_component = beta * d[term].astype(float).to_numpy()

    # All fitted contributions except the distance term, including intercept
    # and formula fixed effects.
    nuisance_component = fitted - distance_component

    # Re-center after nuisance removal so the adjusted outcome remains on the
    # original clinical scale and has the same overall mean as the observed outcome.
    adjusted_improvement = (
        observed
        - nuisance_component
        + float(np.mean(nuisance_component))
    )

    d["outcome_name"] = outcome_col
    d["observed_improvement"] = observed
    d["full_model_fitted_improvement"] = fitted
    d["full_model_residual"] = residual
    d["distance_model_component"] = distance_component
    d["nuisance_model_component"] = nuisance_component
    d["adjusted_improvement"] = adjusted_improvement

    # ------------------------------------------------------------------
    # Equal-frequency distance bins. n_bins is now actually respected.
    # ------------------------------------------------------------------
    usable_bins = min(int(n_bins), int(d[term].nunique()))
    if usable_bins < 2:
        print("Adjusted association plot skipped: fewer than two unique distance values.")
        return

    d["distance_bin"] = pd.qcut(
        d[term],
        q=usable_bins,
        labels=False,
        duplicates="drop",
    ).astype(int)

    bin_rows = []
    for bin_id, bdf in d.groupby("distance_bin", sort=True):
        values = bdf["adjusted_improvement"].astype(float).to_numpy()
        n_bin = int(len(values))
        mean_value = float(np.mean(values))
        within_bin_sd = float(np.std(values, ddof=1)) if n_bin > 1 else np.nan
        within_bin_se = (
            float(within_bin_sd / np.sqrt(n_bin))
            if n_bin > 1 and np.isfinite(within_bin_sd)
            else np.nan
        )

        bin_rows.append({
            "bin_id": int(bin_id),
            "n": n_bin,
            "x_mean": float(bdf[term].mean()),
            "x_min": float(bdf[term].min()),
            "x_max": float(bdf[term].max()),
            "adjusted_mean": mean_value,
            "within_bin_sd": within_bin_sd,
            "within_bin_se": within_bin_se,
            "ci95_low_from_within_bin_se": (
                mean_value - 1.96 * within_bin_se
                if np.isfinite(within_bin_se) else np.nan
            ),
            "ci95_high_from_within_bin_se": (
                mean_value + 1.96 * within_bin_se
                if np.isfinite(within_bin_se) else np.nan
            ),
        })

    bin_summary = pd.DataFrame(bin_rows)

    # ------------------------------------------------------------------
    # Model-based adjusted linear trend and 95% CI.
    # ------------------------------------------------------------------
    def marginal_mean_and_ci(model, new_data):
        design = np.asarray(
            build_design_matrices(
                [model.model.data.design_info],
                new_data,
                return_type="dataframe",
            )[0]
        )
        mean_design = design.mean(axis=0)
        params = np.asarray(model.params)
        covariance = np.asarray(model.cov_params())
        mean = float(mean_design @ params)
        se = float(np.sqrt(mean_design @ covariance @ mean_design))
        return mean, mean - 1.96 * se, mean + 1.96 * se

    x_seq = np.linspace(
        float(np.nanpercentile(d[term], 2)),
        float(np.nanpercentile(d[term], 98)),
        220,
    )
    trend_mean = []
    trend_low = []
    trend_high = []
    for x_value in x_seq:
        new_data = d.copy()
        new_data[term] = x_value
        mean, low, high = marginal_mean_and_ci(linear_model, new_data)
        trend_mean.append(mean)
        trend_low.append(low)
        trend_high.append(high)

    trend_df = pd.DataFrame({
        term: x_seq,
        "adjusted_mean": trend_mean,
        "ci_low": trend_low,
        "ci_high": trend_high,
    })

    q10, q90 = np.quantile(d[term], [0.10, 0.90])
    q10_q90_difference = beta * (q90 - q10)

    # ------------------------------------------------------------------
    # Save all data needed to reproduce the panel.
    # ------------------------------------------------------------------
    preferred_patient_cols = [
        "original_row_idx",
        "case_id",
        "formula_name",
        "formula_label",
        "age",
        "sex_male",
        "sex_label",
        "followup_interval_days",
        "initial_score_vector",
        "followup_score_vector",
        "initial_total_score",
        "followup_total_score",
        "n_initial_score_symptoms",
        "condition_formula_distance",
        "within_formula_z_distance",
        "distance_bin",
        "outcome_name",
        outcome_col,
        "observed_improvement",
        "adjusted_improvement",
        "full_model_fitted_improvement",
        "full_model_residual",
        "distance_model_component",
        "nuisance_model_component",
    ]
    patient_cols = [c for c in preferred_patient_cols if c in d.columns]
    d[patient_cols].to_csv(
        os.path.join(
            out_dir,
            "patient_data.csv",
        ),
        index=False,
        encoding="utf_8_sig",
    )

    bin_summary.to_csv(
        os.path.join(out_dir, "distance_bins.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    trend_df.to_csv(
        os.path.join(out_dir, "adjusted_trend.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    pd.DataFrame([{
        "analysis": "adjusted_condition_formula_association",
        "outcome": outcome_col,
        "n": int(len(d)),
        "n_formulas": int(d["formula_name"].nunique()),
        "n_bins": int(bin_summary.shape[0]),
        "beta": beta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p": p_value,
        "q10": float(q10),
        "q90": float(q90),
        "q10_q90_adjusted_difference": float(q10_q90_difference),
        "patient_adjustment_method": (
            "observed outcome minus fitted nuisance contribution from all model "
            "terms except within-formula distance, plus mean nuisance contribution"
        ),
        "bin_error_bar": "within-bin standard error of adjusted improvement",
        "model_formula": linear_formula,
    }]).to_csv(
        os.path.join(out_dir, "model_summary.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    # ------------------------------------------------------------------
    # Plot: adjusted mean +/- within-bin SE, plus adjusted linear trend.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.1))

    ax.fill_between(
        x_seq,
        trend_low,
        trend_high,
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x_seq,
        trend_mean,
        linewidth=2.0,
        zorder=2,
    )
    ax.errorbar(
        bin_summary["x_mean"],
        bin_summary["adjusted_mean"],
        yerr=bin_summary["within_bin_se"],
        fmt="o",
        markersize=5.5,
        capsize=2.6,
        linewidth=1.0,
        zorder=3,
    )

    p_text = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
    annotation = (
        f"Adjusted beta = {beta:.3f}\n"
        f"95% CI [{ci_low:.3f}, {ci_high:.3f}]\n"
        f"P = {p_text}\n"
        f"Q10-Q90 difference = {q10_q90_difference:.2f}"
    )
    ax.text(
        0.97,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
    )

    outcome_ylabel = {
        "total_improvement_score": "Adjusted total symptom-score reduction",
        "fractional_improvement": "Adjusted fractional symptom-score reduction",
        "mean_improvement_per_initial_symptom": "Adjusted mean reduction per initial symptom",
        "mean_symptom_improvement_fraction": "Adjusted mean per-symptom fractional improvement",
        "alleviated_symptom_fraction": "Adjusted alleviated symptom fraction",
    }.get(outcome_col, f"Adjusted {outcome_col}")

    ax.set_title(
        f"Patient-formula proximity and symptom improvement (n={len(d)})",
        fontsize=10.0,
        fontweight="bold",
        pad=8,
    )
    ax.set_xlabel(
        "Within-formula standardized patient-formula distance",
        fontsize=8.7,
    )
    ax.set_ylabel(outcome_ylabel, fontsize=8.7)

    y_candidates_low = [
        float(np.nanmin(trend_low)),
        float(np.nanmin(
            bin_summary["adjusted_mean"] - bin_summary["within_bin_se"]
        )),
    ]
    y_candidates_high = [
        float(np.nanmax(trend_high)),
        float(np.nanmax(
            bin_summary["adjusted_mean"] + bin_summary["within_bin_se"]
        )),
    ]
    y_low = min(y_candidates_low)
    y_high = max(y_candidates_high)
    y_span = y_high - y_low
    y_pad = 0.10 * y_span if y_span > 0 else 0.5
    ax.set_ylim(y_low - y_pad, y_high + y_pad)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7.8)
    ax.grid(False)

    fig.tight_layout()
    prefix = os.path.join(out_dir, "condition_formula_distance_vs_improvement")
    fig.savefig(prefix + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(prefix + ".svg", bbox_inches="tight")
    fig.savefig(prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)

def make_patient_subgroup_forest(forest_df, args, out_dir, label, min_days=None, max_days=None):
    formulas = parse_formula_list(args.include_formulas)
    df0 = forest_df.copy()
    if formulas is not None:
        df0 = df0[df0["formula_name"].isin(formulas)].copy()
    if min_days is not None:
        df0 = df0[(df0["followup_interval_days"] >= min_days) & (df0["followup_interval_days"] <= max_days)].copy()

    df0["severity_group"] = pd.cut(
        df0["initial_total_score"],
        bins=[-np.inf, 17, 24, np.inf],
        labels=["low <=17", "medium 18-24", "high >=25"],
    )
    df0["age_group"] = pd.cut(
        df0["age"],
        bins=[-np.inf, 44, 59, np.inf],
        labels=["<45", "45-59", ">=60"],
    )

    rows = []
    group_specs = [
        ("severity_group", "severity"),
        ("age_group", "age"),
        ("sex_label", "sex"),
    ]
    for col, domain_name in group_specs:
        for group_label, d in df0.groupby(col, observed=True):
            d = add_within_formula_z_distance(d)
            result = fit_adjusted_distance_model(
                d,
                outcome_col=args.improvement_outcome,
                include_formula_fixed_effects=True,
                adjust_initial_total_score=True,
                min_n=args.sensitivity_min_n,
            )
            row = {
                "analysis": "patient_subgroup_forest",
                "window": label,
                "group_domain": domain_name,
                "label": str(group_label),
            }
            if result is None:
                row.update({"n": int(len(d)), "n_formulas": int(d["formula_name"].nunique()), "beta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "model_formula": ""})
            else:
                row.update(result)
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, f"patient_subgroup_forest_{label}.csv"), index=False, encoding="utf_8_sig")
    save_forest_plot(
        out,
        label_col="label",
        group_col="group_domain",
        output_path_prefix=os.path.join(out_dir, f"patient_subgroup_forest_{label}"),
        xlabel="Adjusted β per within-formula z-score distance",
        title="Patient subgroup analysis",
    )
    return out


def make_initial_severity_forest(forest_df, args, out_dir):
    formulas = parse_formula_list(args.include_formulas)
    df0 = forest_df.copy()
    if formulas is not None:
        df0 = df0[df0["formula_name"].isin(formulas)].copy()

    severity_windows = [("5-9", 5, 9), ("8-12", 8, 12), ("10-14", 10, 14), ("12-16", 12, 16), ("all observed", None, None)]
    rows = []
    for window_label, lo, hi in severity_windows:
        dwin = df0.copy()
        if lo is not None:
            dwin = dwin[(dwin["followup_interval_days"] >= lo) & (dwin["followup_interval_days"] <= hi)].copy()

        dwin["severity_group"] = pd.cut(
            dwin["initial_total_score"],
            bins=[-np.inf, 17, 24, np.inf],
            labels=["low <=17", "medium 18-24", "high >=25"],
        )

        for sev_label, d in dwin.groupby("severity_group", observed=True):
            d = add_within_formula_z_distance(d)
            result = fit_adjusted_distance_model(
                d,
                outcome_col=args.improvement_outcome,
                include_formula_fixed_effects=True,
                adjust_initial_total_score=False,
                min_n=args.sensitivity_min_n,
            )
            row = {"analysis": "initial_severity_forest", "window": window_label, "severity_group": str(sev_label), "label": f"{window_label}: {sev_label}"}
            if result is None:
                row.update({"n": int(len(d)), "n_formulas": int(d["formula_name"].nunique()), "beta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "model_formula": ""})
            else:
                row.update(result)
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "initial_severity_by_time_window_forest.csv"), index=False, encoding="utf_8_sig")
    save_forest_plot(
        out,
        label_col="severity_group",
        group_col="window",
        output_path_prefix=os.path.join(out_dir, "initial_severity_by_time_window_forest"),
        xlabel="Adjusted β per within-formula z-score distance",
        title="Initial severity across time windows",
    )
    return out



def run_formula_to_symptom_relationships(
    full_df,
    analysis_df,
    covid_symptom_list,
    tcm_source,
    args,
    out_dir,
):
    """Compare prescribed-formula distance to alleviated and unalleviated symptoms.

    Symptoms are defined from the 60-item COVID score vectors. Initial symptoms
    with lower follow-up scores are classified as alleviated; the remaining
    baseline-positive symptoms are classified as unalleviated. Distances are
    averaged within case and compared with a paired t-test.
    """
    analysis_dir = out_dir
    ensure_dir(analysis_dir)

    formulas = parse_formula_list(args.include_formulas)

    selected = analysis_df.copy()
    if formulas is not None:
        selected = selected[selected["formula_name"].isin(formulas)].copy()

    selected = add_within_formula_z_distance(selected)
    outcome_col = args.improvement_outcome

    required_cols = [
        outcome_col,
        "within_formula_z_distance",
        "initial_total_score",
        "n_initial_score_symptoms",
        "followup_interval_days",
        "age",
        "sex_male",
        "formula_name",
    ]
    selected = selected.dropna(subset=required_cols).copy().reset_index(drop=True)

    selected.to_csv(
        os.path.join(analysis_dir, "analysis_case_pool.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    print("\nFormula-to-symptom analysis case pool:", len(selected))

    dist_diameter = FORMULA_SYMPTOM_DISTANCE_NORMALIZATION

    symptom_embeddings = np.asarray(
        tcm_source["individual_symptom_embeddings"],
        dtype=float,
    )
    case_formula_embeddings = np.asarray(
        tcm_source["case_herb_embeddings"],
        dtype=float,
    )

    n_symptoms = len(covid_symptom_list)
    rows = []

    for _, selected_row in selected.iterrows():
        original_idx = int(selected_row["original_row_idx"])
        row = full_df.iloc[original_idx]

        initial_scores = safe_vector(row["Initial_score_vector"])[:n_symptoms]
        followup_scores = safe_vector(row["Followup_score_vector"])[:n_symptoms]

        if len(initial_scores) != n_symptoms or len(followup_scores) != n_symptoms:
            raise ValueError(
                f"Score-vector length mismatch for case {row['Cases_id']}: "
                f"initial={len(initial_scores)}, followup={len(followup_scores)}, "
                f"expected={n_symptoms}"
            )

        alleviated_ids = np.where(
            (initial_scores > followup_scores) & (initial_scores > 0)
        )[0]
        unalleviated_ids = np.where(
            (initial_scores <= followup_scores) & (initial_scores > 0)
        )[0]

        if len(alleviated_ids) == 0 or len(unalleviated_ids) == 0:
            continue

        formula_emb = case_formula_embeddings[original_idx]

        alleviated_distances = (
            np.linalg.norm(
                symptom_embeddings[alleviated_ids] - formula_emb[None, :],
                axis=1,
            )
            / dist_diameter
        )
        unalleviated_distances = (
            np.linalg.norm(
                symptom_embeddings[unalleviated_ids] - formula_emb[None, :],
                axis=1,
            )
            / dist_diameter
        )

        rows.append({
            "original_row_idx": original_idx,
            "case_id": row["Cases_id"],
            "formula_name": row["formula_name_for_filter"],
            "followup_interval_days": float(row["followup_interval_days"]),
            "n_alleviated_symptoms": int(len(alleviated_ids)),
            "n_unalleviated_symptoms": int(len(unalleviated_ids)),
            "mean_distance_alleviated": float(np.mean(alleviated_distances)),
            "mean_distance_unalleviated": float(np.mean(unalleviated_distances)),
        })

    paired_df = pd.DataFrame(rows)
    if len(paired_df) == 0:
        raise ValueError(
            "No cases contain both alleviated "
            "and unalleviated initial symptoms."
        )

    paired_df["distance_difference_unalleviated_minus_alleviated"] = (
        paired_df["mean_distance_unalleviated"]
        - paired_df["mean_distance_alleviated"]
    )

    t_result = stats.ttest_rel(
        paired_df["mean_distance_alleviated"].to_numpy(dtype=float),
        paired_df["mean_distance_unalleviated"].to_numpy(dtype=float),
    )

    paired_df.to_csv(
        os.path.join(analysis_dir, "paired_case_distances.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    # Long-format table containing exactly the values used for the two boxes.
    common_cols = ["case_id", "original_row_idx", "formula_name"]
    alleviated_plot = paired_df[common_cols].copy()
    alleviated_plot["group"] = "Alleviated symptoms"
    alleviated_plot["embedding_distance"] = paired_df[
        "mean_distance_alleviated"
    ].to_numpy(dtype=float)

    unalleviated_plot = paired_df[common_cols].copy()
    unalleviated_plot["group"] = "Unalleviated symptoms"
    unalleviated_plot["embedding_distance"] = paired_df[
        "mean_distance_unalleviated"
    ].to_numpy(dtype=float)

    boxplot_df = pd.concat(
        [alleviated_plot, unalleviated_plot],
        ignore_index=True,
    )
    boxplot_df.to_csv(
        os.path.join(analysis_dir, "plot_data.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    summary_df = pd.DataFrame([{
        "base_case_pool_n": int(len(selected)),
        "paired_analyzable_n": int(len(paired_df)),
        "mean_distance_alleviated": float(
            paired_df["mean_distance_alleviated"].mean()
        ),
        "mean_distance_unalleviated": float(
            paired_df["mean_distance_unalleviated"].mean()
        ),
        "mean_difference_unalleviated_minus_alleviated": float(
            paired_df["distance_difference_unalleviated_minus_alleviated"].mean()
        ),
        "paired_t": float(t_result.statistic),
        "paired_t_p": float(t_result.pvalue),
        "distance_normalization_diameter": float(dist_diameter),
    }])
    summary_df.to_csv(
        os.path.join(analysis_dir, "summary.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    # Two-box paired comparison.
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.boxplot(
        [
            paired_df["mean_distance_alleviated"].to_numpy(dtype=float),
            paired_df["mean_distance_unalleviated"].to_numpy(dtype=float),
        ],
        showfliers=True,
        widths=0.55,
    )
    ax.set_xticklabels([
        "Alleviated\nsymptoms",
        "Unalleviated\nsymptoms",
    ])
    ax.set_ylabel("Embedding distance to\nprescribed formulas")
    ax.set_title("COVID-19 clinical cases")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    prefix = os.path.join(
        analysis_dir,
        "alleviated_vs_unalleviated_distance",
    )
    fig.savefig(prefix + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(prefix + ".svg", bbox_inches="tight")
    fig.savefig(prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print(
        "Paired analyzable cases:",
        len(paired_df),
        "| mean alleviated distance:",
        float(paired_df["mean_distance_alleviated"].mean()),
        "| mean unalleviated distance:",
        float(paired_df["mean_distance_unalleviated"].mean()),
        "| paired t-test:",
        t_result,
    )

    return paired_df, boxplot_df, summary_df


def run_sensitivity_analyses(adjusted_df, args):
    """Run optional subgroup, follow-up-window, and formula-specific analyses."""
    set_publication_plot_style()
    out_dir = os.path.join(args.out_dir, "sensitivity_analyses")
    ensure_dir(out_dir)

    all_tables = [
        make_formula_forest(adjusted_df, args, out_dir, window_label="all_observed"),
        make_formula_forest(adjusted_df, args, out_dir, window_label="5_9", min_days=5, max_days=9),
        make_formula_forest(adjusted_df, args, out_dir, window_label="10_14", min_days=10, max_days=14),
        make_time_window_forest(adjusted_df, args, out_dir),
        make_patient_subgroup_forest(adjusted_df, args, out_dir, label="all_observed"),
        make_patient_subgroup_forest(adjusted_df, args, out_dir, label="10_14", min_days=10, max_days=14),
        make_initial_severity_forest(adjusted_df, args, out_dir),
    ]

    combined = pd.concat(all_tables, ignore_index=True, sort=False)
    combined.to_csv(
        os.path.join(out_dir, "sensitivity_analysis_all_results.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    print("\nSensitivity analyses saved to:", out_dir)
    return combined


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "COVID-19 condition-improvement analyses using the dedicated "
            "60-symptom COVID vocabulary."
        )
    )

    # Public example data are the defaults. Replace these three paths with the
    # full-data counterparts to run the complete cohort without changing the workflow.
    parser.add_argument(
        "--clinical-data",
        default="data/COVID_19_data/COVID_19_data_for_embeddings_example_200.pkl",
    )
    parser.add_argument(
        "--raw-covid-csv",
        default="data/COVID_19_data/COVID_data_example_200.csv",
    )
    parser.add_argument(
        "--covid-symptom-list",
        default="data/COVID_19_data/COVID_symptom_list.pkl",
    )

    parser.add_argument(
        "--tcm-covid-emb-dir",
        default="results/embeddings/COVID_19_cases/example_200",
    )

    parser.add_argument(
        "--out-dir",
        default="results/COVID_19_condition_improvement",
    )

    # Explicit eligibility thresholds. The broad defaults retain the public
    # example cohort while keeping all thresholds visible for later adjustment.
    parser.add_argument("--min-followup-days", type=float, default=0.0)
    parser.add_argument("--max-followup-days", type=float, default=100.0)
    parser.add_argument("--min-initial-symptoms", type=int, default=1)
    parser.add_argument("--min-initial-total-score", type=float, default=0.0)
    parser.add_argument("--max-initial-total-score", type=float, default=100.0)
    parser.add_argument(
        "--include-formulas",
        default="方D,方4,方B,方A",
        help="Comma-separated formula names used in the main analyses.",
    )

    parser.add_argument(
        "--improvement-outcome",
        choices=[
            "total_improvement_score",
            "fractional_improvement",
            "mean_improvement_per_initial_symptom",
            "mean_symptom_improvement_fraction",
            "alleviated_symptom_fraction",
        ],
        default="total_improvement_score",
    )

    parser.add_argument(
        "--tcm-distance-metric",
        choices=["euclidean", "cosine"],
        default="euclidean",
    )

    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # Lower sample-size thresholds make the 200-case public example usable.
    # They remain explicit so the full-cohort analysis can use stricter values.
    parser.add_argument(
        "--adjusted-min-n",
        type=int,
        default=20,
        help="Minimum complete-case sample size for the main adjusted association.",
    )
    parser.add_argument(
        "--sensitivity-min-n",
        type=int,
        default=10,
        help="Minimum sample size for each optional sensitivity subgroup model.",
    )
    parser.add_argument(
        "--adjusted-n-bins",
        type=int,
        default=10,
        help="Number of equal-frequency bins in the adjusted association plot.",
    )

    parser.add_argument(
        "--run-sensitivity-analyses",
        action="store_true",
        help="Run formula-specific, follow-up-window, and patient-subgroup sensitivity analyses.",
    )
    parser.add_argument(
        "--sensitivity-time-windows",
        default="5-9,6-10,7-11,8-12,9-13,10-14,11-15,12-16,13-17,all",
        help="Comma-separated follow-up windows for the optional rolling-window analysis.",
    )

    parser.add_argument(
        "--save-pca-projection",
        action="store_true",
        help="Save the optional PCA projection for cases used in the adjusted association.",
    )
    parser.add_argument(
        "--main-pca-model",
        default=(
            "results/TCM_embedding_analysis/original(average)/pca/"
            "TCM_embedding_pca_model.pkl"
        ),
    )
    parser.add_argument(
        "--main-pca-scale",
        default=(
            "results/TCM_embedding_analysis/original(average)/pca/"
            "pca_95percent_scale.npy"
        ),
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_dir(args.out_dir)

    clinical_df, raw_csv, covid_symptom_list = load_inputs(args)
    case_df_all = prepare_case_dataframe(clinical_df, raw_csv)

    tcm_source = load_tcm_es_embeddings(
        args,
        n_cases=len(case_df_all),
    )

    adjusted_df = build_adjusted_analysis_case_dataframe(
        full_df=case_df_all,
        covid_symptom_list=covid_symptom_list,
        tcm_source=tcm_source,
        args=args,
    )
    print("Loaded cases:", len(case_df_all))
    print("Main analysis cases after explicit filters:", len(adjusted_df))

    formula_symptom_dir = os.path.join(
        args.out_dir,
        "formula_to_symptom_relationships",
    )
    run_formula_to_symptom_relationships(
        full_df=case_df_all,
        analysis_df=adjusted_df,
        covid_symptom_list=covid_symptom_list,
        tcm_source=tcm_source,
        args=args,
        out_dir=formula_symptom_dir,
    )

    adjusted_dir = os.path.join(
        args.out_dir,
        "adjusted_condition_formula_association",
    )
    ensure_dir(adjusted_dir)
    adjusted_df.to_csv(
        os.path.join(adjusted_dir, "analysis_case_table.csv"),
        index=False,
        encoding="utf_8_sig",
    )
    run_adjusted_condition_formula_association(
        analysis_df=adjusted_df,
        tcm_source=tcm_source,
        args=args,
        out_dir=adjusted_dir,
        n_bins=args.adjusted_n_bins,
    )

    if args.run_sensitivity_analyses:
        run_sensitivity_analyses(adjusted_df, args)

    print("\nDone.")
    print("Clinical data:", args.clinical_data)
    print("Results saved to:", args.out_dir)


if __name__ == "__main__":
    main()
