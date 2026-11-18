"""
05_analyze_and_plot.py
Final analysis, paired comparisons, summary CSV tables, and 200-dpi figures (Figures 1-8).
"""

import argparse
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from calibrators import (
    BaseRateCalibrator,
    BetaCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    VennAbersCalibrator,
)
from common import (
    CALIBRATION_NAMES,
    EVAL_YEARS,
    FRACTIONS,
    HORIZON,
    MERGED_RACE_NAME,
    METHODS,
    MODEL_NAMES,
    N_BOOT,
    N_REPEATS_FULL,
    N_REPEATS_TEST,
    OUTPUTS_DIR,
    OUTPUTS_NO_OCCP_DIR,
    PROCESSED_DIR,
    RACE_CODE_MAP,
    RACE_GROUPS,
    RAW_DIR,
    SEED,
    SEX_CODE_MAP,
    SEX_GROUPS,
    SMALL_GROUP_THRESHOLD,
    STATE,
    SURVEY,
    TEST_SAMPLE_FRAC,
    YEARS,
    apply_race_merging_rule,
    build_and_cache_all_years,
    build_models,
    build_preprocessor,
    draw_recalibration_sample,
    ensure_dirs,
    fit_and_predict_methods,
    get_group_slices,
    get_outputs_dir,
    get_predictions_cache_path,
    split_2014_data,
    to_dense_if_sparse,
    train_and_predict_all_years,
)
from metrics import (
    bootstrap_ece_and_gap_multi,
    compute_all_metrics,
    compute_brier,
    compute_ece,
    compute_mean_gap,
    compute_reliability_bins,
)

# --- Part 1a: Plot styles and command-line arguments ---
GROUP_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosted Trees",
]

METHOD_COLORS = {
    "None": "#7f7f7f",
    "BaseRate": "#1f77b4",
    "Platt": "#2ca02c",
    "Beta": "#9467bd",
    "VennAbers": "#ff7f0e",
    "Isotonic": "#d62728",
}

METHOD_MARKERS = {
    "None": "x",
    "BaseRate": "o",
    "Platt": "s",
    "Beta": "^",
    "VennAbers": "D",
    "Isotonic": "v",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final tables, paired comparisons, figures, and extension analyses."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode reading *_test.csv inputs and writing *_test outputs.",
    )
    parser.add_argument(
        "--monitoring",
        action="store_true",
        help="Run monitoring rule evaluation (estimate ECE & MeanGap on R, trigger if CI lower bound > threshold).",
    )
    parser.add_argument(
        "--group-vs-shared",
        action="store_true",
        help="Run group-specific calibrators (fitted on R_g) vs one shared calibrator (fitted on R).",
    )
    parser.add_argument(
        "--employment",
        action="store_true",
        help="Run temporal drift and 2018 recalibration analysis on ACSEmployment (CA, 2014-2018).",
    )
    parser.add_argument(
        "--replot-extensions",
        action="store_true",
        help="Regenerate Monitoring, Group-vs-Shared, and ACSEmployment figures (Figures 6-8) from cached CSV outputs.",
    )
    parser.add_argument(
        "--drop-occp",
        action="store_true",
        help="Remove OCCP from ACSIncome features and read/write all outputs in outputs_no_occp/.",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Generate 2017 target-year and no-OCCP robustness comparison CSVs (05_robustness_2017_summary.csv, 05_robustness_no_occp_comparison.csv).",
    )
    return parser.parse_args()


# --- Part 1b: Load drift and recalibration CSV inputs ---
def load_drift_and_recalibration_outputs(test_mode: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    suffix = "_test" if test_mode else ""
    s3_path = os.path.join(OUTPUTS_DIR, f"03_drift_metrics_by_group{suffix}.csv")
    s4_runs_path = os.path.join(OUTPUTS_DIR, f"04_recalibration_runs{suffix}.csv")
    s4_sum_path = os.path.join(OUTPUTS_DIR, f"04_recalibration_summary{suffix}.csv")

    for p in (s3_path, s4_runs_path, s4_sum_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required input file not found: {p}")

    s3_df = pd.read_csv(s3_path, keep_default_na=False, na_values=[""])
    s4_runs = pd.read_csv(s4_runs_path, keep_default_na=False, na_values=[""])
    s4_sum = pd.read_csv(s4_sum_path, keep_default_na=False, na_values=[""])
    return s3_df, s4_runs, s4_sum


# --- Part 1c: Per-draw recovery percentages and paired method comparisons ---
def compute_per_draw_recovery(s4_runs: pd.DataFrame) -> pd.DataFrame:
    join_keys = ["model", "year", "fraction", "repeat", "group_type", "group"]
    none_df = (
        s4_runs[s4_runs["method"] == "None"][join_keys + ["ECE", "MeanGap"]]
        .rename(columns={"ECE": "ECE_None", "MeanGap": "MeanGap_None"})
    )

    merged = s4_runs.merge(none_df, on=join_keys, how="left")
    merged["delta_ece_vs_none"] = merged["ECE"] - merged["ECE_None"]
    merged["ece_recovery_pct"] = np.where(
        merged["ECE_None"] > 1e-9,
        100.0 * (merged["ECE_None"] - merged["ECE"]) / merged["ECE_None"],
        np.nan,
    )
    merged["gap_recovery_pct"] = np.where(
        merged["MeanGap_None"].abs() > 1e-9,
        100.0 * (merged["MeanGap_None"].abs() - merged["MeanGap"].abs()) / merged["MeanGap_None"].abs(),
        np.nan,
    )

    sum_keys = ["model", "year", "fraction", "method", "group_type", "group"]
    grp = merged.groupby(sum_keys, sort=False)

    rec_summary = grp["n_eval"].mean().rename("n_eval_mean").to_frame()
    rec_summary["small_group_flag"] = grp["small_group_flag"].any()
    for col in ["ECE", "Brier", "MeanGap", "ece_recovery_pct", "gap_recovery_pct", "delta_ece_vs_none"]:
        rec_summary[f"{col}_mean"] = grp[col].mean()
        rec_summary[f"{col}_p2.5"] = grp[col].quantile(0.025)
        rec_summary[f"{col}_p97.5"] = grp[col].quantile(0.975)

    return rec_summary.reset_index()


def compute_paired_method_comparisons(s4_runs: pd.DataFrame) -> pd.DataFrame:
    overall_runs = s4_runs[s4_runs["group_type"] == "Overall"].copy()
    piv = overall_runs.pivot_table(
        index=["model", "year", "fraction", "repeat"],
        columns="method",
        values="ECE",
    )

    pairs = [
        ("BaseRate", "None"),
        ("Platt", "None"),
        ("Beta", "None"),
        ("VennAbers", "None"),
        ("Isotonic", "None"),
        ("Platt", "BaseRate"),
        ("Beta", "BaseRate"),
        ("VennAbers", "BaseRate"),
        ("Isotonic", "BaseRate"),
        ("Isotonic", "Platt"),
        ("Isotonic", "Beta"),
        ("Isotonic", "VennAbers"),
        ("Beta", "Platt"),
        ("VennAbers", "Platt"),
    ]

    rows = []
    for m_name in MODEL_NAMES:
        for yr in EVAL_YEARS:
            for frac in FRACTIONS:
                if (m_name, yr, frac) not in piv.index:
                    continue
                sub = piv.loc[(m_name, yr, frac)]
                for m_a, m_b in pairs:
                    if m_a not in sub.columns or m_b not in sub.columns:
                        continue
                    diff = (sub[m_a] - sub[m_b]).dropna()
                    if len(diff) == 0:
                        continue
                    rows.append(
                        {
                            "model": m_name,
                            "year": int(yr),
                            "fraction": float(frac),
                            "method_a": m_a,
                            "method_b": m_b,
                            "comparison": f"{m_a} - {m_b}",
                            "n_draws": int(len(diff)),
                            "delta_ECE_mean": float(diff.mean()),
                            "delta_ECE_p2.5": float(np.percentile(diff, 2.5)),
                            "delta_ECE_p97.5": float(np.percentile(diff, 97.5)),
                            "a_better_or_equal_pct": float((diff <= 0.0).mean() * 100.0),
                        }
                    )
    return pd.DataFrame(rows)


# --- Part 2a: Build and save summary CSV tables (Tables 1-3 and paired comparisons) ---
def build_and_save_tables(
    s3_df: pd.DataFrame,
    rec_summary: pd.DataFrame,
    paired_df: pd.DataFrame,
    test_mode: bool = False,
) -> dict:
    ensure_dirs()
    suffix = "_test" if test_mode else ""
    paths = {}

    t1_df = (
        s3_df[s3_df["group_type"] == "Overall"]
        .drop(columns=["group_type", "group", "small_group_flag"])
        .copy()
    )
    t1_csv = os.path.join(OUTPUTS_DIR, f"05_table1_overall_metrics_by_year{suffix}.csv")
    t1_df.to_csv(t1_csv, index=False)
    paths["table1_csv"] = t1_csv

    s3_platt_grp = s3_df[
        (s3_df["calibration"] == "Platt")
        & (s3_df["model"].isin(GROUP_MODELS))
        & (s3_df["year"].isin([2014, 2018]))
    ].copy()

    s3_14 = s3_platt_grp[s3_platt_grp["year"] == 2014].set_index(["model", "group_type", "group"])
    s3_18 = s3_platt_grp[s3_platt_grp["year"] == 2018].set_index(["model", "group_type", "group"])

    rec_18_grp = rec_summary[
        (rec_summary["year"] == 2018) & (rec_summary["model"].isin(GROUP_MODELS))
    ].copy()

    t2_rows = []
    for _, r in rec_18_grp.iterrows():
        key = (r["model"], r["group_type"], r["group"])
        r14 = s3_14.loc[key]
        r18 = s3_18.loc[key]
        t2_rows.append(
            {
                "model": r["model"],
                "group_type": r["group_type"],
                "group": r["group"],
                "n_2014_test": int(r14["n"]),
                "small_2014_flag": bool(r14["small_group_flag"]),
                "ECE_2014_Platt": float(r14["ECE"]),
                "MeanGap_2014_Platt": float(r14["MeanGap"]),
                "n_2018_full": int(r18["n"]),
                "ECE_2018_Platt": float(r18["ECE"]),
                "MeanGap_2018_Platt": float(r18["MeanGap"]),
                "MeanGap_2018_low": float(r18["MeanGap_low"]),
                "MeanGap_2018_high": float(r18["MeanGap_high"]),
                "fraction": float(r["fraction"]),
                "method": r["method"],
                "n_eval_mean": float(r["n_eval_mean"]),
                "small_group_flag": bool(r["small_group_flag"]),
                "post_ECE_mean": float(r["ECE_mean"]),
                "post_ECE_p2.5": float(r["ECE_p2.5"]),
                "post_ECE_p97.5": float(r["ECE_p97.5"]),
                "post_MeanGap_mean": float(r["MeanGap_mean"]),
                "post_MeanGap_p2.5": float(r["MeanGap_p2.5"]),
                "post_MeanGap_p97.5": float(r["MeanGap_p97.5"]),
                "ece_recovery_pct_mean": float(r["ece_recovery_pct_mean"]),
                "ece_recovery_pct_p2.5": float(r["ece_recovery_pct_p2.5"]),
                "ece_recovery_pct_p97.5": float(r["ece_recovery_pct_p97.5"]),
            }
        )
    t2_df = pd.DataFrame(t2_rows)
    t2_csv = os.path.join(OUTPUTS_DIR, f"05_table2_subgroup_recalibration_results{suffix}.csv")
    t2_df.to_csv(t2_csv, index=False)
    paths["table2_csv"] = t2_csv

    t3_df = rec_summary[rec_summary["group_type"] == "Overall"].drop(
        columns=["group_type", "group", "small_group_flag"]
    ).copy()
    t3_csv = os.path.join(OUTPUTS_DIR, f"05_table3_method_sample_size_summary{suffix}.csv")
    t3_df.to_csv(t3_csv, index=False)

    paired_csv = os.path.join(OUTPUTS_DIR, f"05_paired_method_comparisons{suffix}.csv")
    paired_df.to_csv(paired_csv, index=False)

    paths["table3_csv"] = t3_csv
    paths["paired_csv"] = paired_csv
    return paths


# --- Part 2b: Core figures (Figures 1-5) ---
def make_05_fig1_positive_rate_by_year(
    yearly_dfs: dict, s3_df: pd.DataFrame, test_mode: bool = False
) -> str:
    ensure_dirs()
    suffix = "_test" if test_mode else ""

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=200, sharex=True)

    overall_rates = [yearly_dfs[y]["label"].mean() * 100.0 for y in YEARS]
    axes[0].plot(YEARS, overall_rates, marker="o", linewidth=2.2, color="#1f77b4")
    for x, y_val in zip(YEARS, overall_rates):
        axes[0].annotate(
            f"{y_val:.2f}%",
            (x, y_val),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8.5,
        )
    axes[0].set_title("(a) Overall Positive Rate", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Survey Year")
    axes[0].set_ylabel("Share Earning > $50,000 (%)")
    axes[0].set_xticks(YEARS)
    axes[0].set_xlim(2013.65, 2018.35)
    axes[0].set_ylim(min(overall_rates) - 2.0, max(overall_rates) + 2.8)
    axes[0].grid(True, linestyle="--", alpha=0.45)

    sex_colors = {"Male": "#1f77b4", "Female": "#e377c2"}
    for s_grp in SEX_GROUPS:
        rates = [
            yearly_dfs[y].loc[yearly_dfs[y]["sex_group"] == s_grp, "label"].mean() * 100.0
            for y in YEARS
        ]
        axes[1].plot(YEARS, rates, marker="s", linewidth=2.0, color=sex_colors[s_grp], label=s_grp)
    axes[1].set_title("(b) Positive Rate by Sex", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Survey Year")
    axes[1].set_ylabel("Share Earning > $50,000 (%)")
    axes[1].set_xticks(YEARS)
    axes[1].set_xlim(2013.65, 2018.35)
    axes[1].grid(True, linestyle="--", alpha=0.45)
    axes[1].legend(loc="upper left", fontsize=9)

    race_colors = {
        "White alone": "#1f77b4",
        "Asian alone": "#2ca02c",
        "Some Other Race alone": "#ff7f0e",
        "Black alone": "#d62728",
        "Two or More Races": "#9467bd",
        MERGED_RACE_NAME: "#8c564b",
    }
    for r_grp in RACE_GROUPS:
        rates = [
            yearly_dfs[y].loc[yearly_dfs[y]["race_group"] == r_grp, "label"].mean() * 100.0
            for y in YEARS
        ]
        axes[2].plot(YEARS, rates, marker="^", markersize=4.5, linewidth=1.7, color=race_colors[r_grp], label=r_grp)
    axes[2].set_title("(c) Positive Rate by Race", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Survey Year")
    axes[2].set_ylabel("Share Earning > $50,000 (%)")
    axes[2].set_xticks(YEARS)
    axes[2].set_xlim(2013.65, 2018.35)
    axes[2].set_ylim(2.0, 50.5)
    axes[2].grid(True, linestyle="--", alpha=0.45)
    axes[2].legend(loc="lower right", ncol=2, fontsize=7.0, framealpha=0.92)

    fig.suptitle(
        "Figure 1: ACSIncome (CA, 2014–2018, Full Census Years, No Repeats) — Positive Rate by Year, Sex, and Race",
        fontsize=12.0,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path = os.path.join(OUTPUTS_DIR, f"05_fig1_positive_rate_by_year{suffix}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def make_05_fig2_reliability_2014_vs_2018(
    yearly_dfs: dict, test_mode: bool = False, drop_occp: bool = False
) -> str:
    ensure_dirs(drop_occp=drop_occp)
    suffix = "_test" if test_mode else ""
    cache_path = get_predictions_cache_path(test_mode=test_mode, drop_occp=drop_occp)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cached baseline predictions not found at {cache_path}")
    with open(cache_path, "rb") as f:
        predictions = pickle.load(f)

    _, _, test_2014 = split_2014_data(yearly_dfs[2014])
    y_2014 = test_2014["label"].to_numpy(dtype=int)
    y_2018 = yearly_dfs[2018]["label"].to_numpy(dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), dpi=200, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    styles = [
        ("Uncalibrated", 2014, "#1f77b4", "--", "o", "2014 Uncalibrated"),
        ("Platt", 2014, "#1f77b4", "-", "s", "2014 Platt"),
        ("Uncalibrated", 2018, "#d62728", "--", "^", "2018 Uncalibrated"),
        ("Platt", 2018, "#d62728", "-", "D", "2018 Platt"),
    ]

    for ax, m_name in zip(axes_flat, MODEL_NAMES):
        ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.2, label="Perfect Calibration")
        for cal_name, yr, color, ls, marker, label in styles:
            y_true = y_2014 if yr == 2014 else y_2018
            p_pred = predictions[m_name][cal_name][yr]
            rel = compute_reliability_bins(y_true, p_pred)
            valid = ~np.isnan(rel["mean_pred"])
            ax.plot(
                rel["mean_pred"][valid],
                rel["mean_actual"][valid],
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=5,
                linewidth=1.6,
                label=label,
            )
        ax.set_title(m_name, fontsize=11, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.45)

    for ax in axes[1, :]:
        ax.set_xlabel("Mean Predicted Probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Empirical Positive Rate")

    axes_flat[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(
        "Figure 2: ACSIncome (CA, 2014 Test Slice vs. 2018 Full Year, Single 60/20/20 Split) — Reliability Diagrams",
        fontsize=12.0,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path = os.path.join(OUTPUTS_DIR, f"05_fig2_reliability_2014_vs_2018{suffix}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def make_05_fig3_ece_by_year_per_group(s3_df: pd.DataFrame, test_mode: bool = False) -> str:
    ensure_dirs()
    suffix = "_test" if test_mode else ""
    platt_df = s3_df[
        (s3_df["calibration"] == "Platt") & (s3_df["model"].isin(GROUP_MODELS))
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), dpi=200, sharex=True, sharey=True)

    sex_colors = {"All": "#333333", "Male": "#1f77b4", "Female": "#e377c2"}
    race_colors = {
        "White alone": "#1f77b4",
        "Asian alone": "#2ca02c",
        "Some Other Race alone": "#ff7f0e",
        "Black alone": "#d62728",
        "Two or More Races": "#9467bd",
        MERGED_RACE_NAME: "#8c564b",
    }

    for col_idx, m_name in enumerate(GROUP_MODELS):
        m_sub = platt_df[platt_df["model"] == m_name]

        ax_sex = axes[0, col_idx]
        for g_type, g_name, ls in [("Overall", "All", "--"), ("Sex", "Male", "-"), ("Sex", "Female", "-")]:
            g_sub = m_sub[(m_sub["group_type"] == g_type) & (m_sub["group"] == g_name)].sort_values("year")
            yrs = g_sub["year"].to_numpy()
            ece = g_sub["ECE"].to_numpy()
            low = g_sub["ECE_low"].to_numpy()
            high = g_sub["ECE_high"].to_numpy()
            lbl = "Overall" if g_name == "All" else g_name
            ax_sex.plot(yrs, ece, marker="o", linestyle=ls, linewidth=1.8, color=sex_colors[g_name], label=lbl)
            ax_sex.fill_between(yrs, low, high, color=sex_colors[g_name], alpha=0.16)
        ax_sex.set_title(f"{m_name}\n(By Sex & Overall)", fontsize=11, fontweight="bold")
        ax_sex.set_xticks(YEARS)
        ax_sex.grid(True, linestyle="--", alpha=0.45)

        ax_race = axes[1, col_idx]
        for r_grp in RACE_GROUPS:
            g_sub = m_sub[(m_sub["group_type"] == "Race") & (m_sub["group"] == r_grp)].sort_values("year")
            yrs = g_sub["year"].to_numpy()
            ece = g_sub["ECE"].to_numpy()
            low = g_sub["ECE_low"].to_numpy()
            high = g_sub["ECE_high"].to_numpy()
            ax_race.plot(yrs, ece, marker="s", markersize=4, linewidth=1.6, color=race_colors[r_grp], label=r_grp)
            ax_race.fill_between(yrs, low, high, color=race_colors[r_grp], alpha=0.12)
        ax_race.set_title("(By Race)", fontsize=10.5)
        ax_race.set_xlabel("Evaluation Year")
        ax_race.set_xticks(YEARS)
        ax_race.grid(True, linestyle="--", alpha=0.45)

    axes[0, 0].set_ylabel("10-Bin ECE (with 95% Bootstrap CI)")
    axes[1, 0].set_ylabel("10-Bin ECE (with 95% Bootstrap CI)")
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    axes[1, 0].legend(loc="upper left", fontsize=7.5)

    fig.suptitle(
        "Figure 3: ACSIncome (CA, 2014–2018, 2014-Platt, 200 Bootstrap Resamples) — Subgroup ECE by Year",
        fontsize=12.0,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path = os.path.join(OUTPUTS_DIR, f"05_fig3_ece_by_year_per_group{suffix}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def make_05_fig4_ece_vs_sample_size(s4_sum: pd.DataFrame, test_mode: bool = False) -> str:
    ensure_dirs()
    suffix = "_test" if test_mode else ""
    ov_2018 = s4_sum[
        (s4_sum["group_type"] == "Overall") & (s4_sum["year"] == 2018)
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), dpi=200, sharex=True, sharey=False)
    axes_flat = axes.flatten()
    frac_pct = [f * 100.0 for f in FRACTIONS]

    for ax, m_name in zip(axes_flat, MODEL_NAMES):
        m_sub = ov_2018[ov_2018["model"] == m_name]
        for method in METHODS:
            ms = m_sub[m_sub["method"] == method].sort_values("fraction")
            x_vals = ms["fraction"].to_numpy() * 100.0
            y_mean = ms["ECE_mean"].to_numpy()
            y_low = ms["ECE_p2.5"].to_numpy()
            y_high = ms["ECE_p97.5"].to_numpy()

            ls = "--" if method == "None" else "-"
            ax.plot(
                x_vals,
                y_mean,
                marker=METHOD_MARKERS[method],
                linestyle=ls,
                linewidth=1.8,
                color=METHOD_COLORS[method],
                label=method,
            )
            if method != "None":
                ax.fill_between(x_vals, y_low, y_high, color=METHOD_COLORS[method], alpha=0.14)

        ax.set_title(m_name, fontsize=11, fontweight="bold")
        ax.set_xticks(frac_pct)
        ax.set_xticklabels([f"{int(p)}%" for p in frac_pct])
        ax.grid(True, linestyle="--", alpha=0.45)

    for ax in axes[1, :]:
        ax.set_xlabel("Recalibration Sample Size (% of 2018 Data)")
    for ax in axes[:, 0]:
        ax.set_ylabel("2018 Evaluation ECE (Mean & 95% Interval)")

    axes_flat[0].legend(loc="upper right", fontsize=8.5, ncol=2)
    fig.suptitle(
        "Figure 4: ACSIncome (CA, 2018, 30 Repeats) — Overall Evaluation ECE vs. Recalibration Sample Size",
        fontsize=12.0,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path = os.path.join(OUTPUTS_DIR, f"05_fig4_ece_vs_sample_size{suffix}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def make_05_fig5_recovery_by_group(rec_summary: pd.DataFrame, test_mode: bool = False) -> str:
    ensure_dirs()
    suffix = "_test" if test_mode else ""
    active_methods = ["BaseRate", "Platt", "Beta", "VennAbers", "Isotonic"]
    display_groups = ["All", "Male", "Female"] + RACE_GROUPS
    short_labels = [
        "Overall",
        "Male",
        "Female",
        "White",
        "Asian",
        "Some Other",
        "Black",
        "Two+ Races",
        "Other (mrg)",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6), dpi=200, sharex=True, sharey=True)

    for row_idx, frac_val in enumerate([0.01, 0.05]):
        for col_idx, m_name in enumerate(GROUP_MODELS):
            ax = axes[row_idx, col_idx]
            sub = rec_summary[
                (rec_summary["year"] == 2018)
                & (rec_summary["model"] == m_name)
                & (rec_summary["fraction"] == frac_val)
            ]

            x_base = np.arange(len(display_groups))
            bar_width = 0.15
            offsets = np.linspace(-2 * bar_width, 2 * bar_width, len(active_methods))

            ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
            ax.axhline(100.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)

            for m_i, method in enumerate(active_methods):
                m_sub = sub[sub["method"] == method].set_index("group")
                means = [float(m_sub.loc[g, "ece_recovery_pct_mean"]) for g in display_groups]
                lows = [float(m_sub.loc[g, "ece_recovery_pct_p2.5"]) for g in display_groups]
                highs = [float(m_sub.loc[g, "ece_recovery_pct_p97.5"]) for g in display_groups]
                means_arr = np.array(means)
                lows_arr = np.maximum(-123.0, np.array(lows))
                highs_arr = np.minimum(108.0, np.array(highs))
                yerr_low = np.maximum(0.0, means_arr - lows_arr)
                yerr_high = np.maximum(0.0, highs_arr - means_arr)

                ax.bar(
                    x_base + offsets[m_i],
                    means_arr,
                    width=bar_width,
                    color=METHOD_COLORS[method],
                    alpha=0.86,
                    label=method if (row_idx == 0 and col_idx == 0) else None,
                    yerr=[yerr_low, yerr_high],
                    error_kw={"elinewidth": 0.85, "ecolor": "#222222", "capsize": 1.6},
                )

            if m_name == "Random Forest" and OUTPUTS_DIR == "outputs" and not test_mode:
                if frac_val == 0.01:
                    ax.annotate(
                        "Other (mrg):\n-83% to -105%\n(CI to -180%)",
                        xy=(8.0, -104.6),
                        xytext=(5.7, -116.0),
                        fontsize=7.5,
                        fontweight="bold",
                        color="#b22222",
                        arrowprops={"arrowstyle": "->", "color": "#b22222", "lw": 0.9},
                    )
                elif frac_val == 0.05:
                    ax.annotate(
                        "Other (mrg):\n-69% to -95%\n(CI to -131%)",
                        xy=(8.0, -95.1),
                        xytext=(5.7, -114.0),
                        fontsize=7.5,
                        fontweight="bold",
                        color="#b22222",
                        arrowprops={"arrowstyle": "->", "color": "#b22222", "lw": 0.9},
                    )

            pct_str = f"{int(frac_val * 100)}%"
            if row_idx == 0:
                ax.set_title(f"{m_name}\nSample Size f = {pct_str} (~1,957 rows)", fontsize=10.5, fontweight="bold")
            else:
                ax.set_title(f"{m_name} — Sample Size f = {pct_str} (~9,783 rows)", fontsize=10, fontweight="bold")

            ax.set_xticks(x_base)
            ax.set_xticklabels(short_labels, rotation=32, ha="right", fontsize=8.5)
            ax.set_ylim(-125.0, 110.0)
            ax.grid(True, axis="y", linestyle="--", alpha=0.45)
            if row_idx == 1:
                ax.set_xlabel("Demographic Subgroup")

    axes[0, 0].set_ylabel("% of 2018 ECE Drift Removed (f = 1%)")
    axes[1, 0].set_ylabel("% of 2018 ECE Drift Removed (f = 5%)")
    axes[0, 0].legend(loc="lower left", fontsize=8.5, ncol=3, title="Recalibration Method", title_fontsize=8.5)

    fig.suptitle(
        "Figure 5: ACSIncome (CA, 2018, 30 Repeats) — Subgroup Calibration Drift Recovery (% of 2018 ECE Removed, Mean & 95% Interval)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.0, 0.055, 1.0, 0.96])
    note_text = "Note: Recovery % = 100 * (ECE_None - ECE_method) / ECE_None on disjoint evaluation rows (30 repeats). Negative values mean global recalibration increased subgroup ECE."
    if OUTPUTS_DIR == "outputs" and not test_mode:
        note_text += "\nFor Random Forest on Other (merged), mean recovery is -83.3% to -104.6% at f=1% (95% CI down to -180.2%) and -68.8% to -95.1% at f=5% (95% CI down to -131.2%). Gaussian NB excluded."
    fig.text(
        0.5,
        0.012,
        note_text,
        ha="center",
        va="bottom",
        fontsize=8.5,
        style="italic",
    )

    out_path = os.path.join(OUTPUTS_DIR, f"05_fig5_recovery_by_group{suffix}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# --- Part 2c: Console summary for core results ---
def print_analysis_summary(rec_summary: pd.DataFrame, paired_df: pd.DataFrame):
    print("=" * 114)
    print("ANALYSIS SUMMARY: 2018 OVERALL DRIFT RECOVERY (% OF 2018 ECE DRIFT REMOVED, MEAN [p2.5, p97.5])")
    print("=" * 114)
    ov18 = rec_summary[(rec_summary["group_type"] == "Overall") & (rec_summary["year"] == 2018)]
    for m_name in MODEL_NAMES:
        print(f"\nModel: {m_name}")
        print(
            f"{'Method':<12s} {'1% Sample':>24s} {'2% Sample':>24s} "
            f"{'5% Sample':>24s} {'10% Sample':>24s}"
        )
        print("-" * 112)
        m_sub = ov18[ov18["model"] == m_name]
        for method in ["BaseRate", "Platt", "Beta", "VennAbers", "Isotonic"]:
            cells = []
            for frac in FRACTIONS:
                r = m_sub[(m_sub["method"] == method) & (m_sub["fraction"] == frac)].iloc[0]
                cells.append(
                    f"{r['ece_recovery_pct_mean']:+5.1f}% [{r['ece_recovery_pct_p2.5']:+5.1f},{r['ece_recovery_pct_p97.5']:+5.1f}]"
                )
            print(
                f"{method:<12s} {cells[0]:>24s} {cells[1]:>24s} "
                f"{cells[2]:>24s} {cells[3]:>24s}"
            )

    print("\n" + "=" * 114)
    print("PAIRED PER-DRAW ECE COMPARISONS IN 2018 (Delta_ECE = ECE_A - ECE_B on identical sample R)")
    print("=" * 114)
    p18 = paired_df[
        (paired_df["year"] == 2018)
        & (paired_df["comparison"].isin(["BaseRate - None", "Platt - BaseRate", "Beta - BaseRate", "VennAbers - BaseRate", "Isotonic - Platt", "Isotonic - VennAbers"]))
    ]
    print(
        f"{'Model':<24s} {'Frac':>5s} {'Comparison':<22s} "
        f"{'Mean dECE':>10s} {'95% CI [p2.5, p97.5]':>24s} {'% A<=B':>8s}"
    )
    print("-" * 98)
    for _, r in p18.iterrows():
        ci_s = f"[{r['delta_ECE_p2.5']:+.4f}, {r['delta_ECE_p97.5']:+.4f}]"
        print(
            f"{r['model']:<24s} {r['fraction']*100:>4.0f}% {r['comparison']:<22s} "
            f"{r['delta_ECE_mean']:>+10.4f} {ci_s:>24s} {r['a_better_or_equal_pct']:>7.1f}%"
        )

    print("\n" + "=" * 114)
    print("SUBGROUP RECOVERY IN 2018 AT 1% vs 5% SAMPLE SIZE (EXCLUDING GAUSSIAN NB)")
    print("=" * 114)
    grp18 = rec_summary[
        (rec_summary["year"] == 2018)
        & (rec_summary["model"].isin(GROUP_MODELS))
        & (rec_summary["method"].isin(["BaseRate", "Platt", "Isotonic", "VennAbers"]))
    ]
    for m_name in GROUP_MODELS:
        print(f"\nModel: {m_name}")
        print(
            f"{'Group':<24s} {'n_eval(1%)':>10s} "
            f"{'BaseRate 1%':>14s} {'Platt 1%':>14s} {'Isotonic 1%':>14s} {'VennAbers 1%':>14s} | "
            f"{'BaseRate 5%':>14s} {'Platt 5%':>14s} {'Isotonic 5%':>14s}"
        )
        print("-" * 138)
        m_sub = grp18[grp18["model"] == m_name]
        for g_name in ["All", "Male", "Female"] + RACE_GROUPS:
            g_sub = m_sub[m_sub["group"] == g_name]
            n1 = g_sub[g_sub["fraction"] == 0.01]["n_eval_mean"].iloc[0]
            flag = "*" if g_sub["small_group_flag"].any() else " "

            def cell(f_val: float, meth: str) -> str:
                row = g_sub[(g_sub["fraction"] == f_val) & (g_sub["method"] == meth)].iloc[0]
                return f"{row['ece_recovery_pct_mean']:+5.1f}% ({row['ECE_mean']:.3f})"

            print(
                f"{g_name + flag:<24s} {n1:>10,.0f} "
                f"{cell(0.01, 'BaseRate'):>14s} {cell(0.01, 'Platt'):>14s} "
                f"{cell(0.01, 'Isotonic'):>14s} {cell(0.01, 'VennAbers'):>14s} | "
                f"{cell(0.05, 'BaseRate'):>14s} {cell(0.05, 'Platt'):>14s} "
                f"{cell(0.05, 'Isotonic'):>14s}"
            )


# --- Part 3a: Monitoring rule computation (bootstrap lower bound on R) ---
MONITOR_THRESHOLDS = [0.00, 0.005, 0.01, 0.015, 0.02, 0.03]
MONITOR_METHODS = ["BaseRate", "Platt"]


def compute_monitoring(
    yearly_dfs: dict,
    s4_runs: pd.DataFrame,
    test_mode: bool = False,
    drop_occp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_repeats = N_REPEATS_TEST if test_mode else N_REPEATS_FULL
    eval_dfs, predictions = train_and_predict_all_years(
        yearly_dfs, test_mode=test_mode, use_cache=True, drop_occp=drop_occp
    )

    r_stats_rows = []
    for year in EVAL_YEARS:
        y_all = eval_dfs[year]["label"].to_numpy(dtype=int)
        n_rows = len(y_all)
        for fraction in FRACTIONS:
            frac_pct = int(round(fraction * 100))
            for rep in range(n_repeats):
                in_r = draw_recalibration_sample(n_rows, year, fraction, rep)
                y_r = y_all[in_r]
                n_r = int(in_r.sum())

                probs_r = {m: predictions[m]["Platt"][year][in_r] for m in MODEL_NAMES}
                boot_seed = [SEED, year, frac_pct, rep, 1]
                boot_r = bootstrap_ece_and_gap_multi(
                    y_r, probs_r, n_boot=N_BOOT, seed=boot_seed
                )

                for m_name in MODEL_NAMES:
                    p_r = probs_r[m_name]
                    est_ece = compute_ece(y_r, p_r)
                    est_gap = compute_mean_gap(y_r, p_r)
                    b_info = boot_r[m_name]
                    g_low = b_info["gap_low"]
                    g_high = b_info["gap_high"]

                    if g_low * g_high > 0.0:
                        gap_lb = float(min(abs(g_low), abs(g_high)))
                    else:
                        gap_lb = 0.0

                    row_dict = {
                        "model": m_name,
                        "year": year,
                        "fraction": fraction,
                        "repeat": rep,
                        "n_r": n_r,
                        "est_ECE_R": est_ece,
                        "est_ECE_low_R": b_info["ece_low"],
                        "est_ECE_high_R": b_info["ece_high"],
                        "est_MeanGap_R": est_gap,
                        "est_MeanGap_low_R": g_low,
                        "est_MeanGap_high_R": g_high,
                        "gap_lower_bound_R": gap_lb,
                    }
                    for tau in MONITOR_THRESHOLDS:
                        row_dict[f"trig_{tau:.3f}"] = bool(gap_lb > tau)
                    r_stats_rows.append(row_dict)

    r_stats_df = pd.DataFrame(r_stats_rows)

    s4_sub = s4_runs[s4_runs["method"].isin(["None"] + MONITOR_METHODS)].copy()
    idx_cols = ["model", "year", "fraction", "repeat", "group_type", "group"]
    piv_ece = s4_sub.pivot_table(index=idx_cols + ["n_eval", "small_group_flag"], columns="method", values="ECE").reset_index()
    piv_gap = s4_sub.pivot_table(index=idx_cols, columns="method", values="MeanGap").reset_index()
    piv_brier = s4_sub.pivot_table(index=idx_cols, columns="method", values="Brier").reset_index()

    eval_merged = (
        piv_ece.merge(piv_gap, on=idx_cols, suffixes=("_ECE", "_MeanGap"))
        .merge(
            piv_brier.rename(columns={m: f"{m}_Brier" for m in ["None"] + MONITOR_METHODS}),
            on=idx_cols,
        )
        .merge(r_stats_df, on=["model", "year", "fraction", "repeat"], how="left")
    )

    policy_records = []
    for rec_method in MONITOR_METHODS:
        ece_always = eval_merged[f"{rec_method}_ECE"].to_numpy()
        ece_never = eval_merged["None_ECE"].to_numpy()
        gap_always = eval_merged[f"{rec_method}_MeanGap"].to_numpy()
        gap_never = eval_merged["None_MeanGap"].to_numpy()
        brier_always = eval_merged[f"{rec_method}_Brier"].to_numpy()
        brier_never = eval_merged["None_Brier"].to_numpy()

        policies = [("Never (None)", np.zeros(len(eval_merged), dtype=bool), np.nan)]
        for tau in MONITOR_THRESHOLDS:
            policies.append((f"Monitor (tau={tau:.3f})", eval_merged[f"trig_{tau:.3f}"].to_numpy(dtype=bool), tau))
        policies.append(("Always Recalibrate", np.ones(len(eval_merged), dtype=bool), -1.0))

        for pol_name, trig_mask, tau_val in policies:
            pol_ece = np.where(trig_mask, ece_always, ece_never)
            pol_gap = np.where(trig_mask, gap_always, gap_never)
            pol_brier = np.where(trig_mask, brier_always, brier_never)
            delta_ece_vs_always = pol_ece - ece_always
            delta_ece_vs_never = pol_ece - ece_never

            df_pol = eval_merged[idx_cols + ["n_eval", "small_group_flag"]].copy()
            df_pol["recal_method"] = rec_method
            df_pol["policy"] = pol_name
            df_pol["threshold"] = tau_val
            df_pol["triggered"] = trig_mask.astype(float)
            df_pol["ECE"] = pol_ece
            df_pol["Brier"] = pol_brier
            df_pol["MeanGap"] = pol_gap
            df_pol["delta_ECE_vs_Always"] = delta_ece_vs_always
            df_pol["delta_ECE_vs_Never"] = delta_ece_vs_never
            policy_records.append(df_pol)

    all_pol_runs = pd.concat(policy_records, ignore_index=True)

    g_keys = ["recal_method", "model", "year", "fraction", "policy", "group_type", "group"]
    grp = all_pol_runs.groupby(g_keys, sort=False, dropna=False)
    pol_summary = grp["n_eval"].mean().rename("n_eval_mean").to_frame()
    pol_summary["small_group_flag"] = grp["small_group_flag"].any()
    pol_summary["trigger_rate_pct"] = grp["triggered"].mean() * 100.0
    for col in ["ECE", "Brier", "MeanGap", "delta_ECE_vs_Always", "delta_ECE_vs_Never"]:
        pol_summary[f"{col}_mean"] = grp[col].mean()
        pol_summary[f"{col}_p2.5"] = grp[col].quantile(0.025)
        pol_summary[f"{col}_p97.5"] = grp[col].quantile(0.975)
    pol_summary = pol_summary.reset_index()

    ov_runs = all_pol_runs[all_pol_runs["group_type"] == "Overall"].copy()
    multi_yr_draws = (
        ov_runs.groupby(["recal_method", "model", "fraction", "policy", "repeat"], sort=False)
        .agg(
            recalibrations_triggered=("triggered", "sum"),
            trigger_rate_pct=("triggered", lambda s: s.mean() * 100.0),
            ECE=("ECE", "mean"),
            Brier=("Brier", "mean"),
            MeanGap=("MeanGap", "mean"),
            delta_ECE_vs_Always=("delta_ECE_vs_Always", "mean"),
            delta_ECE_vs_Never=("delta_ECE_vs_Never", "mean"),
        )
        .reset_index()
    )
    m_grp = multi_yr_draws.groupby(["recal_method", "model", "fraction", "policy"], sort=False)
    multi_yr_summary = m_grp["recalibrations_triggered"].mean().rename("avg_years_recalibrated").to_frame()
    multi_yr_summary["trigger_rate_pct"] = m_grp["trigger_rate_pct"].mean()
    for col in ["ECE", "Brier", "MeanGap", "delta_ECE_vs_Always", "delta_ECE_vs_Never"]:
        multi_yr_summary[f"{col}_mean"] = m_grp[col].mean()
        multi_yr_summary[f"{col}_p2.5"] = m_grp[col].quantile(0.025)
        multi_yr_summary[f"{col}_p97.5"] = m_grp[col].quantile(0.975)
    multi_yr_summary = multi_yr_summary.reset_index()

    return r_stats_df, pol_summary, multi_yr_summary


# --- Part 3b: Save monitoring CSVs and plot Figure 6 ---
def save_and_plot_monitoring(
    r_stats_df: pd.DataFrame,
    pol_summary: pd.DataFrame,
    multi_yr_summary: pd.DataFrame,
    test_mode: bool = False,
) -> dict:
    ensure_dirs()
    suffix = "_test" if test_mode else ""

    runs_csv = os.path.join(OUTPUTS_DIR, f"05_monitoring_runs{suffix}.csv")
    sum_csv = os.path.join(OUTPUTS_DIR, f"05_monitoring_summary{suffix}.csv")
    multi_csv = os.path.join(OUTPUTS_DIR, f"05_monitoring_4year_policy_summary{suffix}.csv")

    r_stats_df.to_csv(runs_csv, index=False)
    pol_summary.to_csv(sum_csv, index=False)
    multi_yr_summary.to_csv(multi_csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=200)
    colors = {
        "Logistic Regression": "#1f77b4",
        "Random Forest": "#ff7f0e",
        "Gradient Boosted Trees": "#2ca02c",
        "Gaussian Naive Bayes": "#d62728",
    }

    r_1pct = r_stats_df[r_stats_df["fraction"] == 0.01]
    for m_name in MODEL_NAMES:
        m_sub = r_1pct[r_1pct["model"] == m_name].groupby("year")["gap_lower_bound_R"]
        means = m_sub.mean()
        lows = m_sub.quantile(0.025)
        highs = m_sub.quantile(0.975)
        yrs = means.index.to_numpy()
        axes[0].plot(yrs, means.to_numpy(), marker="o", linewidth=1.8, color=colors[m_name], label=m_name)
        axes[0].fill_between(yrs, lows.to_numpy(), highs.to_numpy(), color=colors[m_name], alpha=0.14)
    for tau_line, ls_tau in [(0.01, "--"), (0.02, ":")]:
        axes[0].axhline(tau_line, color="black", linestyle=ls_tau, linewidth=1.0, label=f"Threshold τ = {tau_line:.2f}")
    axes[0].set_title("(a) 95% Lower Bound of |MeanGap| on R (f = 1%)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Target Year")
    axes[0].set_ylabel("Lower Bound L_gap(R)")
    axes[0].set_xticks(EVAL_YEARS)
    axes[0].grid(True, linestyle="--", alpha=0.45)
    axes[0].legend(loc="upper left", fontsize=7.5)

    lr_ov = pol_summary[
        (pol_summary["recal_method"] == "BaseRate")
        & (pol_summary["model"] == "Logistic Regression")
        & (pol_summary["group_type"] == "Overall")
        & (pol_summary["fraction"] == 0.01)
    ]
    tau_styles = [
        ("Monitor (tau=0.000)", "#1f77b4", "o", "τ = 0.00 (CI excludes 0)"),
        ("Monitor (tau=0.005)", "#2ca02c", "s", "τ = 0.005"),
        ("Monitor (tau=0.010)", "#ff7f0e", "^", "τ = 0.010"),
        ("Monitor (tau=0.015)", "#9467bd", "D", "τ = 0.015"),
        ("Monitor (tau=0.020)", "#d62728", "v", "τ = 0.020"),
        ("Monitor (tau=0.030)", "#8c564b", "P", "τ = 0.030"),
    ]
    for pol_name, col, mk, lbl in tau_styles:
        ps = lr_ov[lr_ov["policy"] == pol_name].sort_values("year")
        axes[1].plot(ps["year"], ps["trigger_rate_pct"], marker=mk, linewidth=1.8, color=col, label=lbl)
    axes[1].set_title("(b) Trigger Rate by Year (LR, f = 1%)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Target Year")
    axes[1].set_ylabel("Recalibration Trigger Rate (%)")
    axes[1].set_xticks(EVAL_YEARS)
    axes[1].set_ylim(-5.0, 105.0)
    axes[1].grid(True, linestyle="--", alpha=0.45)
    axes[1].legend(loc="lower right", fontsize=7.8)

    m_1pct = multi_yr_summary[
        (multi_yr_summary["recal_method"] == "BaseRate")
        & (multi_yr_summary["fraction"] == 0.01)
    ]
    for m_name in MODEL_NAMES:
        ms = m_1pct[m_1pct["model"] == m_name].sort_values("avg_years_recalibrated")
        axes[2].plot(
            ms["avg_years_recalibrated"],
            ms["ECE_mean"],
            marker="o",
            linewidth=1.8,
            color=colors[m_name],
            label=m_name,
        )
        for _, r in ms[ms["policy"].isin(["Never (None)", "Monitor (tau=0.010)", "Always Recalibrate"])].iterrows():
            short_p = "Never" if "Never" in r["policy"] else ("Always" if "Always" in r["policy"] else "τ=0.01")
            if m_name == "Logistic Regression":
                axes[2].annotate(
                    short_p,
                    (r["avg_years_recalibrated"], r["ECE_mean"]),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7.5,
                )
    axes[2].set_title("(c) 4-Yr Avg ECE vs. Recalibrations (f = 1%)", fontsize=10.5, fontweight="bold")
    axes[2].set_xlabel("Avg. Years Recalibrated out of 4 (2015–2018)")
    axes[2].set_ylabel("2015–2018 Mean Evaluation ECE")
    axes[2].set_xlim(-0.25, 4.25)
    axes[2].grid(True, linestyle="--", alpha=0.45)
    axes[2].legend(loc="upper right", fontsize=7.5)

    fig.suptitle(
        "Figure 6: ACSIncome (CA, 2015–2018, 30 Repeats) — Monitoring Rule vs. Recalibrate Every Year",
        fontsize=12.0,
        fontweight="bold",
    )
    fig.tight_layout()

    fig_path = os.path.join(OUTPUTS_DIR, f"05_fig6_monitoring{suffix}.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return {
        "runs_csv": runs_csv,
        "summary_csv": sum_csv,
        "multi_csv": multi_csv,
        "fig_path": fig_path,
    }


def print_monitoring_console(pol_summary: pd.DataFrame, multi_yr_summary: pd.DataFrame):
    print("\n" + "=" * 124)
    print("MONITORING RULE vs. RECALIBRATE EVERY YEAR (2015-2018 4-YEAR AVERAGE, BASERATE RECALIBRATOR)")
    print("=" * 124)
    for frac_val in [0.01, 0.02, 0.05]:
        print(f"\n--- Sample Size f = {frac_val*100:.0f}% ---")
        print(
            f"{'Model':<24s} {'Policy':<22s} {'Avg Yrs Recal':>13s} {'Trig %':>8s} "
            f"{'4-Yr Avg ECE [95% CI]':>26s} {'Paired dECE vs Always [95% CI]':>32s}"
        )
        print("-" * 130)
        sub = multi_yr_summary[
            (multi_yr_summary["recal_method"] == "BaseRate")
            & (multi_yr_summary["fraction"] == frac_val)
        ]
        for m_name in MODEL_NAMES:
            m_sub = sub[sub["model"] == m_name]
            for _, r in m_sub.iterrows():
                ece_s = f"{r['ECE_mean']:.4f} [{r['ECE_p2.5']:.4f}, {r['ECE_p97.5']:.4f}]"
                del_s = f"{r['delta_ECE_vs_Always_mean']:+.4f} [{r['delta_ECE_vs_Always_p2.5']:+.4f}, {r['delta_ECE_vs_Always_p97.5']:+.4f}]"
                print(
                    f"{r['model']:<24s} {r['policy']:<22s} {r['avg_years_recalibrated']:>13.2f} "
                    f"{r['trigger_rate_pct']:>7.1f}% {ece_s:>26s} {del_s:>32s}"
                )

    print("\n" + "=" * 124)
    print("YEAR-BY-YEAR TRIGGER RATE (%) AND EVAL ECE AT f = 1% (BASERATE RECALIBRATOR)")
    print("=" * 124)
    ov1 = pol_summary[
        (pol_summary["recal_method"] == "BaseRate")
        & (pol_summary["group_type"] == "Overall")
        & (pol_summary["fraction"] == 0.01)
        & (pol_summary["policy"].isin(["Never (None)", "Monitor (tau=0.000)", "Monitor (tau=0.010)", "Monitor (tau=0.020)", "Always Recalibrate"]))
    ]
    for m_name in GROUP_MODELS:
        print(f"\nModel: {m_name} (f = 1%)")
        print(
            f"{'Policy':<22s} {'2015 Trig% (ECE)':>20s} {'2016 Trig% (ECE)':>20s} "
            f"{'2017 Trig% (ECE)':>20s} {'2018 Trig% (ECE)':>20s}"
        )
        print("-" * 106)
        m_sub = ov1[ov1["model"] == m_name]
        for pol_name in ["Never (None)", "Monitor (tau=0.000)", "Monitor (tau=0.010)", "Monitor (tau=0.020)", "Always Recalibrate"]:
            p_sub = m_sub[m_sub["policy"] == pol_name].set_index("year")
            cells = [
                f"{p_sub.loc[y, 'trigger_rate_pct']:5.1f}% ({p_sub.loc[y, 'ECE_mean']:.4f})"
                for y in EVAL_YEARS
            ]
            print(
                f"{pol_name:<22s} {cells[0]:>20s} {cells[1]:>20s} "
                f"{cells[2]:>20s} {cells[3]:>20s}"
            )


# --- Part 4a: Group-specific calibrators vs shared calibrator ---
def compute_group_vs_shared(
    yearly_dfs: dict,
    s4_runs: pd.DataFrame,
    test_mode: bool = False,
    drop_occp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    n_repeats = N_REPEATS_TEST if test_mode else N_REPEATS_FULL
    eval_dfs, predictions = train_and_predict_all_years(
        yearly_dfs, test_mode=test_mode, use_cache=True, drop_occp=drop_occp
    )

    s4_sub = s4_runs[
        (s4_runs["model"].isin(GROUP_MODELS))
        & (s4_runs["method"].isin(["None", "BaseRate", "Platt"]))
    ].copy()
    idx_cols = ["model", "year", "fraction", "repeat", "group_type", "group"]
    shared_ece = s4_sub.pivot_table(index=idx_cols, columns="method", values="ECE")
    shared_brier = s4_sub.pivot_table(index=idx_cols, columns="method", values="Brier")
    shared_gap = s4_sub.pivot_table(index=idx_cols, columns="method", values="MeanGap")

    rows = []
    skipped_group_fits = 0

    for year in EVAL_YEARS:
        df_y = eval_dfs[year]
        y_all = df_y["label"].to_numpy(dtype=int)
        n_rows = len(y_all)
        slices = get_group_slices(df_y)
        sex_slices = [(gt, g, m) for gt, g, m in slices if gt == "Sex"]
        race_slices = [(gt, g, m) for gt, g, m in slices if gt == "Race"]

        for fraction in FRACTIONS:
            for rep in range(n_repeats):
                in_r = draw_recalibration_sample(n_rows, year, fraction, rep)
                keep = ~in_r
                y_eval_all = y_all[keep]
                n_eval_overall = int(keep.sum())

                for m_name in GROUP_MODELS:
                    raw_all = predictions[m_name]["Uncalibrated"][year]
                    platt14_all = predictions[m_name]["Platt"][year]

                    for part_name, part_slices in [("Sex", sex_slices), ("Race", race_slices)]:
                        assembled_br = np.full(n_eval_overall, np.nan, dtype=float)
                        assembled_platt = np.full(n_eval_overall, np.nan, dtype=float)
                        br_valid_all = True
                        platt_valid_all = True

                        for g_type, g_name, mask_full in part_slices:
                            mask_r = in_r & mask_full
                            mask_eval = keep & mask_full
                            mask_in_keep = mask_full[keep]

                            n_r_g = int(mask_r.sum())
                            n_eval_g = int(mask_eval.sum())
                            y_r_g = y_all[mask_r]
                            y_eval_g = y_all[mask_eval]

                            raw_r_g = raw_all[mask_r]
                            platt_r_g = platt14_all[mask_r]
                            raw_eval_g = raw_all[mask_eval]
                            platt_eval_g = platt14_all[mask_eval]

                            two_classes_g = (
                                n_r_g >= 2 and len(np.unique(y_r_g)) > 1 and n_eval_g >= 1
                            )

                            if two_classes_g:
                                p_br_g = BaseRateCalibrator().fit(platt_r_g, y_r_g).predict(platt_eval_g)
                                assembled_br[mask_in_keep] = p_br_g
                                ece_br_g = compute_ece(y_eval_g, p_br_g)
                                brier_br_g = compute_brier(y_eval_g, p_br_g)
                                gap_br_g = compute_mean_gap(y_eval_g, p_br_g)

                                p_pl_g = PlattCalibrator().fit(raw_r_g, y_r_g).predict(raw_eval_g)
                                assembled_platt[mask_in_keep] = p_pl_g
                                ece_pl_g = compute_ece(y_eval_g, p_pl_g)
                                brier_pl_g = compute_brier(y_eval_g, p_pl_g)
                                gap_pl_g = compute_mean_gap(y_eval_g, p_pl_g)
                            else:
                                br_valid_all = False
                                platt_valid_all = False
                                skipped_group_fits += 1
                                ece_br_g = brier_br_g = gap_br_g = np.nan
                                ece_pl_g = brier_pl_g = gap_pl_g = np.nan

                            key = (m_name, year, fraction, rep, g_type, g_name)
                            sh_ece_none = float(shared_ece.loc[key, "None"])
                            sh_ece_br = float(shared_ece.loc[key, "BaseRate"])
                            sh_ece_pl = float(shared_ece.loc[key, "Platt"])
                            sh_brier_br = float(shared_brier.loc[key, "BaseRate"])
                            sh_brier_pl = float(shared_brier.loc[key, "Platt"])
                            sh_gap_none = float(shared_gap.loc[key, "None"])
                            sh_gap_br = float(shared_gap.loc[key, "BaseRate"])
                            sh_gap_pl = float(shared_gap.loc[key, "Platt"])

                            rows.append(
                                {
                                    "model": m_name,
                                    "year": year,
                                    "fraction": fraction,
                                    "repeat": rep,
                                    "group_type": g_type,
                                    "group": g_name,
                                    "n_r_group": n_r_g,
                                    "n_eval": n_eval_g,
                                    "small_group_flag": bool(n_eval_g < SMALL_GROUP_THRESHOLD),
                                    "ECE_None": sh_ece_none,
                                    "MeanGap_None": sh_gap_none,
                                    "ECE_Shared_BaseRate": sh_ece_br,
                                    "ECE_Group_BaseRate": ece_br_g,
                                    "delta_ECE_Group_vs_Shared_BaseRate": ece_br_g - sh_ece_br,
                                    "Brier_Shared_BaseRate": sh_brier_br,
                                    "Brier_Group_BaseRate": brier_br_g,
                                    "MeanGap_Shared_BaseRate": sh_gap_br,
                                    "MeanGap_Group_BaseRate": gap_br_g,
                                    "ECE_Shared_Platt": sh_ece_pl,
                                    "ECE_Group_Platt": ece_pl_g,
                                    "delta_ECE_Group_vs_Shared_Platt": ece_pl_g - sh_ece_pl,
                                    "Brier_Shared_Platt": sh_brier_pl,
                                    "Brier_Group_Platt": brier_pl_g,
                                    "MeanGap_Shared_Platt": sh_gap_pl,
                                    "MeanGap_Group_Platt": gap_pl_g,
                                }
                            )

                        ov_key = (m_name, year, fraction, rep, "Overall", "All")
                        sh_ov_ece_none = float(shared_ece.loc[ov_key, "None"])
                        sh_ov_ece_br = float(shared_ece.loc[ov_key, "BaseRate"])
                        sh_ov_ece_pl = float(shared_ece.loc[ov_key, "Platt"])
                        sh_ov_brier_br = float(shared_brier.loc[ov_key, "BaseRate"])
                        sh_ov_brier_pl = float(shared_brier.loc[ov_key, "Platt"])
                        sh_ov_gap_none = float(shared_gap.loc[ov_key, "None"])
                        sh_ov_gap_br = float(shared_gap.loc[ov_key, "BaseRate"])
                        sh_ov_gap_pl = float(shared_gap.loc[ov_key, "Platt"])

                        ov_ece_br = compute_ece(y_eval_all, assembled_br) if br_valid_all else np.nan
                        ov_brier_br = compute_brier(y_eval_all, assembled_br) if br_valid_all else np.nan
                        ov_gap_br = compute_mean_gap(y_eval_all, assembled_br) if br_valid_all else np.nan

                        ov_ece_pl = compute_ece(y_eval_all, assembled_platt) if platt_valid_all else np.nan
                        ov_brier_pl = compute_brier(y_eval_all, assembled_platt) if platt_valid_all else np.nan
                        ov_gap_pl = compute_mean_gap(y_eval_all, assembled_platt) if platt_valid_all else np.nan

                        rows.append(
                            {
                                "model": m_name,
                                "year": year,
                                "fraction": fraction,
                                "repeat": rep,
                                "group_type": "Overall",
                                "group": f"All (by {part_name})",
                                "n_r_group": int(in_r.sum()),
                                "n_eval": n_eval_overall,
                                "small_group_flag": bool(n_eval_overall < SMALL_GROUP_THRESHOLD),
                                "ECE_None": sh_ov_ece_none,
                                "MeanGap_None": sh_ov_gap_none,
                                "ECE_Shared_BaseRate": sh_ov_ece_br,
                                "ECE_Group_BaseRate": ov_ece_br,
                                "delta_ECE_Group_vs_Shared_BaseRate": ov_ece_br - sh_ov_ece_br,
                                "Brier_Shared_BaseRate": sh_ov_brier_br,
                                "Brier_Group_BaseRate": ov_brier_br,
                                "MeanGap_Shared_BaseRate": sh_ov_gap_br,
                                "MeanGap_Group_BaseRate": ov_gap_br,
                                "ECE_Shared_Platt": sh_ov_ece_pl,
                                "ECE_Group_Platt": ov_ece_pl,
                                "delta_ECE_Group_vs_Shared_Platt": ov_ece_pl - sh_ov_ece_pl,
                                "Brier_Shared_Platt": sh_ov_brier_pl,
                                "Brier_Group_Platt": ov_brier_pl,
                                "MeanGap_Shared_Platt": sh_ov_gap_pl,
                                "MeanGap_Group_Platt": ov_gap_pl,
                            }
                        )

    runs_b = pd.DataFrame(rows)

    sum_keys = ["model", "year", "fraction", "group_type", "group"]
    grp = runs_b.groupby(sum_keys, sort=False)
    sum_b = grp["n_r_group"].mean().rename("n_r_group_mean").to_frame()
    sum_b["n_eval_mean"] = grp["n_eval"].mean()
    sum_b["small_group_flag"] = grp["small_group_flag"].any()
    sum_b["n_Valid_BaseRate"] = grp["ECE_Group_BaseRate"].count()
    sum_b["n_Valid_Platt"] = grp["ECE_Group_Platt"].count()

    metric_cols = [
        "ECE_None",
        "MeanGap_None",
        "ECE_Shared_BaseRate",
        "ECE_Group_BaseRate",
        "delta_ECE_Group_vs_Shared_BaseRate",
        "Brier_Shared_BaseRate",
        "Brier_Group_BaseRate",
        "MeanGap_Shared_BaseRate",
        "MeanGap_Group_BaseRate",
        "ECE_Shared_Platt",
        "ECE_Group_Platt",
        "delta_ECE_Group_vs_Shared_Platt",
        "Brier_Shared_Platt",
        "Brier_Group_Platt",
        "MeanGap_Shared_Platt",
        "MeanGap_Group_Platt",
    ]
    for col in metric_cols:
        sum_b[f"{col}_mean"] = grp[col].mean()
        sum_b[f"{col}_p2.5"] = grp[col].quantile(0.025)
        sum_b[f"{col}_p97.5"] = grp[col].quantile(0.975)

    return runs_b, sum_b.reset_index(), skipped_group_fits


# --- Part 4b: Save group-vs-shared CSVs and plot Figure 7 ---
def save_and_plot_group_vs_shared(
    runs_b: pd.DataFrame,
    sum_b: pd.DataFrame,
    test_mode: bool = False,
) -> dict:
    ensure_dirs()
    suffix = "_test" if test_mode else ""

    runs_csv = os.path.join(OUTPUTS_DIR, f"05_group_vs_shared_runs{suffix}.csv")
    sum_csv = os.path.join(OUTPUTS_DIR, f"05_group_vs_shared_summary{suffix}.csv")

    runs_b.to_csv(runs_csv, index=False)
    sum_b.to_csv(sum_csv, index=False)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8), dpi=200, sharex=True, sharey=True)
    display_groups = ["All (by Race)", "Male", "Female"] + RACE_GROUPS
    short_labels = [
        "Overall (Race)",
        "Male",
        "Female",
        "White",
        "Asian",
        "Some Other",
        "Black",
        "Two+ Races",
        "Other (mrg)",
    ]

    for row_idx, frac_val in enumerate([0.01, 0.10]):
        for col_idx, m_name in enumerate(GROUP_MODELS):
            ax = axes[row_idx, col_idx]
            sub = sum_b[
                (sum_b["year"] == 2018)
                & (sum_b["model"] == m_name)
                & (sum_b["fraction"] == frac_val)
            ].set_index("group")

            x_base = np.arange(len(display_groups))
            width = 0.16

            ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)

            for idx_m, (col_prefix, lbl, color) in enumerate(
                [
                    ("ECE_None", "None (2014-Platt)", "#7f7f7f"),
                    ("ECE_Shared_BaseRate", "Shared BaseRate", "#1f77b4"),
                    ("ECE_Group_BaseRate", "Group-Specific BaseRate", "#2ca02c"),
                    ("ECE_Shared_Platt", "Shared Platt", "#9467bd"),
                    ("ECE_Group_Platt", "Group-Specific Platt", "#ff7f0e"),
                ]
            ):
                means = [float(sub.loc[g, f"{col_prefix}_mean"]) for g in display_groups]
                lows = [float(sub.loc[g, f"{col_prefix}_p2.5"]) for g in display_groups]
                highs = [float(sub.loc[g, f"{col_prefix}_p97.5"]) for g in display_groups]
                yerr_l = np.maximum(0.0, np.array(means) - np.array(lows))
                yerr_h = np.maximum(0.0, np.array(highs) - np.array(means))

                ax.bar(
                    x_base + (idx_m - 2) * width,
                    means,
                    width=width,
                    color=color,
                    alpha=0.88,
                    label=lbl if (row_idx == 0 and col_idx == 0) else None,
                    yerr=[yerr_l, yerr_h],
                    error_kw={"elinewidth": 0.8, "ecolor": "#222222", "capsize": 1.5},
                )

            pct_str = f"{int(frac_val * 100)}%"
            ax.set_title(
                f"{m_name} — Sample Size f = {pct_str}",
                fontsize=10.5,
                fontweight="bold",
            )
            ax.set_xticks(x_base)
            ax.set_xticklabels(short_labels, rotation=32, ha="right", fontsize=8.5)
            ax.grid(True, axis="y", linestyle="--", alpha=0.45)

    axes[0, 0].set_ylabel("2018 Evaluation ECE (f = 1%)")
    axes[1, 0].set_ylabel("2018 Evaluation ECE (f = 10%)")
    axes[0, 0].legend(loc="upper left", fontsize=8.0, ncol=2)

    fig.suptitle(
        "Figure 7: ACSIncome (CA, 2018, 30 Repeats) — Group-Specific vs. Shared Calibrator ECE",
        fontsize=12.0,
        fontweight="bold",
    )
    fig.tight_layout()

    fig_path = os.path.join(OUTPUTS_DIR, f"05_fig7_group_vs_shared{suffix}.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return {
        "runs_csv": runs_csv,
        "summary_csv": sum_csv,
        "fig_path": fig_path,
    }


def print_group_vs_shared_console(sum_b: pd.DataFrame, skipped_fits: int):
    print("\n" + "=" * 132)
    print("GROUP-SPECIFIC CALIBRATOR (R_g) vs. ONE SHARED CALIBRATOR (R) IN 2018 (EXCLUDING GNB)")
    print("=" * 132)
    sub18 = sum_b[(sum_b["year"] == 2018) & (sum_b["fraction"].isin([0.01, 0.05, 0.10]))]
    for m_name in GROUP_MODELS:
        print(f"\nModel: {m_name}")
        print(
            f"{'Group':<20s} {'Frac':>5s} {'n_R_g':>7s} | "
            f"{'Shared BR ECE':>13s} {'Group BR ECE':>13s} {'Paired dECE BR [95% CI]':>29s} | "
            f"{'Shared Pl ECE':>13s} {'Group Pl ECE':>13s}"
        )
        print("-" * 126)
        m_sub = sub18[sub18["model"] == m_name]
        for _, r in m_sub.iterrows():
            d_br = f"{r['delta_ECE_Group_vs_Shared_BaseRate_mean']:+.4f} [{r['delta_ECE_Group_vs_Shared_BaseRate_p2.5']:+.4f},{r['delta_ECE_Group_vs_Shared_BaseRate_p97.5']:+.4f}]"
            print(
                f"{r['group']:<20s} {r['fraction']*100:>4.0f}% {r['n_r_group_mean']:>7.1f} | "
                f"{r['ECE_Shared_BaseRate_mean']:>13.4f} {r['ECE_Group_BaseRate_mean']:>13.4f} {d_br:>29s} | "
                f"{r['ECE_Shared_Platt_mean']:>13.4f} {r['ECE_Group_Platt_mean']:>13.4f}"
            )
    if skipped_fits:
        print(f"\nNote: {skipped_fits} tiny subgroup fit(s) skipped due to empty or single-class R_g.")


# --- Part 5a: ACSEmployment data loading, training, and recalibration ---
EMP_Continuous_Cols = ["AGEP"]


def get_emp_processed_path(year: int, test_mode: bool = False) -> str:
    suffix = "_test" if test_mode else ""
    return os.path.join(PROCESSED_DIR, f"acs_employment_ca_{year}{suffix}.csv")


def load_or_build_employment_years(test_mode: bool = False):
    from folktables import ACSDataSource, ACSEmployment

    ensure_dirs()
    target_paths = {y: get_emp_processed_path(y, test_mode=test_mode) for y in YEARS}
    if all(os.path.exists(p) for p in target_paths.values()):
        return {y: pd.read_csv(target_paths[y]) for y in YEARS}

    full_paths = {y: get_emp_processed_path(y, test_mode=False) for y in YEARS}
    if test_mode and all(os.path.exists(p) for p in full_paths.values()):
        dfs = {}
        for y in YEARS:
            full_df = pd.read_csv(full_paths[y])
            test_df = full_df.sample(frac=TEST_SAMPLE_FRAC, random_state=SEED).reset_index(drop=True)
            test_df.to_csv(target_paths[y], index=False)
            dfs[y] = test_df
        return dfs

    raw_dfs = {}
    emp_features = list(ACSEmployment.features)
    for y in YEARS:
        source = ACSDataSource(
            survey_year=str(y),
            horizon=HORIZON,
            survey=SURVEY,
            root_dir=RAW_DIR,
        )
        ca_raw = source.get_data(states=[STATE], download=False)
        X_arr, y_arr, _ = ACSEmployment.df_to_numpy(ca_raw)
        df = pd.DataFrame(X_arr, columns=emp_features)
        df["label"] = y_arr.astype(int)
        df["year"] = int(y)
        df["sex_group"] = df["SEX"].astype(int).map(SEX_CODE_MAP)
        df["race_raw"] = df["RAC1P"].astype(int).map(RACE_CODE_MAP)
        raw_dfs[y] = df

    raw_dfs, _, _ = apply_race_merging_rule(raw_dfs)

    out_dfs = {}
    for y in YEARS:
        df_y = raw_dfs[y]
        if test_mode:
            df_test = df_y.sample(frac=TEST_SAMPLE_FRAC, random_state=SEED).reset_index(drop=True)
            df_test.to_csv(target_paths[y], index=False)
            out_dfs[y] = df_test
        else:
            df_y.to_csv(target_paths[y], index=False)
            out_dfs[y] = df_y
    return out_dfs


def build_employment_preprocessor():
    from folktables import ACSEmployment

    emp_features = list(ACSEmployment.features)
    cat_cols = [c for c in emp_features if c not in EMP_Continuous_Cols]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), EMP_Continuous_Cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )


def train_and_predict_employment(emp_dfs: dict, test_mode: bool = False) -> dict:
    from folktables import ACSEmployment

    suffix = "_test" if test_mode else ""
    cache_path = os.path.join(PROCESSED_DIR, f"employment_predictions{suffix}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    emp_features = list(ACSEmployment.features)
    df_train, df_cal, df_test_2014 = split_2014_data(emp_dfs[2014])

    preprocessor = build_employment_preprocessor()
    X_train_proc = preprocessor.fit_transform(df_train[emp_features])
    X_cal_proc = preprocessor.transform(df_cal[emp_features])

    X_train_raw = df_train[emp_features].to_numpy(dtype=np.float64)
    X_cal_raw = df_cal[emp_features].to_numpy(dtype=np.float64)

    y_train = df_train["label"].to_numpy(dtype=int)
    y_cal = df_cal["label"].to_numpy(dtype=int)

    eval_slices = {2014: df_test_2014}
    for y in EVAL_YEARS:
        eval_slices[y] = emp_dfs[y]

    eval_X_proc = {}
    eval_X_raw = {}
    eval_y = {}
    for y in YEARS:
        df_y = eval_slices[y]
        eval_X_proc[y] = preprocessor.transform(df_y[emp_features])
        eval_X_raw[y] = df_y[emp_features].to_numpy(dtype=np.float64)
        eval_y[y] = df_y["label"].to_numpy(dtype=int)

    models = build_models()
    predictions = {}
    for model_name in MODEL_NAMES:
        clf = models[model_name]
        uses_raw = model_name == "Gradient Boosted Trees"
        needs_dense = model_name == "Gaussian Naive Bayes"

        X_tr = X_train_raw if uses_raw else (to_dense_if_sparse(X_train_proc) if needs_dense else X_train_proc)
        X_ca = X_cal_raw if uses_raw else (to_dense_if_sparse(X_cal_proc) if needs_dense else X_cal_proc)

        clf.fit(X_tr, y_train)

        p_cal_uncal = clf.predict_proba(X_ca)[:, 1]
        platt_2014 = PlattCalibrator().fit(p_cal_uncal, y_cal)
        iso_2014 = IsotonicCalibrator().fit(p_cal_uncal, y_cal)

        predictions[model_name] = {"Uncalibrated": {}, "2014-Platt": {}, "2014-Isotonic": {}}
        for y in YEARS:
            X_ev = eval_X_raw[y] if uses_raw else (to_dense_if_sparse(eval_X_proc[y]) if needs_dense else eval_X_proc[y])
            p_uncal = clf.predict_proba(X_ev)[:, 1]
            predictions[model_name]["Uncalibrated"][y] = p_uncal
            predictions[model_name]["2014-Platt"][y] = platt_2014.predict(p_uncal)
            predictions[model_name]["2014-Isotonic"][y] = iso_2014.predict(p_uncal)

    payload = {
        "predictions": predictions,
        "eval_y": eval_y,
        "split_sizes_2014": {
            "train": len(df_train),
            "cal": len(df_cal),
            "test": len(df_test_2014),
        },
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    return payload


def compute_employment(test_mode: bool = False):
    emp_dfs = load_or_build_employment_years(test_mode=test_mode)
    pred_payload = train_and_predict_employment(emp_dfs, test_mode=test_mode)
    predictions = pred_payload["predictions"]

    _, _, df_test_2014 = split_2014_data(emp_dfs[2014])
    eval_dfs = {2014: df_test_2014}
    for y in EVAL_YEARS:
        eval_dfs[y] = emp_dfs[y]

    rate_rows = []
    for y in YEARS:
        df_y = emp_dfs[y]
        rate_rows.append(
            {
                "year": y,
                "n_total": len(df_y),
                "pos_rate": float(df_y["label"].mean()),
            }
        )
    rate_df = pd.DataFrame(rate_rows)

    n_boot = 50 if test_mode else N_BOOT
    drift_rows = []
    for y in YEARS:
        df_y = eval_dfs[y]
        y_true = df_y["label"].to_numpy(dtype=int)
        slices = get_group_slices(df_y)

        for g_type, g_name, mask in slices:
            idx = np.flatnonzero(mask)
            n_g = int(len(idx))
            if n_g == 0:
                continue
            y_g = y_true[idx]
            pos_g = float(np.mean(y_g))
            small_flag = bool(n_g < SMALL_GROUP_THRESHOLD)

            active_models = MODEL_NAMES if g_type == "Overall" else GROUP_MODELS
            prob_dict = {m: predictions[m]["2014-Platt"][y][idx] for m in active_models}
            boot_seed = SEED + y * 100 + len(drift_rows)
            ci_dict = bootstrap_ece_and_gap_multi(
                y_g, prob_dict, n_boot=n_boot, n_bins=10, seed=boot_seed
            )

            for m_name in active_models:
                p_g = prob_dict[m_name]
                pt = compute_all_metrics(y_g, p_g)
                ci = ci_dict[m_name]
                drift_rows.append(
                    {
                        "task": "ACSEmployment",
                        "year": y,
                        "model": m_name,
                        "calibration": "2014-Platt",
                        "group_type": g_type,
                        "group": g_name,
                        "n": n_g,
                        "small_group_flag": small_flag,
                        "pos_rate": pos_g,
                        "pred_mean": float(np.mean(p_g)),
                        "ECE": pt["ECE"],
                        "ECE_lo": ci["ece_low"],
                        "ECE_hi": ci["ece_high"],
                        "MeanGap": pt["MeanGap"],
                        "MeanGap_lo": ci["gap_low"],
                        "MeanGap_hi": ci["gap_high"],
                        "Brier": pt["Brier"],
                        "Accuracy": pt["Accuracy"],
                        "ROC_AUC": pt["AUC"],
                    }
                )
    drift_df = pd.DataFrame(drift_rows)

    ece_base_lookup = {}
    for _, r in drift_df[drift_df["year"].isin([2014, 2018])].iterrows():
        ece_base_lookup[(int(r["year"]), r["model"], r["group_type"], r["group"])] = float(r["ECE"])

    n_repeats = N_REPEATS_TEST if test_mode else N_REPEATS_FULL
    df_2018 = eval_dfs[2018]
    y_2018 = df_2018["label"].to_numpy(dtype=int)
    n_total_2018 = len(y_2018)
    slices_2018 = get_group_slices(df_2018)

    recal_runs = []
    for f in FRACTIONS:
        f_int = int(round(f * 100))
        for r_idx in range(n_repeats):
            draw_seed = SEED + 100000 * 2018 + 1000 * f_int + r_idx
            in_r = draw_recalibration_sample(n_total_2018, 2018, f, r_idx)
            keep = ~in_r
            n_r = int(np.sum(in_r))
            n_eval_all = int(np.sum(keep))
            assert n_r + n_eval_all == n_total_2018

            y_r = y_2018[in_r]
            y_eval_all = y_2018[keep]
            if len(np.unique(y_r)) < 2:
                continue

            for m_name in MODEL_NAMES:
                raw_2018 = predictions[m_name]["Uncalibrated"][2018]
                platt_2018 = predictions[m_name]["2014-Platt"][2018]
                raw_r, raw_eval = raw_2018[in_r], raw_2018[keep]
                platt_r, platt_eval = platt_2018[in_r], platt_2018[keep]

                method_preds = fit_and_predict_methods(
                    raw_r, platt_r, y_r, raw_eval, platt_eval
                )

                for g_type, g_name, mask_full in slices_2018:
                    if g_type != "Overall" and m_name not in GROUP_MODELS:
                        continue
                    mask_eval = mask_full[keep]
                    n_eval_g = int(np.sum(mask_eval))
                    n_r_g = int(np.sum(mask_full[in_r]))
                    assert n_eval_g + n_r_g == int(np.sum(mask_full))
                    if n_eval_g == 0:
                        continue

                    y_eval_g = y_eval_all[mask_eval]
                    ece_none_g = compute_ece(y_eval_g, method_preds["None"][mask_eval], n_bins=10)

                    for method in METHODS:
                        p_m_full = method_preds[method]
                        if p_m_full is None:
                            continue
                        p_m_g = p_m_full[mask_eval]
                        ece_val = compute_ece(y_eval_g, p_m_g, n_bins=10)
                        gap_val = compute_mean_gap(y_eval_g, p_m_g)
                        brier_val = compute_brier(y_eval_g, p_m_g)
                        rec_pct = (
                            100.0 * (ece_none_g - ece_val) / ece_none_g
                            if ece_none_g > 1e-12
                            else np.nan
                        )
                        recal_runs.append(
                            {
                                "year": 2018,
                                "model": m_name,
                                "fraction": f,
                                "repeat": r_idx,
                                "seed": draw_seed,
                                "method": method,
                                "group_type": g_type,
                                "group": g_name,
                                "n_r": n_r_g,
                                "n_eval": n_eval_g,
                                "ECE": ece_val,
                                "MeanGap": gap_val,
                                "Brier": brier_val,
                                "recovery_pct": rec_pct,
                            }
                        )

    recal_runs_df = pd.DataFrame(recal_runs)

    none_df = recal_runs_df[recal_runs_df["method"] == "None"][
        ["model", "fraction", "repeat", "group_type", "group", "ECE"]
    ].rename(columns={"ECE": "ECE_None"})
    recal_merged = recal_runs_df.merge(
        none_df, on=["model", "fraction", "repeat", "group_type", "group"], how="left"
    )
    recal_merged["delta_ECE_vs_None"] = recal_merged["ECE"] - recal_merged["ECE_None"]

    sum_rows = []
    grp_cols = ["year", "model", "fraction", "method", "group_type", "group"]
    for keys, sub in recal_merged.groupby(grp_cols, sort=False):
        y_val, m_name, frac, method, g_type, g_name = keys
        ece_arr = sub["ECE"].to_numpy(dtype=float)
        gap_arr = sub["MeanGap"].to_numpy(dtype=float)
        rec_arr = sub["recovery_pct"].dropna().to_numpy(dtype=float)
        d_arr = sub["delta_ECE_vs_None"].to_numpy(dtype=float)
        sum_rows.append(
            {
                "year": int(y_val),
                "model": m_name,
                "fraction": float(frac),
                "method": method,
                "group_type": g_type,
                "group": g_name,
                "n_repeats": len(sub),
                "n_r_mean": float(sub["n_r"].mean()),
                "n_eval_mean": float(sub["n_eval"].mean()),
                "ECE_2014": ece_base_lookup[(2014, m_name, g_type, g_name)],
                "ECE_2018_uncal": ece_base_lookup[(2018, m_name, g_type, g_name)],
                "ECE_mean": float(np.mean(ece_arr)),
                "ECE_sd": float(np.std(ece_arr, ddof=0)),
                "ECE_p2.5": float(np.percentile(ece_arr, 2.5)),
                "ECE_p97.5": float(np.percentile(ece_arr, 97.5)),
                "MeanGap_mean": float(np.mean(gap_arr)),
                "MeanGap_p2.5": float(np.percentile(gap_arr, 2.5)),
                "MeanGap_p97.5": float(np.percentile(gap_arr, 97.5)),
                "recovery_mean": float(np.mean(rec_arr)) if len(rec_arr) else np.nan,
                "recovery_p2.5": float(np.percentile(rec_arr, 2.5)) if len(rec_arr) else np.nan,
                "recovery_p97.5": float(np.percentile(rec_arr, 97.5)) if len(rec_arr) else np.nan,
                "delta_ECE_vs_None_mean": float(np.mean(d_arr)),
                "delta_ECE_vs_None_p2.5": float(np.percentile(d_arr, 2.5)),
                "delta_ECE_vs_None_p97.5": float(np.percentile(d_arr, 97.5)),
            }
        )
    recal_sum_df = pd.DataFrame(sum_rows)
    return rate_df, drift_df, recal_runs_df, recal_sum_df


# --- Part 5b: Save ACSEmployment CSVs and plot Figure 8 ---
def save_and_plot_employment(
    rate_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    recal_sum_df: pd.DataFrame,
    yearly_income_dfs: dict,
    test_mode: bool = False,
) -> dict:
    suffix = "_test" if test_mode else ""
    drift_csv = os.path.join(OUTPUTS_DIR, f"05_employment_drift{suffix}.csv")
    recal_csv = os.path.join(OUTPUTS_DIR, f"05_employment_recal_summary{suffix}.csv")

    drift_df.to_csv(drift_csv, index=False)
    recal_sum_df.to_csv(recal_csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=200)

    ax0 = axes[0]
    inc_rates = np.array([float(yearly_income_dfs[y]["label"].mean()) for y in YEARS])
    inc_base14 = float(inc_rates[0])
    inc_shift = (inc_rates - inc_base14) * 100.0

    emp_sorted = rate_df.sort_values("year")
    emp_base14 = float(emp_sorted[emp_sorted["year"] == 2014]["pos_rate"].iloc[0])
    emp_shift = (emp_sorted["pos_rate"].to_numpy(dtype=float) - emp_base14) * 100.0

    ax0.plot(
        YEARS,
        inc_shift,
        marker="o",
        linewidth=2.2,
        color="#d95f02",
        label=f"ACSIncome (2014 base = {inc_base14*100:.1f}%)",
    )
    ax0.plot(
        YEARS,
        emp_shift,
        marker="s",
        linewidth=2.2,
        color="#1b9e77",
        label=f"ACSEmployment (2014 base = {emp_base14*100:.1f}%)",
    )
    ax0.set_title("(A) Base-Rate Shift vs. 2014 (pp)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Evaluation Year")
    ax0.set_ylabel("Positive-Rate Shift vs. 2014 (pp)")
    ax0.set_xticks(YEARS)
    ax0.grid(True, linestyle="--", alpha=0.45)
    ax0.legend(loc="upper left", fontsize=8.5)

    ax1 = axes[1]
    model_colors = {
        "Logistic Regression": "#1f77b4",
        "Random Forest": "#ff7f0e",
        "Gradient Boosted Trees": "#2ca02c",
        "Gaussian Naive Bayes": "#d62728",
    }
    for m_name in MODEL_NAMES:
        m_sub = drift_df[
            (drift_df["group_type"] == "Overall") & (drift_df["model"] == m_name)
        ].sort_values("year")
        ax1.plot(
            m_sub["year"],
            m_sub["ECE"],
            marker="o",
            linewidth=2.0,
            color=model_colors[m_name],
            label=m_name,
        )
    ax1.set_title("(B) ACSEmployment 2014-Platt ECE (2014–2018)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Evaluation Year")
    ax1.set_ylabel("Overall ECE (10 bins)")
    ax1.set_xticks(YEARS)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", fontsize=8.5)

    ax2 = axes[2]
    lr_ov_recal = recal_sum_df[
        (recal_sum_df["group_type"] == "Overall")
        & (recal_sum_df["model"] == "Logistic Regression")
    ]
    x_pcts = [f * 100.0 for f in FRACTIONS]
    for method in ["None", "BaseRate", "Platt", "Beta", "VennAbers", "Isotonic"]:
        m_sub = lr_ov_recal[lr_ov_recal["method"] == method].sort_values("fraction")
        ax2.plot(
            x_pcts,
            m_sub["ECE_mean"],
            marker=METHOD_MARKERS[method],
            linewidth=1.9,
            linestyle="--" if method == "None" else "-",
            color=METHOD_COLORS[method],
            label=method,
        )
    ax2.set_title("(C) ACSEmployment 2018 Recalibration (LogReg)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("2018 Recalibration Sample Size f (%)")
    ax2.set_ylabel("2018 Evaluation ECE")
    ax2.set_xticks(x_pcts)
    ax2.set_xticklabels(["1%", "2%", "5%", "10%"])
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="upper right", fontsize=8.0)

    fig.suptitle(
        "Figure 8: ACSEmployment vs. ACSIncome (CA, 2014–2018, 30 Repeats) — Drift & Recalibration",
        fontsize=12.0,
        fontweight="bold",
    )
    fig.tight_layout()

    fig_path = os.path.join(OUTPUTS_DIR, f"05_fig8_employment{suffix}.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return {
        "drift_csv": drift_csv,
        "recal_csv": recal_csv,
        "fig_path": fig_path,
    }


def print_employment_console(rate_df: pd.DataFrame, drift_df: pd.DataFrame, recal_sum_df: pd.DataFrame):
    print("\n" + "=" * 120)
    print("ACSEMPLOYMENT REPLICATION (CALIFORNIA, 2014–2018)")
    print("=" * 120)
    print("\n1. Yearly Sample Sizes and Employment Base Rate:")
    r14 = float(rate_df[rate_df["year"] == 2014]["pos_rate"].iloc[0])
    for _, r in rate_df.iterrows():
        d_pp = (float(r["pos_rate"]) - r14) * 100.0
        print(f"  {int(r['year'])}: N = {int(r['n_total']):>7,d} | Employment Rate = {float(r['pos_rate'])*100:.2f}% ({d_pp:+.2f} pp vs 2014)")

    print("\n2. Overall Calibration Drift (2014 Test vs. 2018 Uncalibrated) & 2018 Recalibration:")
    print(
        f"{'Model':<24s} {'2014 ECE':>9s} {'2018 ECE':>9s} {'2018 Gap':>9s} | "
        f"{'1% BaseRate ECE':>16s} {'1% Platt ECE':>16s} {'5% Platt ECE':>16s} {'10% Iso ECE':>16s}"
    )
    print("-" * 120)
    ov_drift = drift_df[drift_df["group_type"] == "Overall"]
    ov_recal = recal_sum_df[recal_sum_df["group_type"] == "Overall"]
    for m_name in MODEL_NAMES:
        d14 = ov_drift[(ov_drift["model"] == m_name) & (ov_drift["year"] == 2014)].iloc[0]
        d18 = ov_drift[(ov_drift["model"] == m_name) & (ov_drift["year"] == 2018)].iloc[0]
        br1 = ov_recal[(ov_recal["model"] == m_name) & (ov_recal["fraction"] == 0.01) & (ov_recal["method"] == "BaseRate")].iloc[0]
        pl1 = ov_recal[(ov_recal["model"] == m_name) & (ov_recal["fraction"] == 0.01) & (ov_recal["method"] == "Platt")].iloc[0]
        pl5 = ov_recal[(ov_recal["model"] == m_name) & (ov_recal["fraction"] == 0.05) & (ov_recal["method"] == "Platt")].iloc[0]
        iso10 = ov_recal[(ov_recal["model"] == m_name) & (ov_recal["fraction"] == 0.10) & (ov_recal["method"] == "Isotonic")].iloc[0]
        print(
            f"{m_name:<24s} {d14['ECE']:>9.4f} {d18['ECE']:>9.4f} {d18['MeanGap']:>+9.4f} | "
            f"{br1['ECE_mean']:>16.4f} {pl1['ECE_mean']:>16.4f} {pl5['ECE_mean']:>16.4f} {iso10['ECE_mean']:>16.4f}"
        )


# --- Part 6a: Robustness comparison CSVs (2017 target year and no-OCCP) ---
def run_robustness_summaries(test_mode: bool = False):
    suffix = "_test" if test_mode else ""
    s3_main = pd.read_csv(os.path.join("outputs", f"03_drift_metrics_by_group{suffix}.csv"), keep_default_na=False, na_values=[""])
    t3_main = pd.read_csv(os.path.join("outputs", f"05_table3_method_sample_size_summary{suffix}.csv"), keep_default_na=False, na_values=[""])

    r17_18 = t3_main[t3_main["year"].isin([2017, 2018])].copy()
    out_2017 = os.path.join("outputs", f"05_robustness_2017_summary{suffix}.csv")
    r17_18.to_csv(out_2017, index=False)
    print(f"Saved 2017 robustness summary CSV: {out_2017} ({len(r17_18)} rows)")

    s3_no_path = os.path.join(OUTPUTS_NO_OCCP_DIR, f"03_drift_metrics_by_group{suffix}.csv")
    t3_no_path = os.path.join(OUTPUTS_NO_OCCP_DIR, f"05_table3_method_sample_size_summary{suffix}.csv")
    if os.path.exists(s3_no_path) and os.path.exists(t3_no_path):
        s3_no = pd.read_csv(s3_no_path, keep_default_na=False, na_values=[""])
        m_s3 = s3_main[s3_main["calibration"] == "Platt"].merge(
            s3_no[s3_no["calibration"] == "Platt"],
            on=["model", "calibration", "year", "group_type", "group", "n"],
            suffixes=("_with_occp", "_no_occp"),
        )
        m_s3["delta_AUC_no_minus_with"] = m_s3["AUC_no_occp"] - m_s3["AUC_with_occp"]
        m_s3["delta_ECE_no_minus_with"] = m_s3["ECE_no_occp"] - m_s3["ECE_with_occp"]
        m_s3["delta_MeanGap_no_minus_with"] = m_s3["MeanGap_no_occp"] - m_s3["MeanGap_with_occp"]
        out_no_occp = os.path.join("outputs", f"05_robustness_no_occp_comparison{suffix}.csv")
        m_s3.to_csv(out_no_occp, index=False)
        print(f"Saved no-OCCP comparison CSV:      {out_no_occp} ({len(m_s3)} rows)")


# --- Part 6b: Main entry point ---
def main():
    global OUTPUTS_DIR
    args = parse_args()
    OUTPUTS_DIR = get_outputs_dir(args.drop_occp)
    ensure_dirs(drop_occp=args.drop_occp)
    mode_str = "TEST MODE" if args.test else "FULL DATA MODE"
    occp_str = " [WITHOUT OCCP -> outputs_no_occp/]" if args.drop_occp else ""
    print(f"Running 05_analyze_and_plot.py — {mode_str}{occp_str}\n")

    s3_df, s4_runs, s4_sum = load_drift_and_recalibration_outputs(test_mode=args.test)
    yearly_dfs, _, _ = build_and_cache_all_years(download=False, test_mode=args.test)

    if args.robustness:
        run_robustness_summaries(test_mode=args.test)
        return

    if args.monitoring:
        r_stats_df, pol_summary, multi_yr_summary = compute_monitoring(
            yearly_dfs, s4_runs, test_mode=args.test, drop_occp=args.drop_occp
        )
        print_monitoring_console(pol_summary, multi_yr_summary)
        monitoring_paths = save_and_plot_monitoring(
            r_stats_df, pol_summary, multi_yr_summary, test_mode=args.test
        )
        print("\n" + "=" * 88)
        print("MONITORING OUTPUTS CREATED")
        print("=" * 88)
        for k, p in monitoring_paths.items():
            print(f"  {k:<14s}: {p}")
        return

    if args.group_vs_shared:
        runs_b, sum_b, skipped_b = compute_group_vs_shared(
            yearly_dfs, s4_runs, test_mode=args.test, drop_occp=args.drop_occp
        )
        print_group_vs_shared_console(sum_b, skipped_b)
        b_paths = save_and_plot_group_vs_shared(runs_b, sum_b, test_mode=args.test)
        print("\n" + "=" * 88)
        print("GROUP-VS-SHARED OUTPUTS CREATED")
        print("=" * 88)
        for k, p in b_paths.items():
            print(f"  {k:<14s}: {p}")
        return

    if args.employment:
        rate_c, drift_c, runs_c, sum_c = compute_employment(test_mode=args.test)
        print_employment_console(rate_c, drift_c, sum_c)
        c_paths = save_and_plot_employment(rate_c, drift_c, sum_c, yearly_dfs, test_mode=args.test)
        print("\n" + "=" * 88)
        print("ACSEMPLOYMENT OUTPUTS CREATED")
        print("=" * 88)
        for k, p in c_paths.items():
            print(f"  {k:<14s}: {p}")
        return

    if args.replot_extensions:
        suffix = "_test" if args.test else ""
        r_stats_df = pd.read_csv(os.path.join(OUTPUTS_DIR, f"05_monitoring_runs{suffix}.csv"), keep_default_na=False, na_values=[""])
        pol_summary = pd.read_csv(os.path.join(OUTPUTS_DIR, f"05_monitoring_summary{suffix}.csv"), keep_default_na=False, na_values=[""])
        multi_yr_summary = pd.read_csv(os.path.join(OUTPUTS_DIR, f"05_monitoring_4year_policy_summary{suffix}.csv"), keep_default_na=False, na_values=[""])
        save_and_plot_monitoring(r_stats_df, pol_summary, multi_yr_summary, test_mode=args.test)

        runs_b = pd.read_csv(os.path.join(OUTPUTS_DIR, f"05_group_vs_shared_runs{suffix}.csv"), keep_default_na=False, na_values=[""])
        sum_b = pd.read_csv(os.path.join(OUTPUTS_DIR, f"05_group_vs_shared_summary{suffix}.csv"), keep_default_na=False, na_values=[""])
        save_and_plot_group_vs_shared(runs_b, sum_b, test_mode=args.test)

        emp_dfs = load_or_build_employment_years(test_mode=args.test)
        rate_c = pd.DataFrame([{"year": y, "n_total": len(emp_dfs[y]), "pos_rate": float(emp_dfs[y]["label"].mean())} for y in YEARS])
        drift_c = pd.read_csv(os.path.join(OUTPUTS_DIR, f"05_employment_drift{suffix}.csv"), keep_default_na=False, na_values=[""])
        sum_c = pd.read_csv(os.path.join(OUTPUTS_DIR, f"05_employment_recal_summary{suffix}.csv"), keep_default_na=False, na_values=[""])
        save_and_plot_employment(rate_c, drift_c, sum_c, yearly_dfs, test_mode=args.test)
        print("Regenerated Monitoring, Group-vs-Shared, and ACSEmployment Figures 6-8 from cached CSVs.")
        return

    rec_summary = compute_per_draw_recovery(s4_runs)
    paired_df = compute_paired_method_comparisons(s4_runs)

    print_analysis_summary(rec_summary, paired_df)
    table_paths = build_and_save_tables(s3_df, rec_summary, paired_df, test_mode=args.test)

    f1 = make_05_fig1_positive_rate_by_year(yearly_dfs, s3_df, test_mode=args.test)
    f2 = make_05_fig2_reliability_2014_vs_2018(yearly_dfs, test_mode=args.test, drop_occp=args.drop_occp)
    f3 = make_05_fig3_ece_by_year_per_group(s3_df, test_mode=args.test)
    f4 = make_05_fig4_ece_vs_sample_size(s4_sum, test_mode=args.test)
    f5 = make_05_fig5_recovery_by_group(rec_summary, test_mode=args.test)

    print("\n" + "=" * 88)
    print("ANALYSIS OUTPUTS CREATED")
    print("=" * 88)
    for k, p in table_paths.items():
        print(f"  Table ({k:<10s}): {p}")
    for idx, fp in enumerate([f1, f2, f3, f4, f5], start=1):
        print(f"  Figure {idx}           : {fp}")


if __name__ == "__main__":
    main()
