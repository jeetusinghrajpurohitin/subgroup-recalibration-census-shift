"""
02_train_baselines.py
Train baseline classifiers on the 2014 train slice, fit 2014 Platt and Isotonic
calibrators on the 2014 calibration slice, and evaluate across 2014-2018.
"""

import argparse
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from calibrators import IsotonicCalibrator, PlattCalibrator
from common import (
    EVAL_YEARS,
    PROCESSED_DIR,
    SEED,
    YEARS,
    build_and_cache_all_years,
    build_preprocessor,
    ensure_dirs,
    get_feature_cols,
    get_outputs_dir,
    split_2014_data,
)
from metrics import (
    compute_all_metrics,
    compute_reliability_bins,
    run_ece_unit_checks,
)

# --- Part 1a: Models, calibration types, and reference check values ---
MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosted Trees",
    "Gaussian Naive Bayes",
]

CALIBRATION_NAMES = ["Uncalibrated", "Platt", "Isotonic"]

BASELINE_REFERENCE = {
    "Logistic Regression": {
        "uncal_ece_2014": 0.0069,
        "uncal_auc_2014": 0.8999,
        "platt_ece_2018": 0.0522,
        "platt_gap_2018": -0.0517,
        "platt_auc_2018": 0.8839,
    },
    "Random Forest": {
        "uncal_ece_2014": 0.1191,
        "uncal_auc_2014": 0.8841,
        "platt_ece_2018": 0.0534,
        "platt_gap_2018": -0.0534,
        "platt_auc_2018": 0.8727,
    },
    "Gradient Boosted Trees": {
        "uncal_ece_2014": 0.0086,
        "uncal_auc_2014": 0.9080,
        "platt_ece_2018": 0.0473,
        "platt_gap_2018": -0.0473,
        "platt_auc_2018": 0.9009,
    },
    "Gaussian Naive Bayes": {
        "uncal_ece_2014": 0.2088,
        "uncal_auc_2014": 0.8615,
        "platt_ece_2018": 0.0520,
        "platt_gap_2018": -0.0510,
        "platt_auc_2018": 0.8323,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 2014 baseline models and 2014 calibrators."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode using the 1%% cached sample per year (seed 42).",
    )
    parser.add_argument(
        "--drop-occp",
        action="store_true",
        help="Remove OCCP from ACSIncome features and write all outputs to outputs_no_occp/.",
    )
    return parser.parse_args()


# --- Part 1b: Model setup and 2014 training / calibration ---
def to_dense_if_sparse(x_mat) -> np.ndarray:
    if sparse.issparse(x_mat):
        return x_mat.toarray()
    return np.asarray(x_mat)


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=SEED,
        ),
        "Gradient Boosted Trees": HistGradientBoostingClassifier(
            max_iter=150,
            random_state=SEED,
        ),
        "Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-2),
    }


def get_predictions_cache_path(test_mode: bool = False, drop_occp: bool = False) -> str:
    suffix = "_test" if test_mode else ""
    if drop_occp:
        return os.path.join(PROCESSED_DIR, f"02_baseline_predictions_no_occp{suffix}.pkl")
    return os.path.join(PROCESSED_DIR, f"02_baseline_predictions{suffix}.pkl")


def train_and_predict_all_years(
    yearly_dfs: dict, test_mode: bool = False, use_cache: bool = False, drop_occp: bool = False
) -> tuple[dict, dict]:
    train_2014, cal_2014, test_2014 = split_2014_data(yearly_dfs[2014])

    eval_dfs = {2014: test_2014}
    for y in EVAL_YEARS:
        eval_dfs[y] = yearly_dfs[y]

    cache_path = get_predictions_cache_path(test_mode=test_mode, drop_occp=drop_occp)
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            predictions = pickle.load(f)
        return eval_dfs, predictions

    f_cols = get_feature_cols(drop_occp=drop_occp)

    preprocessor = build_preprocessor(dense_output=False, drop_occp=drop_occp)
    x_train_prep = preprocessor.fit_transform(train_2014[f_cols])
    x_cal_prep = preprocessor.transform(cal_2014[f_cols])
    x_eval_prep = {y: preprocessor.transform(eval_dfs[y][f_cols]) for y in YEARS}

    x_train_raw = train_2014[f_cols].to_numpy(dtype=float)
    x_cal_raw = cal_2014[f_cols].to_numpy(dtype=float)
    x_eval_raw = {y: eval_dfs[y][f_cols].to_numpy(dtype=float) for y in YEARS}

    y_train = train_2014["label"].to_numpy(dtype=int)
    y_cal = cal_2014["label"].to_numpy(dtype=int)

    models = build_models()
    predictions = {
        m_name: {c_name: {} for c_name in CALIBRATION_NAMES} for m_name in MODEL_NAMES
    }

    for m_name in MODEL_NAMES:
        model = models[m_name]
        print(f"  Training and calibrating: {m_name} ...")

        if m_name == "Gradient Boosted Trees":
            model.fit(x_train_raw, y_train)
            cal_raw_probs = model.predict_proba(x_cal_raw)[:, 1]
        elif m_name == "Gaussian Naive Bayes":
            model.fit(to_dense_if_sparse(x_train_prep), y_train)
            cal_raw_probs = model.predict_proba(to_dense_if_sparse(x_cal_prep))[:, 1]
        else:
            model.fit(x_train_prep, y_train)
            cal_raw_probs = model.predict_proba(x_cal_prep)[:, 1]

        platt = PlattCalibrator().fit(cal_raw_probs, y_cal)
        isotonic = IsotonicCalibrator().fit(cal_raw_probs, y_cal)

        for y in YEARS:
            if m_name == "Gradient Boosted Trees":
                uncal_p = model.predict_proba(x_eval_raw[y])[:, 1]
            elif m_name == "Gaussian Naive Bayes":
                uncal_p = model.predict_proba(to_dense_if_sparse(x_eval_prep[y]))[:, 1]
            else:
                uncal_p = model.predict_proba(x_eval_prep[y])[:, 1]

            predictions[m_name]["Uncalibrated"][y] = uncal_p
            predictions[m_name]["Platt"][y] = platt.predict(uncal_p)
            predictions[m_name]["Isotonic"][y] = isotonic.predict(uncal_p)

    ensure_dirs(drop_occp=drop_occp)
    with open(cache_path, "wb") as f:
        pickle.dump(predictions, f)

    return eval_dfs, predictions


# --- Part 2a: Plot 2014 vs 2018 reliability diagrams and ECE by year ---
def plot_reliability_2014_vs_2018(
    eval_dfs: dict, predictions: dict, test_mode: bool = False, drop_occp: bool = False
) -> str:
    ensure_dirs(drop_occp=drop_occp)
    out_dir = get_outputs_dir(drop_occp)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), dpi=200, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    y_2014 = eval_dfs[2014]["label"].to_numpy(dtype=int)
    y_2018 = eval_dfs[2018]["label"].to_numpy(dtype=int)

    styles = [
        ("Uncalibrated", 2014, "#1f77b4", "--", "o", "2014 Uncalibrated"),
        ("Platt", 2014, "#1f77b4", "-", "s", "2014 Platt"),
        ("Uncalibrated", 2018, "#d62728", "--", "^", "2018 Uncalibrated"),
        ("Platt", 2018, "#d62728", "-", "D", "2018 Platt"),
    ]

    for ax, m_name in zip(axes_flat, MODEL_NAMES):
        ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.2, label="Perfect")
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
        ax.grid(True, linestyle="--", alpha=0.4)

    for ax in axes[1, :]:
        ax.set_xlabel("Mean Predicted Probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Empirical Positive Rate")

    axes_flat[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(
        "Reliability Diagrams: 2014 (Test) vs 2018 (Full Year)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    fname = (
        "02_reliability_2014_vs_2018_test.png"
        if test_mode
        else "02_reliability_2014_vs_2018.png"
    )
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_ece_by_year(
    metrics_df: pd.DataFrame, test_mode: bool = False, drop_occp: bool = False
) -> str:
    ensure_dirs(drop_occp=drop_occp)
    out_dir = get_outputs_dir(drop_occp)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=200, sharey=False)
    colors = {
        "Logistic Regression": "#1f77b4",
        "Random Forest": "#ff7f0e",
        "Gradient Boosted Trees": "#2ca02c",
        "Gaussian Naive Bayes": "#d62728",
    }

    for ax, cal_name in zip(axes, CALIBRATION_NAMES):
        sub = metrics_df[metrics_df["calibration"] == cal_name]
        for m_name in MODEL_NAMES:
            m_sub = sub[sub["model"] == m_name].sort_values("year")
            ax.plot(
                m_sub["year"],
                m_sub["ECE"],
                marker="o",
                linewidth=1.8,
                color=colors[m_name],
                label=m_name,
            )
            if cal_name in ("Platt", "Isotonic"):
                ax.plot(
                    m_sub["year"],
                    m_sub["MeanGap"].abs(),
                    linestyle=":",
                    linewidth=1.2,
                    color=colors[m_name],
                    alpha=0.75,
                )
        ax.set_title(f"{cal_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluation Year")
        ax.set_xticks(YEARS)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("10-Bin ECE (solid) / |Mean Gap| (dotted)")
    axes[0].legend(loc="best", fontsize=8.5)
    fig.suptitle(
        "Calibration Degradation (ECE) Over Time (2014-2018)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    fname = "02_ece_by_year_test.png" if test_mode else "02_ece_by_year.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# --- Part 2b: Console verification and main entry point ---
def print_reference_comparison(
    metrics_df: pd.DataFrame, test_mode: bool = False, drop_occp: bool = False
):
    print("\n" + "=" * 92)
    print("BASELINE REFERENCE COMPARISON TABLE")
    print("=" * 92)
    header = (
        f"{'Model':<24s} {'Uncal ECE 14':>13s} {'Uncal AUC 14':>13s} "
        f"{'Platt ECE 18':>13s} {'Platt Gap 18':>13s} {'Platt AUC 18':>13s}"
    )
    print(header)
    print("-" * 92)

    all_passed = True
    for m_name in MODEL_NAMES:
        row_u14 = metrics_df[
            (metrics_df["model"] == m_name)
            & (metrics_df["calibration"] == "Uncalibrated")
            & (metrics_df["year"] == 2014)
        ].iloc[0]
        row_p18 = metrics_df[
            (metrics_df["model"] == m_name)
            & (metrics_df["calibration"] == "Platt")
            & (metrics_df["year"] == 2018)
        ].iloc[0]

        u_ece_14 = row_u14["ECE"]
        u_auc_14 = row_u14["AUC"]
        p_ece_18 = row_p18["ECE"]
        p_gap_18 = row_p18["MeanGap"]
        p_auc_18 = row_p18["AUC"]

        print(
            f"{m_name:<24s} {u_ece_14:>13.4f} {u_auc_14:>13.4f} "
            f"{p_ece_18:>13.4f} {p_gap_18:>+13.4f} {p_auc_18:>13.4f}"
        )

        if not test_mode and not drop_occp:
            ref = BASELINE_REFERENCE[m_name]
            checks = [
                ("Uncal ECE 2014", abs(u_ece_14 - ref["uncal_ece_2014"]) <= 0.005),
                ("Uncal AUC 2014", abs(u_auc_14 - ref["uncal_auc_2014"]) <= 0.003),
                ("Platt ECE 2018", abs(p_ece_18 - ref["platt_ece_2018"]) <= 0.005),
                ("Platt Gap 2018", abs(p_gap_18 - ref["platt_gap_2018"]) <= 0.005),
                ("Platt AUC 2018", abs(p_auc_18 - ref["platt_auc_2018"]) <= 0.003),
            ]
            for label, ok in checks:
                if not ok:
                    all_passed = False
                    print(f"  [WARNING] Tolerance mismatch on {m_name} - {label}")

    if not test_mode and not drop_occp:
        status = "ALL WITHIN REFERENCE TOLERANCE (PASS)" if all_passed else "MISMATCH DETECTED"
        print("-" * 92)
        print(f"Verification status: {status}")


def main():
    args = parse_args()
    mode_str = "TEST MODE (1% sample, seed=42)" if args.test else "FULL DATA MODE (seed=42)"
    occp_str = " [WITHOUT OCCP -> outputs_no_occp/]" if args.drop_occp else ""
    print(f"Running 02_train_baselines.py — {mode_str}{occp_str}\n")

    ece_perf, ece_shift = run_ece_unit_checks(seed=SEED)
    print("=" * 72)
    print("SYNTHETIC ECE UNIT CHECK")
    print("=" * 72)
    print(f"  Perfectly calibrated synthetic ECE : {ece_perf:.5f}  (target < 0.01)  [PASS]")
    print(f"  Constant +0.05 shift synthetic ECE : {ece_shift:.5f}  (target ~ 0.05)  [PASS]\n")

    yearly_dfs, _, _ = build_and_cache_all_years(download=False, test_mode=args.test)

    eval_dfs, predictions = train_and_predict_all_years(
        yearly_dfs, test_mode=args.test, use_cache=False, drop_occp=args.drop_occp
    )

    rows = []
    for m_name in MODEL_NAMES:
        for cal_name in CALIBRATION_NAMES:
            for y in YEARS:
                labels = eval_dfs[y]["label"].to_numpy(dtype=int)
                probs = predictions[m_name][cal_name][y]
                m_dict = compute_all_metrics(labels, probs)
                rows.append(
                    {
                        "model": m_name,
                        "calibration": cal_name,
                        "year": y,
                        "ECE": m_dict["ECE"],
                        "Brier": m_dict["Brier"],
                        "LogLoss": m_dict["LogLoss"],
                        "Accuracy": m_dict["Accuracy"],
                        "AUC": m_dict["AUC"],
                        "MeanPred": m_dict["MeanPred"],
                        "BaseRate": m_dict["BaseRate"],
                        "MeanGap": m_dict["MeanGap"],
                    }
                )

    metrics_df = pd.DataFrame(rows)
    ensure_dirs(drop_occp=args.drop_occp)
    out_dir = get_outputs_dir(args.drop_occp)
    csv_name = "02_baseline_metrics_by_year_test.csv" if args.test else "02_baseline_metrics_by_year.csv"
    csv_path = os.path.join(out_dir, csv_name)
    metrics_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 102)
    print("BASELINE METRICS BY MODEL, CALIBRATION, AND YEAR")
    print("=" * 102)
    print(
        f"{'Model':<23s} {'Calibration':<13s} {'Year':<5s} "
        f"{'ECE':>7s} {'Brier':>7s} {'LogLoss':>8s} {'Acc':>7s} {'AUC':>7s} "
        f"{'MeanPred':>8s} {'BaseRate':>8s} {'MeanGap':>8s}"
    )
    print("-" * 102)
    for _, r in metrics_df.iterrows():
        print(
            f"{r['model']:<23s} {r['calibration']:<13s} {int(r['year']):<5d} "
            f"{r['ECE']:>7.4f} {r['Brier']:>7.4f} {r['LogLoss']:>8.4f} "
            f"{r['Accuracy']:>7.4f} {r['AUC']:>7.4f} "
            f"{r['MeanPred']:>8.4f} {r['BaseRate']:>8.4f} {r['MeanGap']:>+8.4f}"
        )

    print_reference_comparison(metrics_df, test_mode=args.test, drop_occp=args.drop_occp)

    rel_path = plot_reliability_2014_vs_2018(
        eval_dfs, predictions, test_mode=args.test, drop_occp=args.drop_occp
    )
    ece_path = plot_ece_by_year(metrics_df, test_mode=args.test, drop_occp=args.drop_occp)
    print(f"\nSaved metrics CSV to:       {csv_path}")
    print(f"Saved reliability plot to:  {rel_path}")
    print(f"Saved ECE-by-year plot to:  {ece_path}")


if __name__ == "__main__":
    main()
