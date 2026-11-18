"""
03_evaluate_drift.py
Compute overall, sex-group, and race-group calibration and performance
metrics across 2014-2018 with 200-resample 95% bootstrap intervals.
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    CALIBRATION_NAMES,
    MERGED_RACE_NAME,
    MODEL_NAMES,
    SEED,
    YEARS,
    build_and_cache_all_years,
    ensure_dirs,
    get_outputs_dir,
    train_and_predict_all_years,
)
from metrics import bootstrap_ece_and_gap_multi, compute_all_metrics

# --- Part 1a: Group definitions and command-line arguments ---
N_BOOT = 200
SMALL_GROUP_THRESHOLD = 500

SEX_GROUPS = ["Male", "Female"]
RACE_GROUPS = [
    "White alone",
    "Asian alone",
    "Some Other Race alone",
    "Black alone",
    "Two or More Races",
    MERGED_RACE_NAME,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subgroup calibration drift and bootstrap uncertainty (2014-2018)."
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


# --- Part 1b: Group slicing and bootstrap drift table ---
def get_group_slices(df_year: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    n = len(df_year)
    slices = [("Overall", "All", np.ones(n, dtype=bool))]
    for s_grp in SEX_GROUPS:
        mask = (df_year["sex_group"] == s_grp).to_numpy()
        slices.append(("Sex", s_grp, mask))
    for r_grp in RACE_GROUPS:
        mask = (df_year["race_group"] == r_grp).to_numpy()
        slices.append(("Race", r_grp, mask))
    return slices


def compute_drift_table(eval_dfs: dict, predictions: dict) -> pd.DataFrame:
    rows = []

    for y_idx, y in enumerate(YEARS):
        df_y = eval_dfs[y]
        y_true_all = df_y["label"].to_numpy(dtype=int)
        slices = get_group_slices(df_y)

        for g_idx, (group_type, group_name, mask) in enumerate(slices):
            y_sub = y_true_all[mask]
            n_sub = int(len(y_sub))
            small_flag = bool(n_sub < SMALL_GROUP_THRESHOLD)

            probs_by_key = {}
            for m_name in MODEL_NAMES:
                for cal_name in CALIBRATION_NAMES:
                    probs_by_key[(m_name, cal_name)] = predictions[m_name][cal_name][y][mask]

            slice_seed = SEED + y_idx * 100 + g_idx
            boot_res = bootstrap_ece_and_gap_multi(
                y_sub,
                probs_by_key,
                n_boot=N_BOOT,
                seed=slice_seed,
            )

            for m_name in MODEL_NAMES:
                for cal_name in CALIBRATION_NAMES:
                    p_sub = probs_by_key[(m_name, cal_name)]
                    m_dict = compute_all_metrics(y_sub, p_sub)
                    b_dict = boot_res[(m_name, cal_name)]

                    rows.append(
                        {
                            "model": m_name,
                            "calibration": cal_name,
                            "year": y,
                            "group_type": group_type,
                            "group": group_name,
                            "n": n_sub,
                            "small_group_flag": small_flag,
                            "ECE": m_dict["ECE"],
                            "ECE_low": b_dict["ece_low"],
                            "ECE_high": b_dict["ece_high"],
                            "Brier": m_dict["Brier"],
                            "LogLoss": m_dict["LogLoss"],
                            "Accuracy": m_dict["Accuracy"],
                            "AUC": m_dict["AUC"],
                            "MeanPred": m_dict["MeanPred"],
                            "BaseRate": m_dict["BaseRate"],
                            "MeanGap": m_dict["MeanGap"],
                            "MeanGap_low": b_dict["gap_low"],
                            "MeanGap_high": b_dict["gap_high"],
                        }
                    )

    return pd.DataFrame(rows)


# --- Part 2a: Console summary and group row-count checks ---
def print_platt_mean_gap_table(df_metrics: pd.DataFrame):
    print("=" * 120)
    print("PER-GROUP MEAN GAP BY YEAR FOR PLATT-CALIBRATED MODELS (with n and 95% Bootstrap CIs)")
    print("=" * 120)

    platt_df = df_metrics[df_metrics["calibration"] == "Platt"]

    for m_name in MODEL_NAMES:
        print(f"\nModel: {m_name} (Calibration: Platt)")
        print(
            f"{'Group Type':<10s} {'Group':<24s} "
            f"{'2014 Gap (n)':>18s} {'2015 Gap (n)':>18s} {'2016 Gap (n)':>18s} "
            f"{'2017 Gap (n)':>18s} {'2018 Gap (n)':>18s} {'2018 Gap 95% CI':>21s} {'Flag':<6s}"
        )
        print("-" * 145)
        m_sub = platt_df[platt_df["model"] == m_name]
        for g_type, g_name in (
            [("Overall", "All")]
            + [("Sex", g) for g in SEX_GROUPS]
            + [("Race", g) for g in RACE_GROUPS]
        ):
            g_rows = m_sub[
                (m_sub["group_type"] == g_type) & (m_sub["group"] == g_name)
            ].set_index("year")
            cells = []
            any_small = bool(g_rows["small_group_flag"].any())
            for y in YEARS:
                r_y = g_rows.loc[y]
                cells.append(f"{r_y['MeanGap']:+7.4f} ({int(r_y['n']):,d})")
            r18 = g_rows.loc[2018]
            ci18 = f"[{r18['MeanGap_low']:+7.4f}, {r18['MeanGap_high']:+7.4f}]"
            flag_str = "[n<500]" if any_small else ""
            print(
                f"{g_type:<10s} {g_name:<24s} "
                f"{cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s} "
                f"{cells[3]:>18s} {cells[4]:>18s} {ci18:>21s} {flag_str:<6s}"
            )

    print("\n" + "=" * 120)
    print("2018 PLATT-CALIBRATED PAIRWISE GROUP DIFFERENCES (>= 0.02 WITH NON-OVERLAPPING 95% CIs)")
    print("=" * 120)
    for m_name in MODEL_NAMES:
        m18 = platt_df[(platt_df["model"] == m_name) & (platt_df["year"] == 2018)]
        qualifying_pairs = []
        for g_type in ["Sex", "Race"]:
            sub = m18[m18["group_type"] == g_type].reset_index(drop=True)
            for i in range(len(sub)):
                for j in range(i + 1, len(sub)):
                    r_a = sub.iloc[i]
                    r_b = sub.iloc[j]
                    diff = abs(float(r_a["MeanGap"]) - float(r_b["MeanGap"]))
                    non_overlap = (float(r_a["MeanGap_high"]) < float(r_b["MeanGap_low"])) or (
                        float(r_b["MeanGap_high"]) < float(r_a["MeanGap_low"])
                    )
                    if diff >= 0.02 and non_overlap:
                        qualifying_pairs.append(
                            f"{g_type}: {r_a['group']} ({r_a['MeanGap']:+.4f} [{r_a['MeanGap_low']:+.4f}, {r_a['MeanGap_high']:+.4f}], n={int(r_a['n']):,d}) vs "
                            f"{r_b['group']} ({r_b['MeanGap']:+.4f} [{r_b['MeanGap_low']:+.4f}, {r_b['MeanGap_high']:+.4f}], n={int(r_b['n']):,d}), |diff|={diff:.4f}"
                        )
        print(f"  {m_name:<25s}: {len(qualifying_pairs)} qualifying pair(s)")
        for qp in qualifying_pairs:
            print(f"    - {qp}")


def verify_and_print_group_counts(df_metrics: pd.DataFrame):
    print("\n" + "=" * 88)
    print("GROUP ROW COUNT INTEGRITY CHECKS (n sums per evaluation year)")
    print("=" * 88)
    sub = df_metrics[
        (df_metrics["model"] == MODEL_NAMES[0]) & (df_metrics["calibration"] == "Platt")
    ]
    for y in YEARS:
        y_sub = sub[sub["year"] == y]
        n_overall = int(y_sub[y_sub["group_type"] == "Overall"]["n"].iloc[0])
        n_sex_sum = int(y_sub[y_sub["group_type"] == "Sex"]["n"].sum())
        n_race_sum = int(y_sub[y_sub["group_type"] == "Race"]["n"].sum())
        assert n_sex_sum == n_overall, f"Sex n sum {n_sex_sum} != {n_overall} in {y}"
        assert n_race_sum == n_overall, f"Race n sum {n_race_sum} != {n_overall} in {y}"
        print(
            f"  Year {y}: Overall n = {n_overall:>7,d} | "
            f"Sex sum = {n_sex_sum:>7,d} (OK) | Race sum = {n_race_sum:>7,d} (OK)"
        )


# --- Part 2b: Plot subgroup mean gap by year and save outputs ---
def plot_mean_gap_by_group(
    df_metrics: pd.DataFrame, test_mode: bool = False, drop_occp: bool = False
) -> str:
    ensure_dirs(drop_occp=drop_occp)
    out_dir = get_outputs_dir(drop_occp)
    platt_df = df_metrics[df_metrics["calibration"] == "Platt"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), dpi=200, sharex=True, sharey=True)

    sex_colors = {"Male": "#1f77b4", "Female": "#e377c2"}
    race_colors = {
        "White alone": "#1f77b4",
        "Asian alone": "#2ca02c",
        "Some Other Race alone": "#ff7f0e",
        "Black alone": "#d62728",
        "Two or More Races": "#9467bd",
        MERGED_RACE_NAME: "#8c564b",
    }

    for col_idx, m_name in enumerate(MODEL_NAMES):
        m_sub = platt_df[platt_df["model"] == m_name]

        ax_sex = axes[0, col_idx]
        ax_sex.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        for s_grp in SEX_GROUPS:
            g_sub = m_sub[(m_sub["group_type"] == "Sex") & (m_sub["group"] == s_grp)].sort_values("year")
            yrs = g_sub["year"].to_numpy()
            gap = g_sub["MeanGap"].to_numpy()
            low = g_sub["MeanGap_low"].to_numpy()
            high = g_sub["MeanGap_high"].to_numpy()
            ax_sex.plot(yrs, gap, marker="o", linewidth=1.8, color=sex_colors[s_grp], label=s_grp)
            ax_sex.fill_between(yrs, low, high, color=sex_colors[s_grp], alpha=0.18)
        ax_sex.set_title(f"{m_name}\n(By Sex)", fontsize=10.5, fontweight="bold")
        ax_sex.set_xticks(YEARS)
        ax_sex.grid(True, linestyle="--", alpha=0.4)

        ax_race = axes[1, col_idx]
        ax_race.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        for r_grp in RACE_GROUPS:
            g_sub = m_sub[(m_sub["group_type"] == "Race") & (m_sub["group"] == r_grp)].sort_values("year")
            yrs = g_sub["year"].to_numpy()
            gap = g_sub["MeanGap"].to_numpy()
            low = g_sub["MeanGap_low"].to_numpy()
            high = g_sub["MeanGap_high"].to_numpy()
            ax_race.plot(yrs, gap, marker="s", markersize=4, linewidth=1.6, color=race_colors[r_grp], label=r_grp)
            ax_race.fill_between(yrs, low, high, color=race_colors[r_grp], alpha=0.12)
        ax_race.set_title("(By Race)", fontsize=10)
        ax_race.set_xlabel("Evaluation Year")
        ax_race.set_xticks(YEARS)
        ax_race.grid(True, linestyle="--", alpha=0.4)

    axes[0, 0].set_ylabel("Mean Gap (Pred - Actual)")
    axes[1, 0].set_ylabel("Mean Gap (Pred - Actual)")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    axes[1, 0].legend(loc="lower left", fontsize=7)

    fig.suptitle(
        "Subgroup Mean Gap by Year (2014-2018) for Platt-Calibrated Models (with 95% Bootstrap CIs)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    fname = (
        "03_mean_gap_by_group_test.png"
        if test_mode
        else "03_mean_gap_by_group.png"
    )
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()

    yearly_dfs, _, _ = build_and_cache_all_years(download=False, test_mode=args.test)
    eval_dfs, predictions = train_and_predict_all_years(
        yearly_dfs, test_mode=args.test, use_cache=True, drop_occp=args.drop_occp
    )

    df_metrics = compute_drift_table(eval_dfs, predictions)

    print_platt_mean_gap_table(df_metrics)
    verify_and_print_group_counts(df_metrics)

    ensure_dirs(drop_occp=args.drop_occp)
    out_dir = get_outputs_dir(args.drop_occp)
    csv_name = (
        "03_drift_metrics_by_group_test.csv" if args.test else "03_drift_metrics_by_group.csv"
    )
    csv_path = os.path.join(out_dir, csv_name)
    df_metrics.to_csv(csv_path, index=False)

    fig_path = plot_mean_gap_by_group(
        df_metrics, test_mode=args.test, drop_occp=args.drop_occp
    )
    print(f"\nSaved group metrics CSV to: {csv_path} ({len(df_metrics)} rows)")
    print(f"Saved subgroup drift plot:  {fig_path}")


if __name__ == "__main__":
    main()
