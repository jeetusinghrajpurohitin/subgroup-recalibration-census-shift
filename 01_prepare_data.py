"""
01_prepare_data.py
Load and cache ACSIncome data for California (2014-2018), merge small race categories,
and save 01_dataset_summary_by_year.csv and 01_positive_rate_by_year.png.
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import (
    MERGED_RACE_NAME,
    OUTPUTS_DIR,
    RACE_CODE_MAP,
    SEED,
    YEARS,
    build_and_cache_all_years,
    ensure_dirs,
    split_2014_data,
)


# --- Part 1a: Command-line arguments ---
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and summarize California ACSIncome data (2014-2018)."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download raw Census files if not already present in data/raw/.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode using a 1%% random sample per year (seed 42).",
    )
    return parser.parse_args()


# --- Part 1b: Console summary of race merging and yearly counts ---
def print_race_rule_summary(race_shares: pd.Series, merged_categories: list[str]):
    print("=" * 72)
    print("5-YEAR COMBINED RAW RACE SHARES (2014-2018)")
    print("=" * 72)
    for code in sorted(RACE_CODE_MAP.keys()):
        cat_name = RACE_CODE_MAP[code]
        share = float(race_shares.get(cat_name, 0.0)) * 100.0
        status = f"-> merged into '{MERGED_RACE_NAME}'" if cat_name in merged_categories else "-> kept separate"
        print(f"  [{code}] {cat_name:<50s} {share:6.3f}%  {status}")
    print()


def print_yearly_summaries(yearly_dfs: dict):
    print("=" * 72)
    print("YEARLY SUMMARY: ROW COUNTS AND POSITIVE RATES (INCOME > $50,000)")
    print("=" * 72)
    print(f"{'Year':<8s} {'Rows':>10s} {'Positives':>12s} {'Positive Rate':>15s}")
    print("-" * 48)
    for y in YEARS:
        df = yearly_dfs[y]
        n_rows = len(df)
        n_pos = int(df["label"].sum())
        pos_rate = (n_pos / n_rows) * 100.0 if n_rows > 0 else 0.0
        print(f"{y:<8d} {n_rows:>10,d} {n_pos:>12,d} {pos_rate:>14.2f}%")
    print()

    race_order = [
        "White alone",
        "Asian alone",
        "Some Other Race alone",
        "Black alone",
        "Two or More Races",
        MERGED_RACE_NAME,
    ]

    for y in YEARS:
        df = yearly_dfs[y]
        n_total = len(df)
        pos_rate = df["label"].mean() * 100.0
        print("=" * 72)
        print(f"YEAR {y} DETAIL  (Total rows = {n_total:,d}, Positive rate = {pos_rate:.2f}%)")
        print("=" * 72)

        print("  Sex groups:")
        sex_counts = df["sex_group"].value_counts()
        sex_sum = 0
        for sex_name in ["Male", "Female"]:
            cnt = int(sex_counts.get(sex_name, 0))
            sex_sum += cnt
            pct = (cnt / n_total) * 100.0 if n_total > 0 else 0.0
            grp_pos = df.loc[df["sex_group"] == sex_name, "label"].mean() * 100.0 if cnt > 0 else 0.0
            print(f"    {sex_name:<30s} n = {cnt:>7,d} ({pct:6.2f}%) | pos rate = {grp_pos:5.2f}%")
        assert sex_sum == n_total, f"Sex count sum ({sex_sum}) != total rows ({n_total}) in {y}"
        print(f"    [Check] Sex counts sum:        {sex_sum:>7,d} == {n_total:,d} (OK)")

        print("\n  Race groups:")
        race_counts = df["race_group"].value_counts()
        race_sum = 0
        displayed_races = [r for r in race_order if r in race_counts] + [
            r for r in race_counts.index if r not in race_order
        ]
        for race_name in displayed_races:
            cnt = int(race_counts.get(race_name, 0))
            race_sum += cnt
            pct = (cnt / n_total) * 100.0 if n_total > 0 else 0.0
            grp_pos = df.loc[df["race_group"] == race_name, "label"].mean() * 100.0 if cnt > 0 else 0.0
            flag = " [n < 500]" if cnt < 500 else ""
            print(f"    {race_name:<30s} n = {cnt:>7,d} ({pct:6.2f}%) | pos rate = {grp_pos:5.2f}%{flag}")
        assert race_sum == n_total, f"Race count sum ({race_sum}) != total rows ({n_total}) in {y}"
        print(f"    [Check] Race counts sum:       {race_sum:>7,d} == {n_total:,d} (OK)\n")


# --- Part 2a: Save yearly summary CSV ---
def save_01_dataset_summary_by_year(yearly_dfs: dict, test_mode: bool = False) -> str:
    ensure_dirs()
    race_order = [
        "White alone",
        "Asian alone",
        "Some Other Race alone",
        "Black alone",
        "Two or More Races",
        MERGED_RACE_NAME,
    ]
    rows = []
    for y in YEARS:
        df = yearly_dfs[y]
        n_total = len(df)
        rows.append(
            {
                "year": y,
                "group_type": "Overall",
                "group": "All",
                "n": n_total,
                "share_pct": 100.0,
                "pos_rate": float(df["label"].mean()),
                "pos_rate_pct": float(df["label"].mean() * 100.0),
                "small_group_flag": bool(n_total < 500),
            }
        )
        for s_name in ["Male", "Female"]:
            sub = df[df["sex_group"] == s_name]
            cnt = len(sub)
            rows.append(
                {
                    "year": y,
                    "group_type": "Sex",
                    "group": s_name,
                    "n": cnt,
                    "share_pct": (cnt / n_total) * 100.0 if n_total > 0 else 0.0,
                    "pos_rate": float(sub["label"].mean()) if cnt > 0 else 0.0,
                    "pos_rate_pct": float(sub["label"].mean() * 100.0) if cnt > 0 else 0.0,
                    "small_group_flag": bool(cnt < 500),
                }
            )
        for r_name in race_order:
            sub = df[df["race_group"] == r_name]
            cnt = len(sub)
            rows.append(
                {
                    "year": y,
                    "group_type": "Race",
                    "group": r_name,
                    "n": cnt,
                    "share_pct": (cnt / n_total) * 100.0 if n_total > 0 else 0.0,
                    "pos_rate": float(sub["label"].mean()) if cnt > 0 else 0.0,
                    "pos_rate_pct": float(sub["label"].mean() * 100.0) if cnt > 0 else 0.0,
                    "small_group_flag": bool(cnt < 500),
                }
            )
    out_df = pd.DataFrame(rows)
    if test_mode:
        test_dir = os.path.join(OUTPUTS_DIR, "test_runs")
        os.makedirs(test_dir, exist_ok=True)
        out_path = os.path.join(test_dir, "01_dataset_summary_by_year_test.csv")
    else:
        out_path = os.path.join(OUTPUTS_DIR, "01_dataset_summary_by_year.csv")
    out_df.to_csv(out_path, index=False)
    return out_path


# --- Part 2b: Plot overall positive rate by year ---
def plot_01_positive_rate_by_year(yearly_dfs: dict, test_mode: bool = False) -> str:
    ensure_dirs()
    years = YEARS
    pos_rates = [yearly_dfs[y]["label"].mean() * 100.0 for y in years]

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    ax.plot(years, pos_rates, marker="o", linewidth=2.0, color="#1f77b4")
    for x, y_val in zip(years, pos_rates):
        ax.annotate(
            f"{y_val:.2f}%",
            (x, y_val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    ax.set_title("ACSIncome (CA): Positive Rate (> $50,000) by Year (2014-2018)")
    ax.set_xlabel("Survey Year")
    ax.set_ylabel("Positive Rate (%)")
    ax.set_xticks(years)
    ax.set_xlim(2013.65, 2018.35)
    y_min = max(0.0, min(pos_rates) - 2.0)
    y_max = min(100.0, max(pos_rates) + 2.5)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    filename = "01_positive_rate_by_year_test.png" if test_mode else "01_positive_rate_by_year.png"
    out_path = os.path.join(OUTPUTS_DIR, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# --- Part 2c: Main entry point ---
def main():
    args = parse_args()
    mode_str = "TEST MODE (1% sample, seed=42)" if args.test else "FULL DATA MODE (seed=42)"
    print(f"Running 01_prepare_data.py — {mode_str}\n")

    yearly_dfs, race_shares, merged_categories = build_and_cache_all_years(
        download=args.download,
        test_mode=args.test,
    )

    if race_shares is not None and merged_categories is not None:
        print_race_rule_summary(race_shares, merged_categories)

    print_yearly_summaries(yearly_dfs)

    train_df, cal_df, test_df = split_2014_data(yearly_dfs[2014])
    print("=" * 72)
    print(f"2014 STRATIFIED SPLIT CHECK (seed={SEED})")
    print("=" * 72)
    print(f"  Train (60%):       {len(train_df):>8,d} rows (pos rate = {train_df['label'].mean()*100:.2f}%)")
    print(f"  Calibration (20%): {len(cal_df):>8,d} rows (pos rate = {cal_df['label'].mean()*100:.2f}%)")
    print(f"  Test (20%):        {len(test_df):>8,d} rows (pos rate = {test_df['label'].mean()*100:.2f}%)")
    print(f"  Total 2014:        {len(train_df) + len(cal_df) + len(test_df):>8,d} rows\n")

    summary_csv_path = save_01_dataset_summary_by_year(yearly_dfs, test_mode=args.test)
    print(f"Saved dataset summary CSV to: {summary_csv_path}")

    fig_path = plot_01_positive_rate_by_year(yearly_dfs, test_mode=args.test)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()
