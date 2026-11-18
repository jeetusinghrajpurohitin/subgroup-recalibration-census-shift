"""
04_run_recalibration.py
Post-shift recalibration across sample fractions (1%, 2%, 5%, 10%) and 30 paired draws
for None, BaseRate, Platt, Isotonic, Beta, and VennAbers.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from calibrators import (
    BaseRateCalibrator,
    BetaCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    VennAbersCalibrator,
    venn_abers_slow,
)
from common import (
    EVAL_YEARS,
    MODEL_NAMES,
    SEED,
    SMALL_GROUP_THRESHOLD,
    build_and_cache_all_years,
    ensure_dirs,
    get_group_slices,
    get_outputs_dir,
    train_and_predict_all_years,
)
from metrics import compute_brier, compute_ece, compute_mean_gap

# --- Part 1a: Sample fractions, methods, and command-line arguments ---
FRACTIONS = [0.01, 0.02, 0.05, 0.10]
METHODS = ["None", "BaseRate", "Platt", "Isotonic", "Beta", "VennAbers"]
N_REPEATS_FULL = 30
N_REPEATS_TEST = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-shift recalibration with small recent samples.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: 1%% cached sample per year and 5 repeats.",
    )
    parser.add_argument(
        "--drop-occp",
        action="store_true",
        help="Remove OCCP from ACSIncome features and write all outputs to outputs_no_occp/.",
    )
    return parser.parse_args()


# --- Part 1b: Sample drawing and method fitting on R ---
def draw_seed(year: int, fraction: float, repeat: int) -> list[int]:
    return [SEED, year, int(round(fraction * 100)), repeat]


def draw_recalibration_sample(n_rows: int, year: int, fraction: float, repeat: int) -> np.ndarray:
    rng = np.random.default_rng(draw_seed(year, fraction, repeat))
    n_r = int(round(fraction * n_rows))
    chosen = rng.choice(n_rows, size=n_r, replace=False)
    in_r = np.zeros(n_rows, dtype=bool)
    in_r[chosen] = True
    return in_r


def fit_and_predict_methods(raw_r, platt_r, y_r, raw_eval, platt_eval) -> dict:
    out = {"None": platt_eval}
    out["BaseRate"] = BaseRateCalibrator().fit(platt_r, y_r).predict(platt_eval)
    out["Isotonic"] = IsotonicCalibrator().fit(raw_r, y_r).predict(raw_eval)
    two_classes = len(np.unique(y_r)) > 1
    for name, calibrator in (
        ("Platt", PlattCalibrator),
        ("Beta", BetaCalibrator),
        ("VennAbers", VennAbersCalibrator),
    ):
        out[name] = calibrator().fit(raw_r, y_r).predict(raw_eval) if two_classes else None
    return out


def check_venn_abers_against_slow(eval_dfs: dict, predictions: dict):
    year = EVAL_YEARS[-1]
    raw = predictions[MODEL_NAMES[0]]["Uncalibrated"][year]
    y = eval_dfs[year]["label"].to_numpy(dtype=int)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), size=350, replace=False)
    cal, test = idx[:300], idx[300:]
    fast = VennAbersCalibrator().fit(raw[cal], y[cal])
    p0_fast, p1_fast = fast.predict_p0_p1(raw[test])
    p0_slow, p1_slow = venn_abers_slow(raw[cal], y[cal], raw[test])
    max_diff = max(np.abs(p0_fast - p0_slow).max(), np.abs(p1_fast - p1_slow).max())
    print(f"Venn-Abers check (package vs slow literal version): max |difference| = {max_diff:.2e}")


# --- Part 2a: Run all paired recalibration draws and summarize across repeats ---
def run_all_draws(eval_dfs: dict, predictions: dict, n_repeats: int) -> tuple[pd.DataFrame, int]:
    rows = []
    skipped_fits = 0

    for year in EVAL_YEARS:
        df_y = eval_dfs[year]
        y_all = df_y["label"].to_numpy(dtype=int)
        n_rows = len(y_all)
        group_slices = get_group_slices(df_y)

        for fraction in FRACTIONS:
            for rep in range(n_repeats):
                in_r = draw_recalibration_sample(n_rows, year, fraction, rep)
                keep = ~in_r
                y_r, y_eval = y_all[in_r], y_all[keep]
                eval_groups = [(gt, g, m[keep]) for gt, g, m in group_slices]

                for model in MODEL_NAMES:
                    raw = predictions[model]["Uncalibrated"][year]
                    platt14 = predictions[model]["Platt"][year]
                    probs = fit_and_predict_methods(
                        raw[in_r], platt14[in_r], y_r, raw[keep], platt14[keep]
                    )

                    for method in METHODS:
                        p = probs[method]
                        if p is None:
                            skipped_fits += 1
                        for group_type, group, g_mask in eval_groups:
                            n_eval = int(g_mask.sum())
                            if p is None or n_eval == 0:
                                ece = brier = gap = np.nan
                            else:
                                y_g, p_g = y_eval[g_mask], p[g_mask]
                                ece = compute_ece(y_g, p_g)
                                brier = compute_brier(y_g, p_g)
                                gap = compute_mean_gap(y_g, p_g)
                            rows.append(
                                {
                                    "model": model,
                                    "year": year,
                                    "fraction": fraction,
                                    "repeat": rep,
                                    "method": method,
                                    "group_type": group_type,
                                    "group": group,
                                    "n_eval": n_eval,
                                    "ECE": ece,
                                    "Brier": brier,
                                    "MeanGap": gap,
                                    "small_group_flag": n_eval < SMALL_GROUP_THRESHOLD,
                                }
                            )
        print(f"  finished year {year}")

    return pd.DataFrame(rows), skipped_fits


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "year", "fraction", "method", "group_type", "group"]
    grouped = runs.groupby(keys, sort=False)
    summary = grouped["n_eval"].mean().rename("n_eval_mean").to_frame()
    summary["n_repeats_used"] = grouped["ECE"].count()
    summary["small_group_flag"] = grouped["small_group_flag"].any()
    for metric in ["ECE", "Brier", "MeanGap"]:
        summary[f"{metric}_mean"] = grouped[metric].mean()
        summary[f"{metric}_p2.5"] = grouped[metric].quantile(0.025)
        summary[f"{metric}_p97.5"] = grouped[metric].quantile(0.975)
    return summary.reset_index()


# --- Part 2b: Console overview and main entry point ---
def print_overview(summary: pd.DataFrame):
    overall = summary[summary["group_type"] == "Overall"]

    print("\nOVERALL MEAN ECE IN 2018 (mean over repeats), rows = method, columns = fraction")
    for model in MODEL_NAMES:
        sub = overall[(overall["model"] == model) & (overall["year"] == 2018)]
        table = sub.pivot(index="method", columns="fraction", values="ECE_mean").reindex(METHODS)
        print(f"\n{model}")
        print(table.round(4).to_string())

    br = overall[overall["method"] == "BaseRate"]
    print(
        "\nBASERATE CHECK: overall MeanGap on evaluation rows, all models/years/fractions: "
        f"mean {br['MeanGap_mean'].mean():+.4f}, "
        f"largest |mean| {br['MeanGap_mean'].abs().max():.4f}"
    )


def main():
    args = parse_args()
    n_repeats = N_REPEATS_TEST if args.test else N_REPEATS_FULL
    start = time.time()

    yearly_dfs, _, _ = build_and_cache_all_years(download=False, test_mode=args.test)
    eval_dfs, predictions = train_and_predict_all_years(
        yearly_dfs, test_mode=args.test, use_cache=True, drop_occp=args.drop_occp
    )

    if args.test:
        check_venn_abers_against_slow(eval_dfs, predictions)
    print(f"Running 04_run_recalibration.py: {n_repeats} repeats, fractions {FRACTIONS}, methods {METHODS}")
    print("Sample R seed per draw = [42, year, fraction in %, repeat] (same R for all methods and models)")
    runs, skipped_fits = run_all_draws(eval_dfs, predictions, n_repeats)
    summary = summarize_runs(runs)

    print_overview(summary)
    if skipped_fits:
        print(f"\nWARNING: {skipped_fits} method fit(s) skipped because R held only one class (NaN rows).")
    n_small = summary.loc[summary["small_group_flag"], ["year", "group"]].drop_duplicates()
    if len(n_small):
        print(f"Groups with n_eval < {SMALL_GROUP_THRESHOLD} (flagged in CSVs): {len(n_small)} year-group pairs")

    ensure_dirs(drop_occp=args.drop_occp)
    out_dir = get_outputs_dir(args.drop_occp)
    suffix = "_test" if args.test else ""
    runs_path = os.path.join(out_dir, f"04_recalibration_runs{suffix}.csv")
    summary_path = os.path.join(out_dir, f"04_recalibration_summary{suffix}.csv")
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {runs_path} ({len(runs)} rows) and {summary_path} ({len(summary)} rows)")
    print(f"Elapsed: {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
