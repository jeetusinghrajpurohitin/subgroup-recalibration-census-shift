# Subgroup Recalibration of Classical Classifiers Under Temporal Census Shift

This repository evaluates the probability calibration of four classical classifiers (`Logistic Regression`, `Random Forest`, `HistGradientBoostingClassifier`, and `Gaussian Naive Bayes`) under temporal distribution shift on U.S. Census American Community Survey (ACS) Public Use Microdata Sample (PUMS) data for California from 2014 to 2018, accessed via `folktables` (`ACSIncome` primary task and `ACSEmployment` replication task). Models and baseline calibrators (`Platt` and `Isotonic`) are fitted on 2014 data (`60%` train / `20%` calibration / `20%` test, `seed=42`) and evaluated without retraining across 2014–2018 overall and across sex and race subgroups, alongside post-hoc recalibration on small recent labeled samples (`1%`, `2%`, `5%`, `10%` across 30 repeats for `None`, `BaseRate`, `Platt`, `Isotonic`, `Beta`, and `VennAbers`), thresholded bootstrap monitoring rules, group-specific vs. shared recalibration, `ACSEmployment` replication, and robustness to the 2018 occupation-code (`OCCP`) revision.

## Repository Structure

- `common.py`, `calibrators.py`, `metrics.py`: Shared data loading, preprocessing, post-hoc calibrator classes, and evaluation metrics.
- `01_prepare_data.py`: Loads and caches 2014–2018 California `ACSIncome` data, applies the 5-year `<1%` race-merging rule, and saves `01_dataset_summary_by_year.csv` and `01_positive_rate_by_year.png`.
- `02_train_baselines.py`: Trains the four baseline classifiers on the 2014 train slice, fits `2014-Platt` and `2014-Isotonic` calibrators on the 2014 calibration slice, and saves `02_baseline_metrics_by_year.csv`, `02_reliability_2014_vs_2018.png`, and `02_ece_by_year.png`.
- `03_evaluate_drift.py`: Computes overall, sex-group, and race-group drift metrics across 2014–2018 with 200-resample 95% bootstrap intervals and saves `03_drift_metrics_by_group.csv` and `03_mean_gap_by_group.png`.
- `04_run_recalibration.py`: Runs 30-repeat paired post-shift recalibration across sample fractions `1%, 2%, 5%, 10%` and saves `04_recalibration_runs.csv` and `04_recalibration_summary.csv`.
- `05_analyze_and_plot.py`: Generates final summary and paired-comparison CSV tables (`05_table1..05_table3_*.csv`, `05_paired_method_comparisons.csv`), extension CSVs (`--monitoring`, `--group-vs-shared`, `--employment`, `--robustness`), and Figures 1–8 (`05_fig1..05_fig8_*.png`).
- `outputs/`: All full-data CSV tables and 200-dpi PNG figures, prefixed with `01_`–`05_` matching the script that produced each file.
- `outputs_no_occp/`: Full-data CSV tables and PNG figures for the 9-feature sensitivity pipeline run without `OCCP` (`--drop-occp`).
- `requirements.txt`: Pinned Python package versions.

## Installation

Python `3.14.6` environment with pinned dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Raw Census data (`data/`) is not included in this repository and is downloaded by the user on the first run of `01_prepare_data.py` using the `--download` flag:

```bash
python3 01_prepare_data.py --download
python3 02_train_baselines.py
python3 03_evaluate_drift.py
python3 04_run_recalibration.py
python3 05_analyze_and_plot.py
python3 05_analyze_and_plot.py --monitoring
python3 05_analyze_and_plot.py --group-vs-shared
python3 05_analyze_and_plot.py --employment
```

To run the 9-feature `--drop-occp` robustness pipeline and generate the 2017 and no-`OCCP` comparison CSVs:

```bash
python3 02_train_baselines.py --drop-occp
python3 03_evaluate_drift.py --drop-occp
python3 04_run_recalibration.py --drop-occp
python3 05_analyze_and_plot.py --drop-occp
python3 05_analyze_and_plot.py --robustness
```

License: MIT (see `LICENSE`).
