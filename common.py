"""
Shared constants, data loading, demographic group rules, and model helpers.
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Part 1a: Years, paths, and feature lists ---
YEARS = [2014, 2015, 2016, 2017, 2018]
TRAIN_YEAR = 2014
EVAL_YEARS = [2015, 2016, 2017, 2018]
STATE = "CA"
HORIZON = "1-Year"
SURVEY = "person"
SEED = 42
TEST_SAMPLE_FRAC = 0.01

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUTS_DIR = "outputs"
OUTPUTS_NO_OCCP_DIR = "outputs_no_occp"

FEATURE_COLS = [
    "AGEP",
    "COW",
    "SCHL",
    "MAR",
    "OCCP",
    "POBP",
    "RELP",
    "WKHP",
    "SEX",
    "RAC1P",
]
CONTINUOUS_COLS = ["AGEP", "WKHP"]
CATEGORICAL_COLS = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]


def get_outputs_dir(drop_occp: bool = False) -> str:
    return OUTPUTS_NO_OCCP_DIR if drop_occp else OUTPUTS_DIR


def get_feature_cols(drop_occp: bool = False) -> list[str]:
    if drop_occp:
        return [c for c in FEATURE_COLS if c != "OCCP"]
    return list(FEATURE_COLS)


def get_categorical_cols(drop_occp: bool = False) -> list[str]:
    if drop_occp:
        return [c for c in CATEGORICAL_COLS if c != "OCCP"]
    return list(CATEGORICAL_COLS)


# --- Part 1b: Sex and race group mappings ---
SEX_CODE_MAP = {
    1: "Male",
    2: "Female",
}

RACE_CODE_MAP = {
    1: "White alone",
    2: "Black alone",
    3: "American Indian alone",
    4: "Alaska Native alone",
    5: "American Indian and Alaska Native not specified",
    6: "Asian alone",
    7: "Native Hawaiian and Other Pacific Islander alone",
    8: "Some Other Race alone",
    9: "Two or More Races",
}

KEEP_SEPARATE_RACES = {
    "Some Other Race alone",
    "Two or More Races",
}
MERGED_RACE_NAME = "Other (merged)"
RACE_MIN_SHARE = 0.01


# --- Part 1c: Model and recalibration settings ---
MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosted Trees",
    "Gaussian Naive Bayes",
]

CALIBRATION_NAMES = ["Uncalibrated", "Platt", "Isotonic"]

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

FRACTIONS = [0.01, 0.02, 0.05, 0.10]
METHODS = ["None", "BaseRate", "Platt", "Isotonic", "Beta", "VennAbers"]
N_REPEATS_FULL = 30
N_REPEATS_TEST = 5


# --- Part 2a: Load raw Census tables and merge small race categories ---
def ensure_dirs(drop_occp: bool = False):
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(get_outputs_dir(drop_occp), exist_ok=True)


def get_processed_path(year: int, test_mode: bool = False) -> str:
    suffix = "_test" if test_mode else ""
    return os.path.join(PROCESSED_DIR, f"acs_income_{STATE.lower()}_{year}{suffix}.csv")


def load_raw_year_from_folktables(year: int, download: bool = False) -> pd.DataFrame:
    from folktables import ACSDataSource, ACSIncome

    data_source = ACSDataSource(
        survey_year=str(year),
        horizon=HORIZON,
        survey=SURVEY,
        root_dir=RAW_DIR,
    )
    ca_raw = data_source.get_data(states=[STATE], download=download)
    features, labels, _ = ACSIncome.df_to_numpy(ca_raw)

    df = pd.DataFrame(features, columns=FEATURE_COLS)
    df["label"] = labels.astype(int)
    df["sex_group"] = df["SEX"].astype(int).map(SEX_CODE_MAP)
    df["race_raw"] = df["RAC1P"].astype(int).map(RACE_CODE_MAP)
    return df


def apply_race_merging_rule(yearly_dfs: dict) -> tuple[dict, pd.Series, list[str]]:
    """Merge race categories below 1% of combined 2014-2018 rows into 'Other (merged)'."""
    combined_races = pd.concat(
        [df["race_raw"] for df in yearly_dfs.values()], ignore_index=True
    )
    race_counts = combined_races.value_counts()
    race_shares = race_counts / len(combined_races)

    merged_categories = []
    for code in sorted(RACE_CODE_MAP.keys()):
        cat_name = RACE_CODE_MAP[code]
        share = float(race_shares.get(cat_name, 0.0))
        if share < RACE_MIN_SHARE and cat_name not in KEEP_SEPARATE_RACES:
            merged_categories.append(cat_name)

    merged_set = set(merged_categories)
    for year, df in yearly_dfs.items():
        df["race_group"] = df["race_raw"].apply(
            lambda r: MERGED_RACE_NAME if r in merged_set else r
        )

    return yearly_dfs, race_shares, merged_categories


# --- Part 2b: Build and load cached yearly CSV files ---
def build_and_cache_all_years(
    download: bool = False,
    test_mode: bool = False,
    force_rebuild: bool = False,
) -> tuple[dict, pd.Series | None, list[str] | None]:
    ensure_dirs()

    all_cached = all(
        os.path.exists(get_processed_path(y, test_mode=test_mode)) for y in YEARS
    )
    if all_cached and not force_rebuild:
        yearly_dfs = {
            y: pd.read_csv(get_processed_path(y, test_mode=test_mode)) for y in YEARS
        }
        for y, df in yearly_dfs.items():
            df["race_raw"] = df["RAC1P"].astype(int).map(RACE_CODE_MAP)
        combined_races = pd.concat(
            [df["race_raw"] for df in yearly_dfs.values()], ignore_index=True
        )
        race_shares = combined_races.value_counts() / len(combined_races)
        merged_categories = [
            RACE_CODE_MAP[c]
            for c in sorted(RACE_CODE_MAP.keys())
            if float(race_shares.get(RACE_CODE_MAP[c], 0.0)) < RACE_MIN_SHARE
            and RACE_CODE_MAP[c] not in KEEP_SEPARATE_RACES
        ]
        return yearly_dfs, race_shares, merged_categories

    yearly_dfs = {}
    for y in YEARS:
        full_cache_path = get_processed_path(y, test_mode=False)
        if test_mode and os.path.exists(full_cache_path) and not force_rebuild:
            df_year = pd.read_csv(full_cache_path)
            if "race_raw" not in df_year.columns:
                df_year["race_raw"] = df_year["RAC1P"].astype(int).map(RACE_CODE_MAP)
            yearly_dfs[y] = df_year
        else:
            yearly_dfs[y] = load_raw_year_from_folktables(y, download=download)

    yearly_dfs, race_shares, merged_categories = apply_race_merging_rule(yearly_dfs)

    if test_mode:
        sampled_dfs = {}
        for y in YEARS:
            sampled_dfs[y] = (
                yearly_dfs[y]
                .sample(frac=TEST_SAMPLE_FRAC, random_state=SEED)
                .reset_index(drop=True)
            )
        yearly_dfs = sampled_dfs

    cols_to_save = FEATURE_COLS + ["label", "sex_group", "race_group"]
    for y in YEARS:
        out_path = get_processed_path(y, test_mode=test_mode)
        yearly_dfs[y][cols_to_save].to_csv(out_path, index=False)

    return yearly_dfs, race_shares, merged_categories


def load_year_data(year: int, test_mode: bool = False) -> pd.DataFrame:
    path = get_processed_path(year, test_mode=test_mode)
    if not os.path.exists(path):
        build_and_cache_all_years(download=False, test_mode=test_mode)
    return pd.read_csv(path)


# --- Part 2c: 2014 data split (60% train, 20% calibration, 20% test) and preprocessing ---
def split_2014_data(df_2014: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df_2014,
        test_size=0.40,
        stratify=df_2014["label"],
        random_state=SEED,
    )
    cal_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED,
    )
    return (
        train_df.reset_index(drop=True),
        cal_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_preprocessor(dense_output: bool = False, drop_occp: bool = False) -> ColumnTransformer:
    cat_cols = get_categorical_cols(drop_occp=drop_occp)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_COLS),
            ("cat", ohe, cat_cols),
        ]
    )


def to_dense_if_sparse(x_mat) -> np.ndarray:
    if sparse.issparse(x_mat):
        return x_mat.toarray()
    return np.asarray(x_mat)


# --- Part 3a: Classifier definitions and 2014 baseline training ---
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
    """Train classifiers on 2014 train slice, calibrate on 2014 cal slice, and score 2014-2018."""
    from calibrators import IsotonicCalibrator, PlattCalibrator

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


# --- Part 3b: Group slicing and post-shift sample drawing ---
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
    from calibrators import (
        BaseRateCalibrator,
        BetaCalibrator,
        IsotonicCalibrator,
        PlattCalibrator,
        VennAbersCalibrator,
    )

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
