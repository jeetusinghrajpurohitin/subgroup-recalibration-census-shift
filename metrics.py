"""
Evaluation and calibration metrics, reliability bins, and bootstrap intervals.
"""

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

PROB_CLIP_MIN = 1e-6
PROB_CLIP_MAX = 1.0 - 1e-6
NUM_ECE_BINS = 10


# --- Part 1a: Probability clipping and log-odds helpers ---
def clip_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=float), PROB_CLIP_MIN, PROB_CLIP_MAX)


def logit(probs: np.ndarray) -> np.ndarray:
    p = clip_probs(probs)
    return np.log(p / (1.0 - p))


# --- Part 1b: 10-bin Expected Calibration Error (ECE) and reliability curve bins ---
def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = NUM_ECE_BINS) -> float:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    n_total = len(y)
    if n_total == 0:
        return np.nan

    bin_idx = np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins)
    non_empty = counts > 0

    sum_y = np.bincount(bin_idx, weights=y, minlength=n_bins)[non_empty]
    sum_p = np.bincount(bin_idx, weights=p, minlength=n_bins)[non_empty]
    cnt = counts[non_empty]

    mean_actual = sum_y / cnt
    mean_pred = sum_p / cnt
    return float(np.sum((cnt / n_total) * np.abs(mean_actual - mean_pred)))


def compute_reliability_bins(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = NUM_ECE_BINS
) -> dict:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    bin_idx = np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)

    counts = np.bincount(bin_idx, minlength=n_bins)
    sum_y = np.bincount(bin_idx, weights=y, minlength=n_bins)
    sum_p = np.bincount(bin_idx, weights=p, minlength=n_bins)

    mean_pred = np.full(n_bins, np.nan)
    mean_actual = np.full(n_bins, np.nan)
    non_empty = counts > 0
    mean_pred[non_empty] = sum_p[non_empty] / counts[non_empty]
    mean_actual[non_empty] = sum_y[non_empty] / counts[non_empty]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return {
        "bin_centers": bin_centers,
        "counts": counts,
        "mean_pred": mean_pred,
        "mean_actual": mean_actual,
    }


# --- Part 1c: Mean gap, Brier score, log-loss, accuracy, and AUC ---
def compute_mean_gap(labels: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    if len(y) == 0:
        return np.nan
    return float(np.mean(p) - np.mean(y))


def compute_brier(labels: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    if len(y) == 0:
        return np.nan
    return float(np.mean((p - y) ** 2))


def compute_all_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    p_clipped = clip_probs(p)

    ece = compute_ece(y, p)
    brier = compute_brier(y, p)
    ll = float(log_loss(y, p_clipped, labels=[0, 1]))
    acc = float(np.mean((p >= 0.5).astype(int) == y))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan

    mean_pred = float(np.mean(p))
    base_rate = float(np.mean(y))
    mean_gap = mean_pred - base_rate

    return {
        "ECE": ece,
        "Brier": brier,
        "LogLoss": ll,
        "Accuracy": acc,
        "AUC": auc,
        "MeanPred": mean_pred,
        "BaseRate": base_rate,
        "MeanGap": mean_gap,
    }


# --- Part 2a: Bootstrap 95% percentile intervals for ECE and Mean Gap ---
def bootstrap_ece_and_gap(
    labels: np.ndarray,
    probs: np.ndarray,
    n_boot: int = 200,
    seed: int = 42,
    n_bins: int = NUM_ECE_BINS,
) -> dict:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    n = len(y)
    if n == 0:
        return {
            "ece_low": np.nan,
            "ece_high": np.nan,
            "gap_low": np.nan,
            "gap_high": np.nan,
        }

    rng = np.random.default_rng(seed)
    bin_idx = np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)
    residual = y - p

    ece_samples = np.empty(n_boot, dtype=float)
    gap_samples = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bin_res_sum = np.bincount(bin_idx[idx], weights=residual[idx], minlength=n_bins)
        ece_samples[b] = np.sum(np.abs(bin_res_sum)) / n
        gap_samples[b] = -np.sum(bin_res_sum) / n

    ece_low, ece_high = np.percentile(ece_samples, [2.5, 97.5])
    gap_low, gap_high = np.percentile(gap_samples, [2.5, 97.5])
    return {
        "ece_low": float(ece_low),
        "ece_high": float(ece_high),
        "gap_low": float(gap_low),
        "gap_high": float(gap_high),
    }


def bootstrap_ece_and_gap_multi(
    labels: np.ndarray,
    probs_by_key: dict,
    n_boot: int = 200,
    seed: int = 42,
    n_bins: int = NUM_ECE_BINS,
) -> dict:
    y = np.asarray(labels, dtype=float)
    n = len(y)
    keys = list(probs_by_key.keys())
    if n == 0:
        return {
            k: {
                "ece_low": np.nan,
                "ece_high": np.nan,
                "gap_low": np.nan,
                "gap_high": np.nan,
            }
            for k in keys
        }

    rng = np.random.default_rng(seed)
    bins_by_key = {}
    res_by_key = {}
    for k in keys:
        p = np.asarray(probs_by_key[k], dtype=float)
        bins_by_key[k] = np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)
        res_by_key[k] = y - p

    ece_samples = {k: np.empty(n_boot, dtype=float) for k in keys}
    gap_samples = {k: np.empty(n_boot, dtype=float) for k in keys}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for k in keys:
            bin_res_sum = np.bincount(
                bins_by_key[k][idx], weights=res_by_key[k][idx], minlength=n_bins
            )
            ece_samples[k][b] = np.sum(np.abs(bin_res_sum)) / n
            gap_samples[k][b] = -np.sum(bin_res_sum) / n

    results = {}
    for k in keys:
        ece_low, ece_high = np.percentile(ece_samples[k], [2.5, 97.5])
        gap_low, gap_high = np.percentile(gap_samples[k], [2.5, 97.5])
        results[k] = {
            "ece_low": float(ece_low),
            "ece_high": float(ece_high),
            "gap_low": float(gap_low),
            "gap_high": float(gap_high),
        }
    return results


# --- Part 2b: Synthetic unit check for ECE calculation ---
def run_ece_unit_checks(seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_synth = 200_000
    p_true = rng.uniform(0.05, 0.90, size=n_synth)
    y_synth = rng.binomial(1, p_true)

    ece_perfect = compute_ece(y_synth, p_true)
    p_shifted = p_true + 0.05
    ece_shifted = compute_ece(y_synth, p_shifted)

    assert ece_perfect < 0.01, f"Expected ECE < 0.01, got {ece_perfect:.4f}"
    assert abs(ece_shifted - 0.05) < 0.005, f"Expected ECE ~ 0.05, got {ece_shifted:.4f}"
    return ece_perfect, ece_shifted
