"""
Post-hoc probability calibrator classes (Platt, Isotonic, BaseRate, Beta, Venn-Abers).
"""

import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from common import SEED
from metrics import clip_probs, logit


# --- Part 1a: Platt scaling (2 parameters: slope and intercept on log-odds) ---
class PlattCalibrator:
    def __init__(self):
        self.lr = LogisticRegression(C=1e5, solver="lbfgs", random_state=SEED)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        x_logit = logit(probs).reshape(-1, 1)
        y = np.asarray(labels, dtype=int)
        self.lr.fit(x_logit, y)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        x_logit = logit(probs).reshape(-1, 1)
        return self.lr.predict_proba(x_logit)[:, 1]


# --- Part 1b: Isotonic regression (non-decreasing step function) ---
class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        p = np.asarray(probs, dtype=float)
        y = np.asarray(labels, dtype=float)
        self.iso.fit(p, y)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        return np.asarray(self.iso.predict(p), dtype=float)


# --- Part 1c: BaseRate correction (1 parameter: intercept shift on log-odds) ---
class BaseRateCalibrator:
    """Shift log-odds by a constant delta so mean prediction matches sample positive rate."""

    def __init__(self):
        self.delta: float = 0.0

    def fit(self, platt_2014_probs: np.ndarray, labels: np.ndarray) -> "BaseRateCalibrator":
        z = logit(platt_2014_probs)
        target_rate = float(np.mean(labels))
        target_rate = float(np.clip(target_rate, 1e-6, 1.0 - 1e-6))

        def objective(d: float) -> float:
            return float(np.mean(expit(z + d)) - target_rate)

        self.delta = float(brentq(objective, -15.0, 15.0))
        return self

    def predict(self, platt_2014_probs: np.ndarray) -> np.ndarray:
        z = logit(platt_2014_probs)
        return np.asarray(expit(z + self.delta), dtype=float)


# --- Part 1d: Beta calibration (3 parameters: logistic regression on ln(p) and -ln(1-p)) ---
def beta_features(probs: np.ndarray) -> np.ndarray:
    p = clip_probs(probs)
    return np.column_stack([np.log(p), -np.log(1.0 - p)])


class BetaCalibrator:
    def __init__(self):
        self.lr = LogisticRegression(C=1e5, solver="lbfgs", random_state=SEED)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "BetaCalibrator":
        self.lr.fit(beta_features(probs), np.asarray(labels, dtype=int))
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(beta_features(probs))[:, 1]


# --- Part 1e: Venn-Abers calibration (two isotonic fits per test score) ---
class VennAbersCalibrator:
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "VennAbersCalibrator":
        from venn_abers import VennAbers

        p = np.asarray(probs, dtype=float)
        self.va = VennAbers()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            self.va.fit(np.column_stack([1.0 - p, p]), np.asarray(labels, dtype=int))
        return self

    def predict_p0_p1(self, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(probs, dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            _, p0_p1 = self.va.predict_proba(np.column_stack([1.0 - p, p]))
        return p0_p1[:, 0], p0_p1[:, 1]

    def predict(self, probs: np.ndarray) -> np.ndarray:
        p0, p1 = self.predict_p0_p1(probs)
        return p1 / (1.0 - p0 + p1)


def venn_abers_slow(cal_probs, cal_labels, test_probs) -> tuple[np.ndarray, np.ndarray]:
    """Reference two-fit isotonic implementation used to verify VennAbersCalibrator."""
    cal_p = np.asarray(cal_probs, dtype=float)
    cal_y = np.asarray(cal_labels, dtype=float)
    p0 = np.empty(len(test_probs))
    p1 = np.empty(len(test_probs))
    for i, s in enumerate(np.asarray(test_probs, dtype=float)):
        for label, out in ((0.0, p0), (1.0, p1)):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(np.append(cal_p, s), np.append(cal_y, label))
            out[i] = iso.predict([s])[0]
    return p0, p1
