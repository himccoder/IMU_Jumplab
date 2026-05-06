"""
ML Activity Classifier
-----------------------
Classifies activity windows (still / walk / run / jump) from tri-axial
accelerometer data using a sliding-window feature extraction approach and
a HistGradientBoostingClassifier (sklearn's LightGBM-style GBDT).

Why these choices?
  - Window 500 ms / step 50 ms: long enough to capture a full freefall phase
    during a jump (~200–600 ms), fine-grained enough for smooth real-time output.
  - Tri-axial features (ax, ay, az + magnitude): directional information greatly
    helps separate walk (rhythmic vertical) from run (higher lateral sway) and
    still (near-constant on all axes).
  - 44 features: 10 statistical descriptors × 4 signals + 4 cross-signal features.
  - HistGradientBoostingClassifier: faster than RandomForest at the same or better
    accuracy, native balanced-class support, handles small datasets well.

Public API (called from main.py and realtime_classifier.py):
  extract_features(df)            → (X, y_str | None)
  build_dataset(df)               → (X, y_encoded, LabelEncoder)
  train(X, y, le)                 → fitted classifier
  evaluate(clf, X, y, le, ...)
  feature_importance_plot(clf, …)
  save_model(clf, le, path)
  load_model(path)                → (clf, le)
  extract_window_features_from_axes(ax, ay, az) → 1-D feature vector
    ↑ used by realtime_classifier.py so training and inference are identical
"""

import os
import warnings
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

G              = 9.81
SAMPLE_RATE_HZ = 100
WINDOW_SAMPLES = 50    # 500 ms @ 100 Hz — long enough to catch a full freefall
STEP_SAMPLES   = 5     # 50 ms step → smooth real-time output

# Lowpass filter shared between training pipeline and real-time inference
_LPF_CUTOFF_HZ = 10.0
_LPF_ORDER     = 4

# ── Feature names ────────────────────────────────────────────────────────────
_SIGNALS   = ["ax", "ay", "az", "mag"]
_STAT_KEYS = ["mean", "std", "min", "max", "range", "rms",
               "energy", "skew", "kurt", "zcr"]

FEATURE_NAMES: List[str] = (
    [f"{sig}_{stat}" for sig in _SIGNALS for stat in _STAT_KEYS]   # 40
    + ["freefall_ratio", "impulse", "corr_xy", "corr_xz"]           #  4
)   # total = 44


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _make_lpf():
    """Return (b, a) Butterworth lowpass coefficients (cached)."""
    nyq = 0.5 * SAMPLE_RATE_HZ
    return butter(_LPF_ORDER, _LPF_CUTOFF_HZ / nyq, btype="low")


def _apply_lpf(signal: np.ndarray) -> np.ndarray:
    b, a = _make_lpf()
    if len(signal) < 3 * max(len(a), len(b)):
        return signal   # too short to filter safely
    return filtfilt(b, a, signal)


def _signal_stats(x: np.ndarray) -> np.ndarray:
    """10 statistical features for a 1-D signal window."""
    n    = len(x)
    mean = x.mean()
    std  = x.std()
    xmin = x.min()
    xmax = x.max()
    rng  = xmax - xmin
    rms  = float(np.sqrt((x ** 2).mean()))
    eng  = float((x ** 2).sum() / n)
    sk   = float(skew(x))
    kt   = float(kurtosis(x))
    dm   = x - mean
    zcr  = float(((dm[:-1] * dm[1:]) < 0).sum() / (n - 1)) if n > 1 else 0.0
    return np.array([mean, std, xmin, xmax, rng, rms, eng, sk, kt, zcr],
                    dtype=np.float32)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returns 0 if std of either signal is ~0."""
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Core feature extraction — shared between training and real-time inference
# ---------------------------------------------------------------------------

def extract_window_features_from_axes(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
) -> np.ndarray:
    """
    Extract 44 features from a single raw window of (ax, ay, az).

    Applies the Butterworth lowpass filter internally so this function can be
    called identically from the training pipeline and from the real-time
    classifier — guaranteeing that training and inference see the same signals.

    Parameters
    ----------
    ax, ay, az : 1-D arrays of length WINDOW_SAMPLES (50 @ 100 Hz)

    Returns
    -------
    features : np.ndarray of shape (44,), dtype float32
    """
    ax_f = _apply_lpf(ax.astype(np.float64)).astype(np.float32)
    ay_f = _apply_lpf(ay.astype(np.float64)).astype(np.float32)
    az_f = _apply_lpf(az.astype(np.float64)).astype(np.float32)
    mag  = np.sqrt(ax_f**2 + ay_f**2 + az_f**2)

    feats = np.concatenate([
        _signal_stats(ax_f),
        _signal_stats(ay_f),
        _signal_stats(az_f),
        _signal_stats(mag),
        [
            float((mag < 0.40 * G).mean()),         # freefall_ratio
            float(mag.max() - mag.min()),             # impulse (peak-to-peak mag)
            _safe_corr(ax_f, ay_f),                  # corr_xy
            _safe_corr(ax_f, az_f),                  # corr_xz
        ],
    ]).astype(np.float32)

    return feats


# ---------------------------------------------------------------------------
# Sliding-window feature extraction over a DataFrame
# ---------------------------------------------------------------------------

def extract_features(
    df: pd.DataFrame,
    window: int = WINDOW_SAMPLES,
    step:   int = STEP_SAMPLES,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Slide a window over the DataFrame and extract features per window.

    Requires columns: ax, ay, az  (raw, before filtering — filtering is
    done inside extract_window_features_from_axes so training == inference).
    Optional column: label

    Returns
    -------
    X : (n_windows, 44) float32 feature matrix
    y : (n_windows,) string labels (majority vote), or None if no label column
    """
    for col in ("ax", "ay", "az"):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. DataFrame must have raw ax/ay/az columns."
            )

    ax_arr = df["ax"].values.astype(np.float32)
    ay_arr = df["ay"].values.astype(np.float32)
    az_arr = df["az"].values.astype(np.float32)

    has_labels = "label" in df.columns
    labels     = df["label"].values if has_labels else None

    n      = len(ax_arr)
    starts = range(0, n - window + 1, step)

    X_list, y_list = [], []
    for s in starts:
        feats = extract_window_features_from_axes(
            ax_arr[s: s + window],
            ay_arr[s: s + window],
            az_arr[s: s + window],
        )
        X_list.append(feats)

        if has_labels:
            seg = labels[s: s + window]
            vals, cnts = np.unique(seg, return_counts=True)
            y_list.append(vals[cnts.argmax()])

    X = np.vstack(X_list) if X_list else np.empty((0, len(FEATURE_NAMES)), np.float32)
    y = np.array(y_list)  if y_list  else None
    return X, y


def build_dataset(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Build (X, y_encoded, LabelEncoder) from a labelled DataFrame."""
    X, y_str = extract_features(df)
    if y_str is None:
        raise ValueError("DataFrame must have a 'label' column for supervised training.")
    le = LabelEncoder()
    y  = le.fit_transform(y_str)
    return X, y, le


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train(
    X:  np.ndarray,
    y:  np.ndarray,
    le: LabelEncoder,
    random_state: int = 42,
) -> HistGradientBoostingClassifier:
    """
    Train a HistGradientBoostingClassifier with 5-fold stratified CV.

    HistGradientBoosting is sklearn's LightGBM-style GBDT:
    fast training, excellent accuracy on tabular data, native class-balance support.
    """
    clf = HistGradientBoostingClassifier(
        max_iter          = 300,
        learning_rate     = 0.05,
        max_depth         = 6,
        min_samples_leaf  = 5,
        class_weight      = "balanced",
        random_state      = random_state,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")

    print(f"\n[classifier] HistGradientBoosting — CV F1-macro: "
          f"{scores.mean():.3f} ± {scores.std():.3f}")
    print(f"             Per-fold: {[f'{s:.3f}' for s in scores]}")
    print(f"             Classes : {list(le.classes_)}")

    clf.fit(X, y)
    return clf


def evaluate(
    clf,
    X:  np.ndarray,
    y:  np.ndarray,
    le: LabelEncoder,
    save_path: Optional[str] = None,
):
    """Confusion matrix + classification report."""
    y_pred      = clf.predict(X)
    class_names = le.classes_

    print("\n[classifier] Classification Report:")
    print(classification_report(y, y_pred, target_names=class_names))

    cm  = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix — HistGradientBoosting")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[classifier] Confusion matrix saved → {save_path}")

    return fig


def feature_importance_plot(
    clf,
    X:         Optional[np.ndarray] = None,
    y:         Optional[np.ndarray] = None,
    save_path: Optional[str]        = None,
    n_repeats: int = 10,
    random_state: int = 42,
):
    """
    Bar chart of feature importances.

    Uses permutation importance (model-agnostic, works with any sklearn
    estimator including HistGradientBoostingClassifier).  Falls back to
    `feature_importances_` for tree ensembles that expose it directly.

    Parameters
    ----------
    clf        : fitted classifier
    X, y       : validation data for permutation importance (required for
                 HistGradientBoostingClassifier; can be the training set)
    save_path  : optional path to save the PNG
    """
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        method_label = "Mean Decrease in Impurity"
    elif X is not None and y is not None:
        from sklearn.inspection import permutation_importance
        print("[classifier] Computing permutation importances (this takes a moment)…")
        result = permutation_importance(
            clf, X, y,
            n_repeats=n_repeats,
            scoring="f1_macro",
            random_state=random_state,
            n_jobs=-1,
        )
        importances  = result.importances_mean
        method_label = "Permutation Importance (F1-macro)"
    else:
        print("[classifier] Skipping feature importance plot "
              "(pass X and y to enable permutation importance).")
        return None

    idx = np.argsort(importances)[::-1][:20]   # top-20

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(idx)), importances[idx],
           color=sns.color_palette("muted", len(idx)))
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=35, ha="right")
    ax.set_title(f"HistGradientBoosting — Top-20 Features ({method_label})")
    ax.set_ylabel("Importance")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[classifier] Feature importances saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(clf, le: LabelEncoder, path: str = "data/model.joblib"):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump({"clf": clf, "le": le}, path)
    print(f"[classifier] Model saved → {path}")


def load_model(path: str = "data/model.joblib") -> Tuple[object, LabelEncoder]:
    obj = joblib.load(path)
    return obj["clf"], obj["le"]
