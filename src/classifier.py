"""
ML Activity Classifier — Milestone 4 Scaffold
----------------------------------------------
Classifies activity windows (idle / walk / run / jump) using a
sliding-window feature extraction approach and a Random Forest classifier.

This module is intentionally structured to be runnable NOW on simulated
(labelled) data. Once real labelled hardware data is collected, swap in
the real CSV files and retrain.

Pipeline:
  1. extract_features()  — Slide a 200ms window over the magnitude signal,
                           compute statistical + domain-specific features per window
  2. build_dataset()     — Run feature extraction over a labelled DataFrame
  3. train()             — Train a RandomForestClassifier, print cross-val report
  4. evaluate()          — Confusion matrix + classification report
  5. save_model() / load_model() — Persist to disk with joblib

Features per window (12 total):
  mean, std, min, max, range, rms, peak-to-peak,
  zero_crossing_rate, energy, skewness, kurtosis, freefall_ratio
"""

import os
import warnings
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

G = 9.81
SAMPLE_RATE_HZ = 100
WINDOW_SAMPLES = 20       # 200 ms window @ 100 Hz
STEP_SAMPLES = 10         # 50% overlap = 100 ms step

FEATURE_NAMES = [
    "mean", "std", "min_val", "max_val", "range_val",
    "rms", "peak_to_peak", "zero_crossing_rate",
    "energy", "skewness", "kurtosis", "freefall_ratio",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _window_features(window: np.ndarray) -> np.ndarray:
    """Extract all features from a single 1-D magnitude window."""
    n = len(window)
    mean = window.mean()
    std = window.std()
    min_v = window.min()
    max_v = window.max()
    range_v = max_v - min_v
    rms = np.sqrt((window ** 2).mean())
    p2p = max_v - min_v
    # Zero-crossing rate on demeaned signal
    demeaned = window - mean
    zcr = ((demeaned[:-1] * demeaned[1:]) < 0).sum() / (n - 1) if n > 1 else 0.0
    energy = (window ** 2).sum() / n
    sk = skew(window)
    kurt = kurtosis(window)
    freefall_ratio = (window < 0.35 * G).mean()

    return np.array([mean, std, min_v, max_v, range_v, rms, p2p, zcr,
                     energy, sk, kurt, freefall_ratio], dtype=np.float32)


def extract_features(df: pd.DataFrame,
                     window: int = WINDOW_SAMPLES,
                     step: int = STEP_SAMPLES) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sliding-window feature extraction over the magnitude signal.

    Returns:
      X : (n_windows, n_features) feature matrix
      y : (n_windows,) majority-vote labels per window, or None if no 'label' column
    """
    if "magnitude" not in df.columns:
        raise ValueError("'magnitude' column required. Run signal_processor.get_processed() first.")

    mag = df["magnitude"].values
    has_labels = "label" in df.columns
    labels = df["label"].values if has_labels else None

    n = len(mag)
    starts = range(0, n - window + 1, step)

    X_list, y_list = [], []
    for s in starts:
        seg = mag[s: s + window]
        X_list.append(_window_features(seg))
        if has_labels:
            # Majority vote for window label
            lbl_seg = labels[s: s + window]
            unique, counts = np.unique(lbl_seg, return_counts=True)
            y_list.append(unique[counts.argmax()])

    X = np.vstack(X_list) if X_list else np.empty((0, len(FEATURE_NAMES)))
    y = np.array(y_list) if y_list else None

    return X, y


def build_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Build the (X, y_encoded) training dataset from a labelled DataFrame.
    Returns X, y (integer-encoded), and the fitted LabelEncoder.
    """
    X, y_str = extract_features(df)
    if y_str is None:
        raise ValueError("DataFrame must have a 'label' column for supervised training.")

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    return X, y, le


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train(X: np.ndarray, y: np.ndarray, le: LabelEncoder,
          n_estimators: int = 150, random_state: int = 42) -> RandomForestClassifier:
    """
    Train a RandomForest classifier with 5-fold stratified cross-validation.
    Prints per-fold accuracy and mean ± std.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")

    print(f"\n[classifier] Cross-validation F1-macro: "
          f"{scores.mean():.3f} ± {scores.std():.3f}")
    print(f"             Per-fold: {[f'{s:.3f}' for s in scores]}")

    clf.fit(X, y)
    return clf


def evaluate(clf: RandomForestClassifier, X: np.ndarray, y: np.ndarray,
             le: LabelEncoder, save_path: Optional[str] = None):
    """
    Print classification report and plot a confusion matrix.
    Note: for a proper evaluation use a held-out test split, not the full training data.
    """
    y_pred = clf.predict(X)
    class_names = le.classes_

    print("\n[classifier] Classification Report (training data — use test split for real eval):")
    print(classification_report(y, y_pred, target_names=class_names))

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[classifier] Confusion matrix saved -> {save_path}")

    return fig


def feature_importance_plot(clf: RandomForestClassifier, save_path: Optional[str] = None):
    """Bar chart of feature importances from the trained RandomForest."""
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(importances)), importances[idx], color=sns.color_palette("muted", len(importances)))
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=30, ha="right")
    ax.set_title("Random Forest — Feature Importances")
    ax.set_ylabel("Importance")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[classifier] Feature importance saved -> {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(clf: RandomForestClassifier, le: LabelEncoder, path: str = "data/model.joblib"):
    """Save the trained classifier + label encoder to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump({"clf": clf, "le": le}, path)
    print(f"[classifier] Model saved -> {path}")


def load_model(path: str = "data/model.joblib") -> Tuple[RandomForestClassifier, LabelEncoder]:
    """Load a previously saved model."""
    obj = joblib.load(path)
    return obj["clf"], obj["le"]


if __name__ == "__main__":
    import os
    from src.simulator import simulate_mixed_activities, save_simulation
    from src.signal_processor import get_processed

    csv_path = os.path.join("data", "simulated", "mixed_activities.csv")
    sim_raw_path = os.path.join("data", "simulated", "mixed_activities_labelled.csv")

    # Generate with labels
    df_raw = simulate_mixed_activities()
    df_raw.to_csv(sim_raw_path, index=False)

    df = get_processed(csv_path if os.path.exists(csv_path) else sim_raw_path)
    # Labels come from the simulator's raw CSV
    df_labels = pd.read_csv(sim_raw_path)
    df["label"] = df_labels["label"].values

    X, y, le = build_dataset(df)
    print(f"[classifier] Dataset: {X.shape[0]} windows, classes: {list(le.classes_)}")

    clf = train(X, y, le)
    os.makedirs("data/reports", exist_ok=True)
    evaluate(clf, X, y, le, save_path="data/reports/confusion_matrix.png")
    feature_importance_plot(clf, save_path="data/reports/feature_importances.png")
    save_model(clf, le)
    plt.close("all")
