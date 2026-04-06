"""
Signal Processing Module
------------------------
Handles all data loading, filtering, and feature preparation steps for the
raw MPU6050 accelerometer CSV data (whether from simulator or real hardware).

Pipeline steps:
  1. load_csv()        — Read CSV into a tidy DataFrame
  2. apply_lowpass()   — Butterworth low-pass filter to remove high-freq noise
  3. compute_magnitude() — |a| = sqrt(ax^2 + ay^2 + az^2)
  4. calibrate_gravity() — Estimate and subtract static gravity offset so that
                           the "net dynamic acceleration" is centred near 0 at rest.
  5. get_processed()   — Convenience wrapper running all steps in sequence.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

SAMPLE_RATE_HZ = 100
G = 9.81  # m/s^2


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a raw IMU CSV file.

    Expected columns: timestamp_ms, ax, ay, az
    Optional column:  label (present in simulated data, absent in real recordings)
    """
    df = pd.read_csv(path)

    required = {"timestamp_ms", "ax", "ay", "az"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    # Estimate actual sample rate from timestamps for sanity check
    if len(df) > 10:
        dt_ms = np.median(np.diff(df["timestamp_ms"].values))
        actual_hz = 1000.0 / dt_ms if dt_ms > 0 else 0
        if abs(actual_hz - SAMPLE_RATE_HZ) > 20:
            print(f"[signal_processor] WARNING: Detected sample rate {actual_hz:.1f} Hz "
                  f"(expected ~{SAMPLE_RATE_HZ} Hz). Filtering will use {SAMPLE_RATE_HZ} Hz.")

    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_lowpass(df: pd.DataFrame, cutoff_hz: float = 10.0, order: int = 4,
                  fs: float = SAMPLE_RATE_HZ) -> pd.DataFrame:
    """
    Apply a zero-phase Butterworth low-pass filter to ax, ay, az columns.

    10 Hz cutoff removes high-frequency vibration/noise while preserving
    the jump dynamics (takeoff ~1-5 Hz, landing ~5-15 Hz onset).

    Returns a new DataFrame with columns ax_filt, ay_filt, az_filt added.
    """
    df = df.copy()
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq

    if normal_cutoff >= 1.0:
        raise ValueError(f"Cutoff {cutoff_hz} Hz is at or above Nyquist ({nyq} Hz). Lower the cutoff.")

    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    for axis in ["ax", "ay", "az"]:
        df[f"{axis}_filt"] = filtfilt(b, a, df[axis].values)

    return df


# ---------------------------------------------------------------------------
# Magnitude
# ---------------------------------------------------------------------------

def compute_magnitude(df: pd.DataFrame, use_filtered: bool = True) -> pd.DataFrame:
    """
    Compute total acceleration magnitude |a| = sqrt(ax^2 + ay^2 + az^2).

    If use_filtered=True (recommended), uses the _filt columns.
    Adds column 'magnitude' (m/s^2) to the DataFrame.
    """
    df = df.copy()
    suffix = "_filt" if use_filtered else ""

    cols_needed = [f"ax{suffix}", f"ay{suffix}", f"az{suffix}"]
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Run apply_lowpass() first if use_filtered=True.")

    df["magnitude"] = np.sqrt(
        df[f"ax{suffix}"] ** 2 +
        df[f"ay{suffix}"] ** 2 +
        df[f"az{suffix}"] ** 2
    )
    return df


# ---------------------------------------------------------------------------
# Gravity calibration
# ---------------------------------------------------------------------------

def calibrate_gravity(df: pd.DataFrame, calibration_window_s: float = 1.0,
                      fs: float = SAMPLE_RATE_HZ) -> pd.DataFrame:
    """
    Estimate the gravity baseline from a quiet window at the start of the recording
    and subtract it from the magnitude, yielding 'net_accel' centred near 0 at rest.

    net_accel ≈ 0 at rest, negative during freefall, positive during impact.

    Adds column 'net_accel' to the DataFrame.
    """
    df = df.copy()
    n_cal = int(calibration_window_s * fs)
    n_cal = min(n_cal, len(df))

    gravity_estimate = df["magnitude"].iloc[:n_cal].mean()
    df["gravity_estimate"] = gravity_estimate
    df["net_accel"] = df["magnitude"] - gravity_estimate

    return df


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------

def get_processed(path_or_df, cutoff_hz: float = 10.0) -> pd.DataFrame:
    """
    Run the complete processing pipeline on a CSV file path or raw DataFrame.

    Returns a DataFrame with all original columns plus:
      ax_filt, ay_filt, az_filt — filtered axes
      magnitude                  — total acceleration magnitude
      net_accel                  — magnitude minus gravity baseline
    """
    if isinstance(path_or_df, str):
        df = load_csv(path_or_df)
    else:
        df = path_or_df.copy()

    df = apply_lowpass(df, cutoff_hz=cutoff_hz)
    df = compute_magnitude(df, use_filtered=True)
    df = calibrate_gravity(df)

    return df


if __name__ == "__main__":
    # Quick smoke test using simulated data
    import os
    from src.simulator import simulate_jump_session, save_simulation

    csv_path = os.path.join("data", "simulated", "jump_session.csv")
    if not os.path.exists(csv_path):
        df_raw = simulate_jump_session(n_jumps=3)
        save_simulation(df_raw, "jump_session.csv")

    df = get_processed(csv_path)
    print(df[["timestamp_ms", "magnitude", "net_accel"]].describe())
    print(f"\nMin magnitude: {df['magnitude'].min():.3f} m/s^2")
    print(f"Max magnitude: {df['magnitude'].max():.3f} m/s^2")
    print(f"Gravity estimate: {df['gravity_estimate'].iloc[0]:.3f} m/s^2")
