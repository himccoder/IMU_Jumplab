"""
IMU Data Simulator
------------------
Generates realistic MPU6050 accelerometer data for testing the pipeline
without physical hardware.

Simulates three activity types at 100 Hz (matching the ESP32 firmware):
  - idle:    Sensor at rest, only gravity + small noise
  - walk:    Periodic vertical oscillation (~1.5 Hz), no freefall
  - run:     Higher-frequency oscillation (~2.5 Hz), larger amplitude
  - jump:    Pre-jump loading spike → freefall (~0g) → landing impact

Sensor orientation: z-axis is vertical (same as MPU6050 flat on waist).
Gravity appears as +9.81 m/s^2 on the z-axis at rest.

Output: pandas DataFrame with columns [timestamp_ms, ax, ay, az]
        and a 'label' column for ground truth (used later for ML training).
"""

import numpy as np
import pandas as pd
import os

SAMPLE_RATE_HZ = 100
G = 9.81  # m/s^2

RNG = np.random.default_rng(42)


def _noise(n: int, std: float = 0.05) -> np.ndarray:
    return RNG.normal(0, std, n)


def simulate_idle(duration_s: float = 2.0) -> pd.DataFrame:
    """Sensor at rest on a table or held still at the waist."""
    n = int(duration_s * SAMPLE_RATE_HZ)
    t = np.arange(n) * (1000 / SAMPLE_RATE_HZ)  # ms

    ax = _noise(n, 0.04)
    ay = _noise(n, 0.04)
    az = G + _noise(n, 0.05)

    return pd.DataFrame({"timestamp_ms": t, "ax": ax, "ay": ay, "az": az, "label": "idle"})


def simulate_walk(duration_s: float = 5.0, step_freq_hz: float = 1.5) -> pd.DataFrame:
    """
    Walking produces a sinusoidal vertical bounce at step frequency.
    Total magnitude never drops near 0g — this is the key distinguisher from jumps.
    """
    n = int(duration_s * SAMPLE_RATE_HZ)
    t = np.arange(n) * (1000 / SAMPLE_RATE_HZ)
    t_s = t / 1000.0

    vertical_oscillation = 1.8 * np.sin(2 * np.pi * step_freq_hz * t_s)
    forward_sway = 0.4 * np.sin(2 * np.pi * step_freq_hz * t_s + np.pi / 4)

    ax = forward_sway + _noise(n, 0.08)
    ay = 0.2 * np.sin(2 * np.pi * step_freq_hz * t_s * 2) + _noise(n, 0.06)
    az = G + vertical_oscillation + _noise(n, 0.08)

    return pd.DataFrame({"timestamp_ms": t, "ax": ax, "ay": ay, "az": az, "label": "walk"})


def simulate_run(duration_s: float = 5.0, step_freq_hz: float = 2.6) -> pd.DataFrame:
    """
    Running has higher frequency and amplitude than walking.
    Brief air phases but not long enough to qualify as a jump.
    """
    n = int(duration_s * SAMPLE_RATE_HZ)
    t = np.arange(n) * (1000 / SAMPLE_RATE_HZ)
    t_s = t / 1000.0

    vertical_oscillation = 3.5 * np.sin(2 * np.pi * step_freq_hz * t_s)
    forward_sway = 0.9 * np.sin(2 * np.pi * step_freq_hz * t_s + np.pi / 6)

    ax = forward_sway + _noise(n, 0.15)
    ay = 0.3 * np.sin(2 * np.pi * step_freq_hz * t_s * 2) + _noise(n, 0.10)
    az = G + vertical_oscillation + _noise(n, 0.12)

    return pd.DataFrame({"timestamp_ms": t, "ax": ax, "ay": ay, "az": az, "label": "run"})


def simulate_single_jump(height_m: float = 0.35, pre_idle_s: float = 0.5, post_idle_s: float = 0.5) -> pd.DataFrame:
    """
    Simulate one vertical jump of the given height.

    Physics:
      - Jump height h = g * t_air^2 / 8  (symmetrical flight parabola)
      - Rearranging: t_air = sqrt(8 * h / g)

    Phases:
      1. Pre-jump idle (~0.3 s)
      2. Crouch/loading: rapid downward acceleration (~0.15 s)
      3. Takeoff spike: large upward acceleration (~0.1 s)
      4. Freefall: all axes near 0g (t_air seconds)
      5. Landing impact: massive spike (~0.05 s)
      6. Post-landing damping (~0.2 s)
      7. Return to idle
    """
    sr = SAMPLE_RATE_HZ
    segments = []
    t_offset = 0.0

    def _make_segment(duration_s, ax_fn, ay_fn, az_fn, lbl):
        nonlocal t_offset
        n = max(1, int(duration_s * sr))
        t_local = np.linspace(0, duration_s, n, endpoint=False)
        t_ms = (t_offset + t_local) * 1000.0
        t_offset += duration_s
        return pd.DataFrame({
            "timestamp_ms": t_ms,
            "ax": ax_fn(t_local, n),
            "ay": ay_fn(t_local, n),
            "az": az_fn(t_local, n),
            "label": lbl,
        })

    # Pre-jump idle
    def idle_az(t, n): return G + _noise(n, 0.05)
    def zero_ax(t, n): return _noise(n, 0.04)
    segments.append(_make_segment(pre_idle_s, zero_ax, zero_ax, idle_az, "idle"))

    # Crouch (loading): body sinks, sensor sees reduced vertical force briefly
    def crouch_az(t, n):
        profile = G - 3.5 * np.sin(np.pi * t / 0.15)
        return profile + _noise(n, 0.1)
    segments.append(_make_segment(0.15, zero_ax, zero_ax, crouch_az, "jump"))

    # Takeoff: leg extension — large upward force
    def takeoff_az(t, n):
        profile = G + 18.0 * np.sin(np.pi * t / 0.10)
        return profile + _noise(n, 0.2)
    def takeoff_ax(t, n): return 1.0 * np.sin(np.pi * t / 0.10) + _noise(n, 0.1)
    segments.append(_make_segment(0.10, takeoff_ax, zero_ax, takeoff_az, "jump"))

    # Freefall: near-zero acceleration (just tiny residual noise)
    t_air = np.sqrt(8 * height_m / G)

    def freefall_az(t, n): return _noise(n, 0.08)  # ~0g
    def freefall_ax(t, n): return _noise(n, 0.06)
    segments.append(_make_segment(t_air, freefall_ax, freefall_ax, freefall_az, "freefall"))

    # Landing impact
    def landing_az(t, n):
        profile = G + 35.0 * np.exp(-t / 0.015) * np.sin(np.pi * t / 0.05)
        return profile + _noise(n, 0.3)
    def landing_ax(t, n): return 2.0 * np.exp(-t / 0.02) + _noise(n, 0.2)
    segments.append(_make_segment(0.08, landing_ax, zero_ax, landing_az, "jump"))

    # Post-landing damping (bouncing oscillation dying out)
    def damp_az(t, n):
        profile = G + 4.0 * np.exp(-t / 0.08) * np.sin(2 * np.pi * 5 * t)
        return profile + _noise(n, 0.1)
    segments.append(_make_segment(0.20, zero_ax, zero_ax, damp_az, "jump"))

    # Post-jump idle
    segments.append(_make_segment(post_idle_s, zero_ax, zero_ax, idle_az, "idle"))

    df = pd.concat(segments, ignore_index=True)
    return df


def simulate_jump_session(n_jumps: int = 5, heights_m: list = None) -> pd.DataFrame:
    """
    Simulate a full session: idle → walk → [jump × n] → walk → idle
    Heights default to a random spread between 0.20 and 0.55 m.
    """
    if heights_m is None:
        heights_m = RNG.uniform(0.20, 0.55, n_jumps).tolist()

    segments = [
        simulate_idle(2.0),
        simulate_walk(4.0),
        simulate_idle(1.0),
    ]

    for h in heights_m:
        segments.append(simulate_single_jump(height_m=h, pre_idle_s=0.8, post_idle_s=0.8))

    segments.append(simulate_idle(1.0))
    segments.append(simulate_walk(3.0))
    segments.append(simulate_idle(2.0))

    # Re-stamp timestamps as continuous
    dfs = []
    t_offset = 0.0
    for df in segments:
        df = df.copy()
        df["timestamp_ms"] = df["timestamp_ms"] - df["timestamp_ms"].iloc[0] + t_offset
        t_offset = df["timestamp_ms"].iloc[-1] + (1000.0 / SAMPLE_RATE_HZ)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def simulate_mixed_activities(duration_s: float = 30.0) -> pd.DataFrame:
    """
    Longer mixed session interleaving all activity types.
    Good for testing the classifier's ability to distinguish activities.
    """
    segments = [
        simulate_idle(3.0),
        simulate_walk(6.0),
        simulate_run(5.0),
        simulate_idle(1.0),
        simulate_single_jump(0.40),
        simulate_idle(0.5),
        simulate_single_jump(0.30),
        simulate_idle(0.5),
        simulate_single_jump(0.50),
        simulate_walk(4.0),
        simulate_run(3.0),
        simulate_idle(2.0),
    ]

    dfs = []
    t_offset = 0.0
    for df in segments:
        df = df.copy()
        df["timestamp_ms"] = df["timestamp_ms"] - df["timestamp_ms"].iloc[0] + t_offset
        t_offset = df["timestamp_ms"].iloc[-1] + (1000.0 / SAMPLE_RATE_HZ)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def save_simulation(df: pd.DataFrame, filename: str, output_dir: str = "data/simulated") -> str:
    """Save a simulated dataset to CSV. Returns the saved file path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df[["timestamp_ms", "ax", "ay", "az"]].to_csv(path, index=False)
    print(f"[simulator] Saved {len(df)} samples ({df['timestamp_ms'].iloc[-1]/1000:.1f}s) -> {path}")
    return path


if __name__ == "__main__":
    print("Generating simulated datasets...")
    df_jump = simulate_jump_session(n_jumps=5)
    save_simulation(df_jump, "jump_session.csv")

    df_mixed = simulate_mixed_activities()
    save_simulation(df_mixed, "mixed_activities.csv")
    print("Done.")
