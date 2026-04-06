"""
Jump Detection Module
---------------------
Detects vertical jumps from processed IMU data using a freefall threshold approach.

Algorithm (Milestone 3):
  1. Identify "freefall windows" where magnitude < FREEFALL_THRESHOLD for at least
     MIN_FREEFALL_MS milliseconds. During true freefall, the body is weightless
     and the sensor reads near 0g.
  2. Verify each candidate window is preceded by a takeoff spike (magnitude > TAKEOFF_THRESHOLD)
     and followed by a landing spike — this eliminates false positives.
  3. Calculate jump height from time-in-air using the kinematic equation:
        h = g * t_air^2 / 8
     (derived from: t_air = 2 * sqrt(2h/g), so h = g * (t_air/2)^2 / 2)

Thresholds are tunable via the JumpDetector class and should be calibrated
against real data during Milestone 2/3.

Output:
  A list of JumpEvent dataclasses, each containing:
    - start_ms, end_ms      : freefall window timestamps
    - flight_time_s         : duration of freefall
    - height_m, height_cm   : calculated jump height
    - takeoff_peak          : pre-jump acceleration peak (m/s^2)
    - landing_peak          : post-jump acceleration peak (m/s^2)
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

G = 9.81  # m/s^2


@dataclass
class JumpEvent:
    start_ms: float
    end_ms: float
    flight_time_s: float
    height_m: float
    height_cm: float
    takeoff_peak: float
    landing_peak: float

    def __str__(self):
        return (
            f"Jump @ {self.start_ms/1000:.2f}s–{self.end_ms/1000:.2f}s | "
            f"flight={self.flight_time_s*1000:.0f}ms | "
            f"height={self.height_cm:.1f} cm | "
            f"takeoff={self.takeoff_peak:.1f} m/s² | "
            f"landing={self.landing_peak:.1f} m/s²"
        )


class JumpDetector:
    """
    Threshold-based jump detector.

    Parameters (tune these once you have real hardware data):
      freefall_threshold_g : Acceleration below this (in units of g) is considered freefall.
                             Start with 0.35g; lower if missing jumps, raise to reduce false positives.
      min_freefall_ms      : Minimum freefall duration to be counted as a jump.
                             ~100 ms corresponds to a ~1.2 cm jump (very low).
      max_freefall_ms      : Sanity cap — no human jumps more than ~2 seconds of air time.
      context_window_ms    : How far before/after a freefall window to look for peaks.
      takeoff_threshold_g  : Minimum pre-jump spike to confirm takeoff (not just stillness).
    """

    def __init__(
        self,
        freefall_threshold_g: float = 0.35,
        min_freefall_ms: float = 100.0,
        max_freefall_ms: float = 1800.0,
        context_window_ms: float = 300.0,
        takeoff_threshold_g: float = 1.3,
    ):
        self.freefall_threshold = freefall_threshold_g * G
        self.min_freefall_ms = min_freefall_ms
        self.max_freefall_ms = max_freefall_ms
        self.context_window_ms = context_window_ms
        self.takeoff_threshold = takeoff_threshold_g * G

    def detect(self, df: pd.DataFrame) -> List[JumpEvent]:
        """
        Detect all jump events in a processed DataFrame.

        Requires columns: timestamp_ms, magnitude
        (i.e., run signal_processor.get_processed() first)
        """
        if "magnitude" not in df.columns:
            raise ValueError("DataFrame must have a 'magnitude' column. Run signal_processor.get_processed() first.")

        ts = df["timestamp_ms"].values
        mag = df["magnitude"].values

        in_freefall = mag < self.freefall_threshold
        jumps: List[JumpEvent] = []

        i = 0
        while i < len(in_freefall):
            if not in_freefall[i]:
                i += 1
                continue

            # Found start of a freefall window
            start_idx = i
            while i < len(in_freefall) and in_freefall[i]:
                i += 1
            end_idx = i - 1

            duration_ms = ts[end_idx] - ts[start_idx]

            if duration_ms < self.min_freefall_ms:
                continue
            if duration_ms > self.max_freefall_ms:
                continue

            # Look for takeoff spike in context window before freefall
            pre_mask = (ts >= ts[start_idx] - self.context_window_ms) & (ts < ts[start_idx])
            pre_mag = mag[pre_mask]

            if len(pre_mag) == 0 or pre_mag.max() < self.takeoff_threshold:
                continue

            # Look for landing spike in context window after freefall
            post_mask = (ts > ts[end_idx]) & (ts <= ts[end_idx] + self.context_window_ms)
            post_mag = mag[post_mask]

            if len(post_mag) == 0 or post_mag.max() < self.takeoff_threshold:
                continue

            # All checks passed — compute height
            flight_time_s = duration_ms / 1000.0
            height_m = G * (flight_time_s ** 2) / 8.0
            height_cm = height_m * 100.0

            jump = JumpEvent(
                start_ms=float(ts[start_idx]),
                end_ms=float(ts[end_idx]),
                flight_time_s=flight_time_s,
                height_m=height_m,
                height_cm=height_cm,
                takeoff_peak=float(pre_mag.max()),
                landing_peak=float(post_mag.max()),
            )
            jumps.append(jump)

        return jumps


def summarize_jumps(jumps: List[JumpEvent]) -> dict:
    """Return a summary dict of detected jump statistics."""
    if not jumps:
        return {"count": 0}

    heights = [j.height_cm for j in jumps]
    flights = [j.flight_time_s * 1000 for j in jumps]

    return {
        "count": len(jumps),
        "mean_height_cm": round(np.mean(heights), 1),
        "max_height_cm": round(np.max(heights), 1),
        "min_height_cm": round(np.min(heights), 1),
        "std_height_cm": round(np.std(heights), 1),
        "mean_flight_ms": round(np.mean(flights), 1),
    }


def print_jump_report(jumps: List[JumpEvent]):
    """Pretty-print a jump detection report to stdout."""
    if not jumps:
        print("[jump_detector] No jumps detected.")
        return

    print(f"\n{'='*60}")
    print(f"  JUMP DETECTION REPORT — {len(jumps)} jump(s) detected")
    print(f"{'='*60}")
    for idx, j in enumerate(jumps, 1):
        print(f"  #{idx:02d}  {j}")

    stats = summarize_jumps(jumps)
    print(f"\n  Summary:")
    print(f"    Mean height : {stats['mean_height_cm']:.1f} cm")
    print(f"    Max height  : {stats['max_height_cm']:.1f} cm")
    print(f"    Std dev     : {stats['std_height_cm']:.1f} cm")
    print(f"    Mean flight : {stats['mean_flight_ms']:.0f} ms")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import os
    from src.simulator import simulate_jump_session, save_simulation
    from src.signal_processor import get_processed

    csv_path = os.path.join("data", "simulated", "jump_session.csv")
    if not os.path.exists(csv_path):
        df_raw = simulate_jump_session(n_jumps=5)
        save_simulation(df_raw, "jump_session.csv")

    df = get_processed(csv_path)
    detector = JumpDetector()
    jumps = detector.detect(df)
    print_jump_report(jumps)
