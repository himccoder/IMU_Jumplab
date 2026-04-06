"""
Visualizer Module
-----------------
Matplotlib + Seaborn plots for the IMU jump analysis pipeline.

Available plots:
  plot_raw_axes()         — Raw ax, ay, az time series (3-panel)
  plot_filtered_vs_raw()  — Raw vs filtered overlay per axis
  plot_magnitude()        — Magnitude with freefall threshold line
  plot_jump_events()      — Magnitude with jump windows shaded and height labels
  plot_activity_comparison() — Side-by-side magnitude comparison of activity types
  save_report()           — Save all key figures to an output directory
"""

import os
from typing import List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.jump_detector import JumpEvent

# Global style
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})

G = 9.81
FREEFALL_THRESHOLD = 0.35 * G  # default, matches JumpDetector


def _time_axis(df: pd.DataFrame) -> np.ndarray:
    """Return time in seconds from the first sample."""
    return (df["timestamp_ms"].values - df["timestamp_ms"].values[0]) / 1000.0


def plot_raw_axes(df: pd.DataFrame, title: str = "Raw IMU Accelerometer Data",
                  save_path: Optional[str] = None) -> plt.Figure:
    """Three-panel plot of ax, ay, az over time."""
    t = _time_axis(df)
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    colors = ["#2196F3", "#4CAF50", "#F44336"]
    labels = ["ax (m/s²)", "ay (m/s²)", "az (m/s²)"]
    cols = ["ax", "ay", "az"]

    for ax_plot, col, color, lbl in zip(axes, cols, colors, labels):
        ax_plot.plot(t, df[col].values, color=color, lw=0.7, alpha=0.85)
        ax_plot.set_ylabel(lbl)
        ax_plot.axhline(0, color="white", lw=0.5, ls="--", alpha=0.4)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[visualizer] Saved -> {save_path}")
    return fig


def plot_filtered_vs_raw(df: pd.DataFrame, axis: str = "az",
                         save_path: Optional[str] = None) -> plt.Figure:
    """Overlay raw vs filtered signal for one axis."""
    if f"{axis}_filt" not in df.columns:
        raise ValueError(f"Filtered column '{axis}_filt' not found. Run signal_processor.apply_lowpass() first.")

    t = _time_axis(df)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, df[axis].values, color="#90A4AE", lw=0.6, alpha=0.7, label="Raw")
    ax.plot(t, df[f"{axis}_filt"].values, color="#E53935", lw=1.2, label="Filtered (Butterworth LP)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{axis} (m/s²)")
    ax.set_title(f"Raw vs Filtered — {axis}")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[visualizer] Saved -> {save_path}")
    return fig


def plot_magnitude(df: pd.DataFrame, freefall_threshold: float = FREEFALL_THRESHOLD,
                   title: str = "Acceleration Magnitude",
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot total acceleration magnitude with the freefall detection threshold.
    Highlights the freefall zone (below threshold) in orange.
    """
    if "magnitude" not in df.columns:
        raise ValueError("'magnitude' column not found. Run signal_processor.compute_magnitude() first.")

    t = _time_axis(df)
    mag = df["magnitude"].values

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, mag, color="#1565C0", lw=1.0, label="|a| magnitude")
    ax.axhline(freefall_threshold, color="#FF6F00", lw=1.5, ls="--",
               label=f"Freefall threshold ({freefall_threshold:.2f} m/s²)")
    ax.fill_between(t, 0, mag, where=(mag < freefall_threshold),
                    color="#FF6F00", alpha=0.25, label="Freefall zone")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|a| (m/s²)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[visualizer] Saved -> {save_path}")
    return fig


def plot_jump_events(df: pd.DataFrame, jumps: List[JumpEvent],
                     freefall_threshold: float = FREEFALL_THRESHOLD,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Magnitude plot with each detected jump shaded and labelled with its height.
    """
    if "magnitude" not in df.columns:
        raise ValueError("'magnitude' column not found.")

    t = _time_axis(df)
    ts_abs = df["timestamp_ms"].values
    mag = df["magnitude"].values
    t0 = df["timestamp_ms"].values[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, mag, color="#1565C0", lw=1.0, zorder=2, label="|a| magnitude")
    ax.axhline(freefall_threshold, color="#FF6F00", lw=1.2, ls="--",
               label=f"Freefall threshold")

    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]
    for idx, jump in enumerate(jumps):
        c = colors[idx % len(colors)]
        t_start = (jump.start_ms - t0) / 1000.0
        t_end = (jump.end_ms - t0) / 1000.0
        ax.axvspan(t_start, t_end, alpha=0.3, color=c, zorder=1)
        y_pos = mag.max() * 0.85
        ax.text(
            (t_start + t_end) / 2, y_pos,
            f"#{idx+1}\n{jump.height_cm:.1f} cm",
            ha="center", va="top", fontsize=8.5, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc=c, alpha=0.6, ec="none"),
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|a| (m/s²)")
    title_suffix = f"({len(jumps)} jump(s) detected)" if jumps else "(no jumps detected)"
    ax.set_title(f"Jump Detection Results — {title_suffix}")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[visualizer] Saved -> {save_path}")
    return fig


def plot_jump_height_bar(jumps: List[JumpEvent], save_path: Optional[str] = None) -> plt.Figure:
    """Bar chart of detected jump heights."""
    if not jumps:
        print("[visualizer] No jumps to plot.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No jumps detected", ha="center", va="center", transform=ax.transAxes)
        return fig

    heights = [j.height_cm for j in jumps]
    labels = [f"#{i+1}" for i in range(len(jumps))]
    mean_h = np.mean(heights)

    fig, ax = plt.subplots(figsize=(max(6, len(jumps) * 1.2), 5))
    bars = ax.bar(labels, heights, color=sns.color_palette("muted", len(jumps)), edgecolor="white")
    ax.axhline(mean_h, color="#E53935", ls="--", lw=1.5, label=f"Mean: {mean_h:.1f} cm")

    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Jump #")
    ax.set_ylabel("Height (cm)")
    ax.set_title("Detected Jump Heights")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[visualizer] Saved -> {save_path}")
    return fig


def plot_activity_comparison(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    If a 'label' column is present (simulated data), plot magnitude distributions
    per activity type using a violin plot.
    """
    if "label" not in df.columns:
        print("[visualizer] No 'label' column — skipping activity comparison.")
        return plt.figure()

    if "magnitude" not in df.columns:
        raise ValueError("'magnitude' column not found.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Activity Comparison — Magnitude Distribution", fontweight="bold")

    sns.violinplot(data=df, x="label", y="magnitude", ax=axes[0],
                   order=["idle", "walk", "run", "jump", "freefall"],
                   palette="muted", inner="quartile")
    axes[0].set_title("Violin Plot")
    axes[0].set_ylabel("|a| (m/s²)")
    axes[0].set_xlabel("Activity Label")

    sns.boxplot(data=df, x="label", y="magnitude", ax=axes[1],
                order=["idle", "walk", "run", "jump", "freefall"],
                palette="muted")
    axes[1].set_title("Box Plot")
    axes[1].set_ylabel("|a| (m/s²)")
    axes[1].set_xlabel("Activity Label")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[visualizer] Saved -> {save_path}")
    return fig


def save_report(df: pd.DataFrame, jumps: List[JumpEvent], output_dir: str = "data/reports"):
    """Save all key visualizations as PNG files to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    plot_raw_axes(df, save_path=os.path.join(output_dir, "01_raw_axes.png"))
    plot_filtered_vs_raw(df, save_path=os.path.join(output_dir, "02_filtered_vs_raw.png"))
    plot_magnitude(df, save_path=os.path.join(output_dir, "03_magnitude.png"))
    plot_jump_events(df, jumps, save_path=os.path.join(output_dir, "04_jump_events.png"))
    plot_jump_height_bar(jumps, save_path=os.path.join(output_dir, "05_jump_heights.png"))
    plot_activity_comparison(df, save_path=os.path.join(output_dir, "06_activity_comparison.png"))

    plt.close("all")
    print(f"[visualizer] Report saved to: {output_dir}/")


if __name__ == "__main__":
    import os
    from src.simulator import simulate_jump_session, save_simulation
    from src.signal_processor import get_processed
    from src.jump_detector import JumpDetector

    csv_path = os.path.join("data", "simulated", "jump_session.csv")
    if not os.path.exists(csv_path):
        df_raw = simulate_jump_session(n_jumps=5)
        save_simulation(df_raw, "jump_session.csv")

    df = get_processed(csv_path)
    # Re-attach labels from simulated data
    df_raw2 = pd.read_csv(csv_path)
    if "label" in df_raw2.columns:
        df["label"] = df_raw2["label"].values

    detector = JumpDetector()
    jumps = detector.detect(df)
    save_report(df, jumps, output_dir="data/reports")
