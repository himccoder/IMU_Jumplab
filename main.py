"""
IMU Jump Analysis — Main Entry Point
-------------------------------------
Command-line interface for the full software pipeline.

Subcommands:
  simulate    — Generate synthetic IMU datasets for pipeline testing
  collect     — Stream data from the real ESP32 over serial USB
  process     — Filter + magnitude + jump detection on a CSV file
  classify    — Train & evaluate the ML activity classifier
  report      — Generate a full visual report from a processed CSV

Examples:
  # Test the full pipeline on simulated data (no hardware needed)
  python main.py simulate
  python main.py process --input data/simulated/jump_session.csv

  # When hardware is available:
  python main.py collect --port COM3 --output data/my_session.csv --duration 30
  python main.py process --input data/my_session.csv

  # Train classifier on mixed activities
  python main.py classify

  # Full report (plots saved to data/reports/)
  python main.py report --input data/simulated/jump_session.csv
"""

import argparse
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_simulate(args):
    """Generate simulated IMU datasets."""
    from src.simulator import (
        simulate_jump_session,
        simulate_mixed_activities,
        save_simulation,
    )

    os.makedirs("data/simulated", exist_ok=True)

    print(f"[main] Generating jump session ({args.n_jumps} jumps)...")
    df_jump = simulate_jump_session(n_jumps=args.n_jumps)
    save_simulation(df_jump, "jump_session.csv")

    print("[main] Generating mixed activities session...")
    df_mixed = simulate_mixed_activities()
    # Save with labels for classifier
    df_mixed.to_csv("data/simulated/mixed_activities_labelled.csv", index=False)
    save_simulation(df_mixed, "mixed_activities.csv")

    print("\n[main] Simulation complete. Files written to data/simulated/")
    print("  Run:  python main.py process --input data/simulated/jump_session.csv")


def cmd_collect(args):
    """Stream data from the ESP32 over USB serial."""
    from src.data_collector import collect, list_ports

    if args.list_ports:
        list_ports()
        return

    if not args.port:
        print("ERROR: --port is required. Use --list-ports to find your device.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "data", exist_ok=True)
    collect(args.port, args.output, args.duration, args.baud)


def cmd_process(args):
    """Run the full signal processing + jump detection pipeline on a CSV."""
    from src.signal_processor import get_processed
    from src.jump_detector import JumpDetector, print_jump_report
    import matplotlib.pyplot as plt
    from src.visualizer import plot_magnitude, plot_jump_events, plot_jump_height_bar

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        print("Run 'python main.py simulate' first to generate test data.")
        sys.exit(1)

    print(f"[main] Processing: {args.input}")
    df = get_processed(args.input, cutoff_hz=args.cutoff)
    print(f"[main] Loaded {len(df)} samples spanning {df['timestamp_ms'].iloc[-1]/1000:.1f}s")
    print(f"[main] Magnitude stats:  mean={df['magnitude'].mean():.2f}  "
          f"min={df['magnitude'].min():.2f}  max={df['magnitude'].max():.2f}  (m/s²)")

    detector = JumpDetector(
        freefall_threshold_g=args.freefall_g,
        min_freefall_ms=args.min_freefall_ms,
    )
    jumps = detector.detect(df)
    print_jump_report(jumps)

    if args.plot:
        fig1 = plot_magnitude(df)
        fig2 = plot_jump_events(df, jumps)
        fig3 = plot_jump_height_bar(jumps)
        plt.show()

    if args.save_report:
        from src.visualizer import save_report
        # Re-attach labels if present in source file
        df_raw = pd.read_csv(args.input)
        if "label" in df_raw.columns:
            df["label"] = df_raw["label"].values
        save_report(df, jumps, output_dir="data/reports")

    return df, jumps


def cmd_classify(args):
    """Train and evaluate the activity classifier."""
    import matplotlib.pyplot as plt
    from src.signal_processor import get_processed
    from src.classifier import build_dataset, train, evaluate, save_model, feature_importance_plot

    labelled_path = args.input or "data/simulated/mixed_activities_labelled.csv"

    if not os.path.exists(labelled_path):
        print(f"[main] Labelled data not found at {labelled_path}.")
        print("[main] Generating simulated data first...")
        from src.simulator import simulate_mixed_activities
        os.makedirs("data/simulated", exist_ok=True)
        df_raw = simulate_mixed_activities()
        df_raw.to_csv(labelled_path, index=False)
        df_raw[["timestamp_ms", "ax", "ay", "az"]].to_csv(
            "data/simulated/mixed_activities.csv", index=False)

    print(f"[main] Loading labelled data: {labelled_path}")
    df_labelled = pd.read_csv(labelled_path)
    df = get_processed(df_labelled[["timestamp_ms", "ax", "ay", "az"]])
    df["label"] = df_labelled["label"].values

    X, y, le = build_dataset(df)
    print(f"[main] Feature matrix: {X.shape} | Classes: {list(le.classes_)}")

    clf = train(X, y, le)
    os.makedirs("data/reports", exist_ok=True)
    evaluate(clf, X, y, le, save_path="data/reports/confusion_matrix.png")
    feature_importance_plot(clf, save_path="data/reports/feature_importances.png")
    save_model(clf, le)

    plt.show()


def cmd_report(args):
    """Generate and save a complete visual report."""
    df, jumps = cmd_process(args)
    from src.visualizer import save_report
    df_raw = pd.read_csv(args.input)
    if "label" in df_raw.columns:
        df["label"] = df_raw["label"].values
    save_report(df, jumps, output_dir="data/reports")
    print("[main] Report complete. Open data/reports/ to view PNG files.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="IMU Vertical Jump Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- simulate ---
    p_sim = sub.add_parser("simulate", help="Generate synthetic IMU test data")
    p_sim.add_argument("--n-jumps", type=int, default=5, help="Number of jumps to simulate (default: 5)")

    # --- collect ---
    p_col = sub.add_parser("collect", help="Collect data from the ESP32 over serial USB")
    p_col.add_argument("--port", type=str, help="Serial port (e.g., COM3 or /dev/ttyUSB0)")
    p_col.add_argument("--output", type=str, default="data/session.csv", help="Output CSV path")
    p_col.add_argument("--duration", type=float, default=None, help="Recording duration in seconds")
    p_col.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    p_col.add_argument("--list-ports", action="store_true", help="List serial ports and exit")

    # --- process ---
    p_proc = sub.add_parser("process", help="Process a CSV: filter, detect jumps, print report")
    p_proc.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    p_proc.add_argument("--cutoff", type=float, default=10.0, help="Low-pass filter cutoff Hz (default: 10)")
    p_proc.add_argument("--freefall-g", type=float, default=0.35,
                        help="Freefall detection threshold in g (default: 0.35)")
    p_proc.add_argument("--min-freefall-ms", type=float, default=100.0,
                        help="Minimum freefall duration ms (default: 100)")
    p_proc.add_argument("--plot", action="store_true", help="Show interactive plots")
    p_proc.add_argument("--save-report", action="store_true", help="Save PNG report to data/reports/")

    # --- classify ---
    p_clf = sub.add_parser("classify", help="Train and evaluate the ML activity classifier")
    p_clf.add_argument("--input", type=str, default=None,
                       help="Path to labelled CSV (default: data/simulated/mixed_activities_labelled.csv)")

    # --- report ---
    p_rep = sub.add_parser("report", help="Generate full visual report from a CSV")
    p_rep.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    p_rep.add_argument("--cutoff", type=float, default=10.0)
    p_rep.add_argument("--freefall-g", type=float, default=0.35)
    p_rep.add_argument("--min-freefall-ms", type=float, default=100.0)
    p_rep.add_argument("--plot", action="store_true")
    p_rep.add_argument("--save-report", action="store_true", default=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "simulate": cmd_simulate,
        "collect": cmd_collect,
        "process": cmd_process,
        "classify": cmd_classify,
        "report": cmd_report,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
