"""
IMU Jump Analysis — Main Entry Point
-------------------------------------
Command-line interface for the full software pipeline.

Subcommands:
  simulate    — Generate synthetic IMU datasets for pipeline testing
  collect     — Stream data from the real ESP32 over serial USB
  process     — Filter + magnitude + jump detection on a CSV file
  classify    — Train & evaluate the ML activity classifier (simulated data)
  train       — Train the classifier on REAL recorded CSVs (one per activity)
  report      — Generate a full visual report from a processed CSV
  play        — Launch the Dino Jump game (IMU or demo mode)

Examples:
  # Test the full pipeline on simulated data (no hardware needed)
  python main.py simulate
  python main.py process --input data/simulated/jump_session.csv

  # Record real data (30 s per activity, replace COM3 with your port)
  python main.py collect --port COM3 --duration 30 --output data/raw/still.csv
  python main.py collect --port COM3 --duration 30 --output data/raw/walk.csv
  python main.py collect --port COM3 --duration 30 --output data/raw/run.csv
  python main.py collect --port COM3 --duration 30 --output data/raw/jump.csv

  # Train on real data
  python main.py train --still data/raw/still.csv --walk data/raw/walk.csv \
                       --run data/raw/run.csv --jump data/raw/jump.csv

  # Play the Dino game
  python main.py play --demo               # keyboard-only (no hardware)
  python main.py play --port COM3          # live IMU mode

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


def cmd_train(args):
    """
    Train the RandomForest classifier on real recorded CSVs.

    Each activity CSV is processed independently (filter + magnitude),
    labelled, and then combined into a single feature matrix for training.
    This avoids any boundary artefacts from concatenating raw signals.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from src.classifier import (
        extract_features, train, evaluate, save_model,
        feature_importance_plot,
    )
    from sklearn.preprocessing import LabelEncoder

    activity_files = {
        "still": args.still,
        "walk":  args.walk,
        "run":   args.run,
        "jump":  args.jump,
    }

    X_parts, y_parts = [], []

    for label, path in activity_files.items():
        if not path:
            print(f"[train] Skipping '{label}' — no file provided.")
            continue
        if not os.path.exists(path):
            print(f"[train] WARNING: '{label}' file not found: {path}")
            continue

        print(f"[train] Processing '{label}': {path}")
        df = pd.read_csv(path)
        df["label"] = label

        X_sess, y_sess = extract_features(df)
        if X_sess.shape[0] == 0:
            print(f"[train]   WARNING: No windows extracted from {path} — skipping.")
            continue

        X_parts.append(X_sess)
        y_parts.append(y_sess)
        print(f"[train]   → {X_sess.shape[0]} windows extracted.")

    if not X_parts:
        print("[train] No data found. Record CSV files first with 'python main.py collect'.")
        sys.exit(1)

    X = np.vstack(X_parts)
    y_str = np.concatenate(y_parts)

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    print(f"\n[train] Dataset ready — {X.shape[0]} windows, classes: {list(le.classes_)}")

    clf = train(X, y, le)
    os.makedirs("data/reports", exist_ok=True)

    # Save first so the model is always on disk even if plotting fails
    save_model(clf, le)

    evaluate(clf, X, y, le, save_path="data/reports/confusion_matrix.png")
    feature_importance_plot(clf, X=X, y=y,
                            save_path="data/reports/feature_importances.png")

    print("\n[train] Done! Model saved to data/model.joblib")
    print("[train] Now run:  python main.py play --port COMX")
    plt.show()


def cmd_play(args):
    """Launch the Dino Jump game."""
    from dino_game.game import DinoGame

    classifier = None
    if not args.demo:
        if not args.port:
            print("ERROR: Provide --port COMX, or use --demo for keyboard mode.")
            sys.exit(1)
        if not os.path.exists(args.model):
            print(f"ERROR: Model not found at '{args.model}'.")
            print("  Train first:  python main.py train --still ... --walk ... --run ... --jump ...")
            print("  Or test game: python main.py play --demo")
            sys.exit(1)
        from src.realtime_classifier import RealtimeClassifier
        classifier = RealtimeClassifier(
            port=args.port,
            baud=args.baud,
            model_path=args.model,
        )
        classifier.start()

    game = DinoGame(
        classifier=classifier,
        demo_mode=args.demo or (args.port is None),
    )
    try:
        game.run()
    finally:
        if classifier:
            classifier.stop()


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

    # --- train ---
    p_tr = sub.add_parser(
        "train",
        help="Train classifier on real recorded CSVs (one per activity)",
    )
    p_tr.add_argument("--still", type=str, default=None, help="CSV recorded while standing still")
    p_tr.add_argument("--walk",  type=str, default=None, help="CSV recorded while walking")
    p_tr.add_argument("--run",   type=str, default=None, help="CSV recorded while running")
    p_tr.add_argument("--jump",  type=str, default=None, help="CSV recorded while jumping")

    # --- play ---
    p_play = sub.add_parser("play", help="Launch the IMU Dino Jump game")
    p_play.add_argument("--demo",  action="store_true",
                        help="Demo/keyboard mode — no IMU hardware required")
    p_play.add_argument("--port",  type=str, default=None,
                        help="Serial port for live IMU (e.g. COM3)")
    p_play.add_argument("--baud",  type=int, default=115200)
    p_play.add_argument("--model", type=str, default="data/model.joblib",
                        help="Path to trained model (default: data/model.joblib)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "simulate": cmd_simulate,
        "collect":  cmd_collect,
        "process":  cmd_process,
        "classify": cmd_classify,
        "report":   cmd_report,
        "train":    cmd_train,
        "play":     cmd_play,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
