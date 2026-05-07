"""
IMU Activity Classifier
=======================
Record labelled 3-second clips from the ESP32, train a classifier, then
identify activities from new live recordings.

Commands
--------
  ports      List available serial ports
  status     Show how many clips you have per activity
  record     Record one or more 3-second clips for an activity
  train      Train the classifier from all clips in data/raw/
  classify   Record 3 seconds live and predict the activity

Quick start
-----------
  python main.py ports
  python main.py record --port COM6 --activity still
  python main.py record --port COM6 --activity walk
  python main.py record --port COM6 --activity run
  python main.py record --port COM6 --activity jump
  python main.py train
  python main.py classify --port COM6
"""

import argparse
import glob
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

ACTIVITIES = ["still", "walk", "run", "jump"]
RAW_DIR    = os.path.join("data", "raw")
MODEL_PATH = os.path.join("data", "model.joblib")

MIN_SAMPLES_PER_CLIP = 50   # sanity floor — less than this means no serial data


# ---------------------------------------------------------------------------
# ports
# ---------------------------------------------------------------------------

def cmd_ports(args):
    """List available serial ports."""
    from src.data_collector import list_ports
    list_ports()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args):
    """Show clip counts per activity and whether a model exists."""
    print()
    print("  Activity   Clips   Files")
    print("  --------   -----   -----")
    total = 0
    for act in ACTIVITIES:
        folder = os.path.join(RAW_DIR, act)
        clips  = sorted(glob.glob(os.path.join(folder, "*.csv"))) if os.path.isdir(folder) else []
        names  = ", ".join(os.path.basename(c) for c in clips[:4])
        if len(clips) > 4:
            names += f"  (+{len(clips)-4} more)"
        print(f"  {act:<8}   {len(clips):<5}   {names}")
        total += len(clips)

    model_status = "ready  (data/model.joblib)" if os.path.exists(MODEL_PATH) else "not trained yet"
    print()
    print(f"  Total clips : {total}")
    print(f"  Model       : {model_status}")
    print()


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

def cmd_record(args):
    """Record one or more 3-second clips for a given activity."""
    from src.data_collector import collect_clip

    activity = args.activity.lower()
    if activity not in ACTIVITIES:
        print(f"ERROR: --activity must be one of: {ACTIVITIES}")
        sys.exit(1)

    folder = os.path.join(RAW_DIR, activity)
    os.makedirs(folder, exist_ok=True)

    while True:
        existing = sorted(glob.glob(os.path.join(folder, f"{activity}*.csv")))
        clip_num = len(existing) + 1
        out_path = os.path.join(folder, f"{activity}{clip_num}.csv")

        print()
        print(f"  Activity  : {activity.upper()}")
        print(f"  Output    : {out_path}")
        print(f"  Duration  : {args.duration}s")
        input("  Press ENTER when ready...")

        _countdown(int(args.duration))

        n = collect_clip(args.port, out_path, duration_s=args.duration, baud=args.baud)

        if n < MIN_SAMPLES_PER_CLIP:
            print(f"  WARNING: Only {n} samples received.")
            print("  Check that the ESP32 is connected and streaming (try: python main.py ports)")
            os.remove(out_path)
        else:
            print(f"  Saved {n} samples ({n/100:.1f}s)  ->  {out_path}")

        if args.once:
            break
        again = input("\n  Record another clip? [y/N]: ").strip().lower()
        if again != "y":
            break

    print(f"\n  Done. Run 'python main.py status' to see your clip counts.")


def _countdown(seconds: int):
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end="", flush=True)
        time.sleep(1)
    print("  GO!")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args):
    """Train the activity classifier from all clips in data/raw/<activity>/."""
    import matplotlib
    matplotlib.use("Agg")  # save plots to file, no popup needed during training

    from sklearn.preprocessing import LabelEncoder
    from src.classifier import extract_features, train, evaluate, save_model, feature_importance_plot

    X_parts: list = []
    y_parts: list = []

    print()
    print("  Loading clips:")
    for activity in ACTIVITIES:
        folder = os.path.join(RAW_DIR, activity)
        clips  = sorted(glob.glob(os.path.join(folder, "*.csv"))) if os.path.isdir(folder) else []

        if not clips:
            print(f"  {activity:<6}  (no clips — skipping)")
            continue

        act_wins = 0
        for path in clips:
            df = pd.read_csv(path)
            if not {"ax", "ay", "az"}.issubset(df.columns):
                continue
            df["label"] = activity
            X, y = extract_features(df)
            if X.shape[0] > 0:
                X_parts.append(X)
                y_parts.append(y)
                act_wins += X.shape[0]

        print(f"  {activity:<6}  {len(clips)} clip(s)  ->  {act_wins} windows")

    if not X_parts:
        print()
        print("  ERROR: No data found in data/raw/. Record clips first:")
        print("         python main.py record --port COM6 --activity still")
        sys.exit(1)

    X     = np.vstack(X_parts)
    y_str = np.concatenate(y_parts)
    le    = LabelEncoder()
    y     = le.fit_transform(y_str)

    classes = list(le.classes_)
    print()
    print(f"  Dataset  : {X.shape[0]} windows | Classes: {classes}")

    if len(classes) < 2:
        print("  ERROR: Need at least 2 activity classes to train.")
        sys.exit(1)

    clf = train(X, y, le)

    os.makedirs("data/reports", exist_ok=True)
    save_model(clf, le)
    evaluate(clf, X, y, le, save_path="data/reports/confusion_matrix.png")
    feature_importance_plot(clf, X=X, y=y, save_path="data/reports/feature_importances.png")

    print()
    print(f"  Model saved to {MODEL_PATH}")
    print(f"  Confusion matrix -> data/reports/confusion_matrix.png")
    print()
    print("  Next:  python main.py classify --port COM6")


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def cmd_classify(args):
    """Record a 3-second clip live and predict the activity."""
    import tempfile
    from src.data_collector import collect_clip
    from src.classifier import load_model, extract_window_features_from_axes
    from src.classifier import WINDOW_SAMPLES, STEP_SAMPLES

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No model found at '{MODEL_PATH}'.")
        print("       Train first:  python main.py train")
        sys.exit(1)

    clf, le  = load_model(MODEL_PATH)
    classes  = list(le.classes_)

    print()
    print(f"  Model loaded.  Classes: {classes}")
    print("  Press Ctrl+C to quit.\n")

    while True:
        input("  Press ENTER to start recording...")
        _countdown(int(args.duration))

        # Write to a temp file so we don't litter the data folder
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp.close()

        n = collect_clip(args.port, tmp.name, duration_s=args.duration, baud=args.baud)

        if n < WINDOW_SAMPLES:
            print(f"  Only {n} samples received — need at least {WINDOW_SAMPLES}.")
            print("  Check the serial connection and try again.")
            os.unlink(tmp.name)
            continue

        df = pd.read_csv(tmp.name)
        os.unlink(tmp.name)

        ax_arr = df["ax"].values.astype(np.float32)
        ay_arr = df["ay"].values.astype(np.float32)
        az_arr = df["az"].values.astype(np.float32)

        # Sliding-window classification
        votes: list = []
        for i in range(0, len(ax_arr) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            feats    = extract_window_features_from_axes(
                ax_arr[i: i + WINDOW_SAMPLES],
                ay_arr[i: i + WINDOW_SAMPLES],
                az_arr[i: i + WINDOW_SAMPLES],
            )
            pred_idx = int(clf.predict(feats.reshape(1, -1))[0])
            votes.append(pred_idx)

        if not votes:
            print("  Could not extract any windows. Try again.")
            continue

        tally        = Counter(votes)
        total        = len(votes)
        winner_idx   = tally.most_common(1)[0][0]
        winner_label = str(le.inverse_transform([winner_idx])[0])
        winner_pct   = tally[winner_idx] / total * 100

        # Results display
        BAR_WIDTH = 24
        print()
        print(f"  {n} samples ({n/100:.1f}s)  |  {total} windows")
        print()
        print(f"  {'Activity':<8}  {'':>{BAR_WIDTH}}   Share")
        print(f"  {'-'*8}  {'-'*BAR_WIDTH}   -----")
        for cls in classes:
            idx  = int(le.transform([cls])[0])
            pct  = tally.get(idx, 0) / total * 100
            bar  = "#" * int(round(pct / 100 * BAR_WIDTH))
            flag = "  <-- PREDICTED" if cls == winner_label else ""
            print(f"  {cls:<8}  {bar:<{BAR_WIDTH}}   {pct:5.1f}%{flag}")

        print()
        print(f"  Predicted: {winner_label.upper()}  ({winner_pct:.0f}% of windows)")
        print()

        again = input("  Classify again? [y/N]: ").strip().lower()
        if again != "y":
            break


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="IMU Activity Classifier — record, train, classify",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ports
    sub.add_parser("ports", help="List available serial ports")

    # status
    sub.add_parser("status", help="Show clip counts per activity")

    # record
    p_rec = sub.add_parser("record", help="Record a 3-second clip for one activity")
    p_rec.add_argument("--port",     required=True,  help="Serial port (e.g. COM6)")
    p_rec.add_argument("--activity", required=True,  choices=ACTIVITIES,
                       help="Activity label: still / walk / run / jump")
    p_rec.add_argument("--duration", type=float, default=3.0,
                       help="Clip duration in seconds (default: 3)")
    p_rec.add_argument("--baud",     type=int,   default=115200)
    p_rec.add_argument("--once",     action="store_true",
                       help="Record exactly one clip then exit (no prompt)")

    # train
    sub.add_parser("train", help="Train classifier from all clips in data/raw/")

    # classify
    p_clf = sub.add_parser("classify", help="Record 3 seconds live and predict the activity")
    p_clf.add_argument("--port",     required=True, help="Serial port (e.g. COM6)")
    p_clf.add_argument("--duration", type=float, default=3.0,
                       help="Recording duration in seconds (default: 3)")
    p_clf.add_argument("--baud",     type=int,   default=115200)

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "ports":    cmd_ports,
        "status":   cmd_status,
        "record":   cmd_record,
        "train":    cmd_train,
        "classify": cmd_classify,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
