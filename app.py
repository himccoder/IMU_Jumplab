"""
IMU Activity Classifier — GUI
==============================
Tkinter interface for recording a short clip from the ESP32 and
classifying the activity in real time.

Usage:
    python app.py

Requirements:
    - Trained model at data/model.joblib  (run: python main.py train)
    - ESP32 flashed and connected over USB serial
"""

import os
import sys
import tempfile
import threading
from collections import Counter

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join("data", "model.joblib")

ACTIVITY_COLORS = {
    "still": "#64748B",
    "walk":  "#10B981",
    "run":   "#F59E0B",
    "jump":  "#EF4444",
}
ACTIVITIES = ["still", "walk", "run", "jump"]

BG          = "#F0F4F8"
CARD_BG     = "#FFFFFF"
ACCENT      = "#4F46E5"
TEXT        = "#1E293B"
SUBTEXT     = "#64748B"
BTN_FG      = "#FFFFFF"
BAR_TRACK   = "#E2E8F0"
BAR_MAX_PX  = 320


# ── Main application ──────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IMU Activity Classifier")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.clf = None
        self.le  = None
        self._load_model()
        self._build_ui()
        self._refresh_ports()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            return
        try:
            from src.classifier import load_model
            self.clf, self.le = load_model(MODEL_PATH)
        except Exception as exc:
            print(f"[app] Could not load model: {exc}")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_settings()
        self._build_record_button()
        self._build_status()
        self._build_results()
        self._build_prediction()
        self._build_footer()
        self.geometry("520x640")

    def _build_header(self):
        frame = tk.Frame(self, bg=ACCENT, pady=20)
        frame.pack(fill="x")
        tk.Label(frame, text="IMU Activity Classifier",
                 font=("Helvetica", 22, "bold"),
                 bg=ACCENT, fg="white").pack()
        tk.Label(frame, text="ESP32-S3  +  MPU6050",
                 font=("Helvetica", 11),
                 bg=ACCENT, fg="#C7D2FE").pack()

    def _build_settings(self):
        card = tk.Frame(self, bg=CARD_BG, padx=24, pady=16)
        card.pack(fill="x", padx=20, pady=(14, 0))

        # Port row
        row = tk.Frame(card, bg=CARD_BG)
        row.pack(fill="x", pady=4)
        tk.Label(row, text="Serial port", width=12, anchor="w",
                 font=("Helvetica", 11), bg=CARD_BG, fg=TEXT).pack(side="left")
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(row, textvariable=self.port_var,
                                       width=14, state="readonly")
        self.port_combo.pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Refresh", command=self._refresh_ports).pack(side="left")

        # Duration row
        row2 = tk.Frame(card, bg=CARD_BG)
        row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Duration (s)", width=12, anchor="w",
                 font=("Helvetica", 11), bg=CARD_BG, fg=TEXT).pack(side="left")
        self.dur_var = tk.IntVar(value=3)
        tk.Spinbox(row2, from_=1, to=10, textvariable=self.dur_var,
                   width=4, font=("Helvetica", 11)).pack(side="left")

    def _build_record_button(self):
        frame = tk.Frame(self, bg=BG, pady=16)
        frame.pack()
        self.record_btn = tk.Button(
            frame,
            text="  Record & Classify  ",
            font=("Helvetica", 15, "bold"),
            bg=ACCENT, fg=BTN_FG,
            activebackground="#4338CA",
            activeforeground=BTN_FG,
            relief="flat", padx=24, pady=12,
            cursor="hand2",
            command=self._on_record_clicked,
        )
        self.record_btn.pack()

    def _build_status(self):
        frame = tk.Frame(self, bg=BG)
        frame.pack()
        self.status_var = tk.StringVar(value="Ready")
        self.status_lbl = tk.Label(
            frame,
            textvariable=self.status_var,
            font=("Helvetica", 13),
            bg=BG, fg=SUBTEXT,
        )
        self.status_lbl.pack()

    def _build_results(self):
        card = tk.Frame(self, bg=CARD_BG, padx=24, pady=16)
        card.pack(fill="x", padx=20, pady=(10, 0))

        tk.Label(card, text="Results", font=("Helvetica", 13, "bold"),
                 bg=CARD_BG, fg=TEXT).pack(anchor="w", pady=(0, 8))

        self.bar_fills  = {}
        self.pct_labels = {}

        for act in ACTIVITIES:
            color = ACTIVITY_COLORS[act]

            row = tk.Frame(card, bg=CARD_BG, pady=4)
            row.pack(fill="x")

            tk.Label(row, text=act.upper(), font=("Helvetica", 11, "bold"),
                     bg=CARD_BG, fg=color, width=7, anchor="w").pack(side="left")

            # Track (background)
            track = tk.Frame(row, bg=BAR_TRACK, width=BAR_MAX_PX, height=20)
            track.pack(side="left", padx=(4, 8))
            track.pack_propagate(False)

            # Fill
            fill = tk.Frame(track, bg=color, width=0, height=20)
            fill.pack(side="left", fill="y")
            self.bar_fills[act] = fill

            pct_lbl = tk.Label(row, text="—", font=("Helvetica", 11),
                               bg=CARD_BG, fg=TEXT, width=5, anchor="w")
            pct_lbl.pack(side="left")
            self.pct_labels[act] = pct_lbl

    def _build_prediction(self):
        frame = tk.Frame(self, bg=BG, pady=10)
        frame.pack()
        self.pred_var = tk.StringVar(value="")
        tk.Label(frame, textvariable=self.pred_var,
                 font=("Helvetica", 30, "bold"),
                 bg=BG, fg=ACCENT).pack()

    def _build_footer(self):
        if self.clf:
            classes = list(self.le.classes_)
            msg = f"Model ready  |  Classes: {', '.join(classes)}"
            color = "#10B981"
        else:
            msg = "No model found — run:  python main.py train"
            color = "#EF4444"

        tk.Label(self, text=msg, font=("Helvetica", 9),
                 bg=BG, fg=color).pack(pady=(0, 10))

    # ── Port refresh ──────────────────────────────────────────────────────────

    def _refresh_ports(self):
        try:
            import serial.tools.list_ports
            ports = [p.device for p in serial.tools.list_ports.comports()]
        except Exception:
            ports = []
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])

    # ── Recording flow ────────────────────────────────────────────────────────

    def _on_record_clicked(self):
        if not self.clf:
            self.status_var.set("No model — run: python main.py train")
            return
        port = self.port_var.get().strip()
        if not port:
            self.status_var.set("Select a serial port first")
            return

        self.record_btn.config(state="disabled")
        self.pred_var.set("")
        self._reset_bars()
        self._countdown(3, port)

    def _countdown(self, n: int, port: str):
        """Tick countdown on the main thread, then launch recording thread."""
        if n > 0:
            self.status_var.set(f"Get ready...  {n}")
            self.after(1000, lambda: self._countdown(n - 1, port))
        else:
            self.status_var.set(f"Recording {self.dur_var.get()}s  ...")
            threading.Thread(
                target=self._do_record,
                args=(port, self.dur_var.get()),
                daemon=True,
            ).start()

    def _do_record(self, port: str, duration: int):
        """Runs in a background thread — no direct UI access."""
        from src.data_collector import collect_clip
        from src.classifier import extract_window_features_from_axes
        from src.classifier import WINDOW_SAMPLES, STEP_SAMPLES

        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp.close()

        n = collect_clip(port, tmp.name, duration_s=float(duration))

        if n < WINDOW_SAMPLES:
            self.after(0, lambda: self._show_error(
                f"Only {n} samples received — check serial connection"
            ))
            os.unlink(tmp.name)
            return

        df = pd.read_csv(tmp.name)
        os.unlink(tmp.name)

        ax = df["ax"].values.astype(np.float32)
        ay = df["ay"].values.astype(np.float32)
        az = df["az"].values.astype(np.float32)

        votes: list = []
        for i in range(0, len(ax) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            feats    = extract_window_features_from_axes(
                ax[i: i + WINDOW_SAMPLES],
                ay[i: i + WINDOW_SAMPLES],
                az[i: i + WINDOW_SAMPLES],
            )
            pred_idx = int(self.clf.predict(feats.reshape(1, -1))[0])
            votes.append(pred_idx)

        if not votes:
            self.after(0, lambda: self._show_error("No windows extracted"))
            return

        tally      = Counter(votes)
        total      = len(votes)
        classes    = list(self.le.classes_)
        winner_idx = tally.most_common(1)[0][0]
        winner     = str(self.le.inverse_transform([winner_idx])[0])
        winner_pct = tally[winner_idx] / total * 100

        percentages = {}
        for act in ACTIVITIES:
            if act in classes:
                idx = int(self.le.transform([act])[0])
                percentages[act] = tally.get(idx, 0) / total * 100
            else:
                percentages[act] = 0.0

        self.after(0, lambda: self._show_results(
            n, total, winner, winner_pct, percentages
        ))

    # ── UI update helpers (always called on main thread via after()) ──────────

    def _show_results(self, n_samples, n_windows, winner, winner_pct, pcts):
        for act in ACTIVITIES:
            pct   = pcts[act]
            width = int(pct / 100 * BAR_MAX_PX)
            self.bar_fills[act].config(width=max(width, 1))
            self.pct_labels[act].config(text=f"{pct:.0f}%")

        self.pred_var.set(f"{winner.upper()}")
        self.status_var.set(
            f"{n_samples} samples  |  {n_windows} windows  |  {winner_pct:.0f}% confidence"
        )
        self.record_btn.config(state="normal")

    def _show_error(self, msg: str):
        self.status_var.set(msg)
        self.record_btn.config(state="normal")

    def _reset_bars(self):
        for act in ACTIVITIES:
            self.bar_fills[act].config(width=1)
            self.pct_labels[act].config(text="—")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
