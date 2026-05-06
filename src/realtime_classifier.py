"""
Real-time IMU Activity Classifier
----------------------------------
Reads serial data from the ESP32, maintains a sliding buffer, and
classifies activity in real-time using the trained model.

Feature extraction is imported directly from src.classifier so that
training and real-time inference use *exactly* the same Butterworth filter,
window size, and 44-feature vector — preventing any train/serve skew.

Event types pushed to the output queue
----------------------------------------
  {"type": "activity", "label": "walk"}          # emitted every ~0.4 s
  {"type": "jump", "height_cm": 35.2, "flight_ms": 535}
"""

import threading
import time
import queue
from collections import deque
from typing import List, Optional

import numpy as np

G              = 9.81
SAMPLE_RATE_HZ = 100

# Must match src/classifier.py
WINDOW_SAMPLES = 50    # 500 ms @ 100 Hz
STEP_SAMPLES   = 5     # classify every 50 ms

# Jump detector thresholds (real-hardware tuned)
FREEFALL_THRESHOLD   = 0.40 * G   # < 3.92 m/s² → candidate freefall
TAKEOFF_THRESHOLD    = 1.1  * G   # > 10.79 m/s² → push-off / landing spike
MIN_FREEFALL_SAMPLES = 8           # ≥ 80 ms air time
MAX_FREEFALL_SAMPLES = 200         # ≤ 2 s sanity cap
CONTEXT_SAMPLES      = 35          # look 350 ms back for takeoff spike

# How often to push an activity event even when the label has not changed
ACTIVITY_EMIT_INTERVAL_S = 0.4


class RealtimeClassifier:
    """
    Thread-safe real-time activity classifier + jump detector.

    Architecture
    ------------
    Background thread reads serial → stores raw (ax, ay, az) in a ring buffer.
    Every STEP_SAMPLES new samples the latest WINDOW_SAMPLES are extracted via
    src.classifier.extract_window_features_from_axes() (same function used
    during training) and fed to the loaded model.
    A separate state machine detects jump events independently of the ML model.
    All output goes through a thread-safe Queue polled by the game.

    Public methods
    --------------
    start()          – begin background thread
    stop()           – stop and join background thread
    get_event()      – non-blocking pop from event queue (returns None if empty)
    snapshot()       – copy of recent magnitude buffer for offline analysis
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        model_path: str = "data/model.joblib",
    ):
        from src.classifier import load_model
        self.clf, self.le = load_model(model_path)

        self.port = port
        self.baud = baud

        # Ring buffers — store raw tri-axial samples
        buf_size = max(600, WINDOW_SAMPLES * 4)
        self._ax_buf: deque = deque(maxlen=buf_size)
        self._ay_buf: deque = deque(maxlen=buf_size)
        self._az_buf: deque = deque(maxlen=buf_size)
        self._mag_buf: deque = deque(maxlen=buf_size)   # for jump SM + snapshot

        self._pre_mag: deque = deque(maxlen=CONTEXT_SAMPLES)
        self._new_samples = 0

        self._event_queue: queue.Queue = queue.Queue(maxsize=100)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Jump state machine
        self._jump_state     = "idle"
        self._freefall_count = 0

        self.current_label         = "still"
        self._last_emit_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="imu-reader"
        )
        self._thread.start()
        print(f"[realtime] Started on {self.port} @ {self.baud} baud")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        print("[realtime] Stopped.")

    def get_event(self) -> Optional[dict]:
        try:
            return self._event_queue.get_nowait()
        except queue.Empty:
            return None

    def snapshot(self) -> List[float]:
        """Return a copy of the recent magnitude buffer (thread-safe)."""
        return list(self._mag_buf)

    def snapshot_axes(self):
        """Return (ax_list, ay_list, az_list) copies of the recent raw buffers."""
        return list(self._ax_buf), list(self._ay_buf), list(self._az_buf)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _push_event(self, event: dict):
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            pass

    def _run(self):
        try:
            import serial
        except ImportError:
            print("[realtime] ERROR: pyserial not installed.")
            return

        try:
            ser = serial.Serial(self.port, self.baud, timeout=1.0)
        except Exception as exc:
            print(f"[realtime] Cannot open {self.port}: {exc}")
            return

        time.sleep(0.5)
        ser.flushInput()
        header_seen = False
        print("[realtime] Streaming from ESP32…")

        while self._running:
            try:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if "timestamp" in line or line.startswith("ax"):
                    header_seen = True
                    continue
                if not header_seen:
                    continue

                parts = line.split(",")
                if len(parts) != 4:
                    continue

                ax  = float(parts[1])
                ay  = float(parts[2])
                az  = float(parts[3])
                mag = float(np.sqrt(ax**2 + ay**2 + az**2))
                self._on_sample(ax, ay, az, mag)

            except (ValueError, UnicodeDecodeError):
                continue
            except Exception as exc:
                if self._running:
                    print(f"[realtime] Serial error: {exc}")
                break

        ser.close()
        print("[realtime] Serial closed.")

    def _on_sample(self, ax: float, ay: float, az: float, mag: float):
        self._ax_buf.append(ax)
        self._ay_buf.append(ay)
        self._az_buf.append(az)
        self._mag_buf.append(mag)
        self._pre_mag.append(mag)
        self._new_samples += 1

        # ── Jump state machine (rule-based, independent of ML) ────────
        if self._jump_state == "idle":
            if mag < FREEFALL_THRESHOLD:
                pre = list(self._pre_mag)
                if pre and max(pre) > TAKEOFF_THRESHOLD:
                    self._jump_state     = "freefall"
                    self._freefall_count = 1

        elif self._jump_state == "freefall":
            if mag < FREEFALL_THRESHOLD:
                self._freefall_count += 1
                if self._freefall_count > MAX_FREEFALL_SAMPLES:
                    self._jump_state     = "idle"
                    self._freefall_count = 0
            else:
                if (self._freefall_count >= MIN_FREEFALL_SAMPLES
                        and mag > TAKEOFF_THRESHOLD):
                    flight_s  = self._freefall_count / SAMPLE_RATE_HZ
                    height_cm = G * (flight_s ** 2) / 8.0 * 100.0
                    self._push_event({
                        "type":      "jump",
                        "height_cm": round(height_cm, 1),
                        "flight_ms": round(flight_s * 1000),
                    })
                    self.current_label   = "jump"
                    self._last_emit_time = time.time()
                    self._push_event({"type": "activity", "label": str("jump")})

                self._jump_state     = "idle"
                self._freefall_count = 0

        # ── ML activity classification every STEP_SAMPLES ────────────
        if (self._new_samples >= STEP_SAMPLES
                and len(self._ax_buf) >= WINDOW_SAMPLES):
            self._new_samples = 0

            ax_w = np.array(list(self._ax_buf)[-WINDOW_SAMPLES:], dtype=np.float32)
            ay_w = np.array(list(self._ay_buf)[-WINDOW_SAMPLES:], dtype=np.float32)
            az_w = np.array(list(self._az_buf)[-WINDOW_SAMPLES:], dtype=np.float32)

            try:
                from src.classifier import extract_window_features_from_axes
                feats    = extract_window_features_from_axes(ax_w, ay_w, az_w)
                pred_idx = self.clf.predict(feats.reshape(1, -1))[0]
                label    = str(self.le.inverse_transform([pred_idx])[0])
            except Exception as exc:
                print(f"[realtime] Prediction error: {exc}")
                return

            now           = time.time()
            label_changed = label != self.current_label
            due_for_emit  = (now - self._last_emit_time) >= ACTIVITY_EMIT_INTERVAL_S

            if label_changed or due_for_emit:
                self.current_label   = label
                self._last_emit_time = now
                self._push_event({"type": "activity", "label": label})


# ---------------------------------------------------------------------------
# Offline batch classification (used by the Analyze screen in the game)
# ---------------------------------------------------------------------------

def classify_buffer(
    ax_buf: List[float],
    ay_buf: List[float],
    az_buf: List[float],
    clf,
    le,
    window: int = WINDOW_SAMPLES,
    step:   int = STEP_SAMPLES,
) -> dict:
    """
    Classify a recorded segment offline.

    Parameters
    ----------
    ax_buf, ay_buf, az_buf : raw axis buffers (same length)
    clf, le                : loaded model + LabelEncoder

    Returns
    -------
    dict mapping label → fraction, sorted by fraction descending.
    e.g. {"run": 0.72, "walk": 0.20, "still": 0.08}
    """
    from collections import Counter
    from src.classifier import extract_window_features_from_axes

    n = min(len(ax_buf), len(ay_buf), len(az_buf))
    if n < window:
        return {}

    ax = np.array(ax_buf[-n:], dtype=np.float32)
    ay = np.array(ay_buf[-n:], dtype=np.float32)
    az = np.array(az_buf[-n:], dtype=np.float32)

    predictions = []
    for i in range(0, n - window + 1, step):
        try:
            feats = extract_window_features_from_axes(
                ax[i: i + window],
                ay[i: i + window],
                az[i: i + window],
            )
            idx = clf.predict(feats.reshape(1, -1))[0]
            predictions.append(str(le.inverse_transform([idx])[0]))
        except Exception as exc:
            print(f"[classify_buffer] Window error: {exc}")
            continue

    if not predictions:
        return {}

    counts = Counter(predictions)
    total  = len(predictions)
    result = {lbl: cnt / total for lbl, cnt in counts.items()}
    return dict(sorted(result.items(), key=lambda x: -x[1]))
