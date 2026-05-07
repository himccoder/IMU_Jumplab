"""
Serial Data Collector
---------------------
Reads CSV lines streamed from the ESP32 over USB serial and saves to a CSV file.

Expected serial format (set in firmware):
    timestamp_ms,ax,ay,az
    1234,0.1234,-0.0512,9.7821
    ...

Public API
----------
  list_ports()                                   — print available serial ports
  collect_clip(port, path, duration_s, baud)     — record a short clip to CSV
"""

import csv
import os
import sys
import time

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

BAUD_RATE = 115200


def list_ports():
    """Print all detected serial ports."""
    if not SERIAL_AVAILABLE:
        print("pyserial is not installed. Run: pip install pyserial")
        return
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports detected.")
    else:
        print("Available serial ports:")
        for p in ports:
            print(f"  {p.device:15s}  {p.description}")


def collect_clip(
    port: str,
    output_path: str,
    duration_s: float = 3.0,
    baud: int = BAUD_RATE,
) -> int:
    """
    Record a fixed-duration IMU clip from the ESP32 and save to CSV.

    Saves columns: timestamp_ms, ax, ay, az  (no label column — label comes
    from the folder the file is stored in).

    Parameters
    ----------
    port        : Serial port, e.g. 'COM6' or '/dev/ttyUSB0'
    output_path : Destination CSV file
    duration_s  : Recording duration in seconds (default 3)
    baud        : Baud rate (must match firmware, default 115200)

    Returns
    -------
    Number of samples saved.
    """
    if not SERIAL_AVAILABLE:
        print("ERROR: pyserial not installed.  Run: pip install pyserial")
        sys.exit(1)

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    try:
        ser = serial.Serial(port, baud, timeout=2.0)
    except serial.SerialException as exc:
        print(f"ERROR: Could not open {port}: {exc}")
        sys.exit(1)

    # Brief pause so the ESP32 settles, then flush any boot noise
    time.sleep(0.5)
    ser.flushInput()

    rows: list = []
    start = time.time()
    deadline = start + duration_s

    try:
        while time.time() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or "timestamp" in line or line.startswith("ax"):
                continue
            parts = line.split(",")
            if len(parts) != 4:
                continue
            try:
                ts = float(parts[0])
                ax = float(parts[1])
                ay = float(parts[2])
                az = float(parts[3])
            except ValueError:
                continue

            rows.append([ts, ax, ay, az])

            remaining = deadline - time.time()
            if len(rows) % 50 == 0:
                print(
                    f"\r  {len(rows):>4} samples  |  {remaining:.1f}s left   ",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n  Stopped early.")

    ser.close()
    print()  # newline after the progress line

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "ax", "ay", "az"])
        writer.writerows(rows)

    return len(rows)
