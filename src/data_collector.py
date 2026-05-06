"""
Serial Data Collector
---------------------
Reads CSV lines streamed from the ESP32 over USB serial and saves to a CSV file.

Expected serial format (set in firmware):
    timestamp_ms,ax,ay,az
    1234,0.1234,-0.0512,9.7821
    ...

Usage:
    python -m src.data_collector --port COM3 --output data/session_01.csv --duration 30
    python -m src.data_collector --port /dev/ttyUSB0 --output data/session_01.csv

Tip: Run `python -m src.data_collector --list-ports` to see available COM ports.
"""

import argparse
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
EXPECTED_HEADER = "timestamp_ms,ax,ay,az"


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
            print(f"  {p.device:15s} — {p.description}")


def collect(port: str, output_path: str, duration_s: float = None, baud: int = BAUD_RATE):
    """
    Open the serial port and stream data to a CSV file.

    Args:
        port:        Serial port name (e.g., 'COM3' or '/dev/ttyUSB0')
        output_path: Path to the output CSV file
        duration_s:  Recording duration in seconds (None = record until Ctrl+C)
        baud:        Baud rate (must match firmware, default 115200)
    """
    if not SERIAL_AVAILABLE:
        print("ERROR: pyserial not installed. Run: pip install pyserial")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    print(f"[collector] Opening {port} at {baud} baud...")
    print(f"[collector] Saving to: {output_path}")
    if duration_s:
        print(f"[collector] Recording for {duration_s:.1f} seconds. Press Ctrl+C to stop early.")
    else:
        print("[collector] Recording indefinitely. Press Ctrl+C to stop.")

    try:
        ser = serial.Serial(port, baud, timeout=2.0)
    except serial.SerialException as e:
        print(f"ERROR: Could not open {port}: {e}")
        sys.exit(1)

    # Give the ESP32 a moment to boot, then clear any garbage bytes that
    # accumulated during the USB enumeration phase.  We do NOT flush after
    # this sleep — the header line arrives at boot and we must not discard it.
    print("[collector] Waiting for ESP32 to start streaming...")
    time.sleep(0.5)
    ser.flushInput()   # flush only the pre-boot noise, then immediately read

    sample_count = 0
    start_time = time.time()

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp_ms", "ax", "ay", "az"])

        try:
            while True:
                if duration_s and (time.time() - start_time) >= duration_s:
                    print(f"\n[collector] Duration reached ({duration_s:.1f}s).")
                    break

                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # Skip the CSV header row — we already wrote our own above
                if line == EXPECTED_HEADER or line.startswith("timestamp"):
                    continue

                # Accept any line that parses as four numbers — no header gating
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

                writer.writerow([ts, ax, ay, az])
                sample_count += 1

                if sample_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"\r[collector] {sample_count} samples | {elapsed:.1f}s elapsed", end="", flush=True)

        except KeyboardInterrupt:
            print(f"\n[collector] Stopped by user.")

    ser.close()
    elapsed = time.time() - start_time
    print(f"[collector] Saved {sample_count} samples ({elapsed:.1f}s) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ESP32 IMU Serial Data Collector")
    parser.add_argument("--port", type=str, help="Serial port (e.g., COM3 or /dev/ttyUSB0)")
    parser.add_argument("--output", type=str, default="data/session.csv", help="Output CSV file path")
    parser.add_argument("--duration", type=float, default=None, help="Recording duration in seconds")
    parser.add_argument("--baud", type=int, default=BAUD_RATE, help=f"Baud rate (default: {BAUD_RATE})")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    args = parser.parse_args()

    if args.list_ports:
        list_ports()
        return

    if not args.port:
        print("ERROR: --port is required. Use --list-ports to find your ESP32's port.")
        parser.print_help()
        sys.exit(1)

    collect(args.port, args.output, args.duration, args.baud)


if __name__ == "__main__":
    main()
