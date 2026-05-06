# IMU Dino Jump

Classify human activity (still, walk, run, jump) in real-time using an **ESP32-S3 + MPU6050** IMU, and control a Chrome-dino-style game with your body.  Jump in real life → dino jumps in the game, with your actual jump height displayed on screen.

---

## Hardware

| Component | Connection |
|---|---|
| MPU6050 VCC | ESP32-S3 3.3V |
| MPU6050 GND | ESP32-S3 GND |
| MPU6050 SDA | ESP32-S3 **GPIO 18** |
| MPU6050 SCL | ESP32-S3 **GPIO 19** |
| MPU6050 AD0 | GND (I²C address 0x68) |

Wear the sensor on your **waist / hip** for best results.

---

## Setup

### 1 — Python dependencies

```powershell
pip install -r requirements.txt
```

### 2 — Find your COM port

```powershell
python main.py collect --list-ports
```

---

## Firmware — Build & Flash

> Run these commands from a **PowerShell terminal**.  
> First, activate the ESP-IDF environment:

```powershell
. C:\Espressif\frameworks\esp-idf-v5.3.1\export.ps1
```

Then build and flash:

```powershell
cd C:\Users\himni\Documents\IMU_Jumplab-1\firmware

idf.py set-target esp32s3
idf.py build
idf.py -p COM6 flash
```

After `Hard resetting via RTS pin...` the ESP32 is live and streaming.  
You only need to flash once — the firmware persists on the chip.

---

## Quickstart — Guided Data Collection & Training

> **This is the recommended workflow.**  
> Plug in the ESP32, then run a single command and follow the on-screen prompts.

```powershell
cd C:\Users\himni\Documents\IMU_Jumplab-1

python main.py guided --port COM6
```

The tool walks you through **4 phases of 5 seconds each** — press **Enter** before each one:

| Phase | Activity | What to do |
|---|---|---|
| 1 | **still** | Stand completely still |
| 2 | **walk** | Walk at a normal pace |
| 3 | **run** | Run or jog |
| 4 | **jump** | Jump once, then hold still after landing |

After all phases are recorded the classifier is **trained automatically** and the model is saved to `data/model.joblib`.  A confusion matrix and feature importance chart open on screen and are saved to `data/reports/`.

> You can use `--duration 10` to record 10 seconds per phase instead of 5.

---

## Manual Data Collection (optional)

> Replace `COM6` with your port.  Wear the sensor and perform each activity for the full duration.

```powershell
cd C:\Users\himni\Documents\IMU_Jumplab-1

# Stand completely still
python main.py collect --port COM6 --duration 5 --output data/raw/still.csv

# Walk around the room
python main.py collect --port COM6 --duration 5 --output data/raw/walk.csv

# Run / jog
python main.py collect --port COM6 --duration 5 --output data/raw/run.csv

# Jump once
python main.py collect --port COM6 --duration 5 --output data/raw/jump.csv
```

---

## Train the Classifier (manual)

```powershell
python main.py train \
  --still data/raw/still.csv \
  --walk  data/raw/walk.csv  \
  --run   data/raw/run.csv   \
  --jump  data/raw/jump.csv
```

This trains a **HistGradientBoostingClassifier** on 44 tri-axial features (500 ms window, 50 ms step) and saves the model to `data/model.joblib`.  Cross-validation F1-macro and a confusion matrix are printed/saved.

---

## Run the Game

### Live IMU mode (requires trained model + ESP32)

```powershell
python main.py play --port COM6
```

### Keyboard demo mode (no hardware needed)

```powershell
python main.py play --demo
```

| Key | Action |
|---|---|
| `Space` | Jump (demo mode) |
| `TAB` | Open Analyze screen (record 5 s → classify) |
| `R` | Restart after game over |
| `ESC` | Quit |

---

## Analyze Screen

Press **TAB** at any time to open the activity analyser:

1. A 3-second countdown plays
2. The sensor records 5 seconds of your movement
3. A colour-coded bar chart shows the fraction of time spent in each activity

Press **TAB** or **Space** to return to the game.

---

## Pipeline Overview

```
ESP32-S3 (MPU6050)
    │  USB Serial · 115200 baud · 100 Hz
    ▼
src/realtime_classifier.py
    ├── sliding window (500 ms, 50 ms step)
    ├── 44 features from ax / ay / az / magnitude
    ├── HistGradientBoostingClassifier → activity label (every 0.4 s)
    └── rule-based jump state machine → height from air-time
         │  Queue
         ▼
dino_game/game.py  (Pygame)
    ├── dino jumps with IMU-proportional arc
    ├── HUD: activity label + last jump height (cm)
    ├── TAB: Analyze screen with classification bar chart
    └── scrolling cactus obstacles
```

---

## Project Structure

```
IMU_Jumplab-1/
├── firmware/
│   ├── main/
│   │   ├── main.c          # ESP-IDF firmware (ESP32-S3, new I2C API)
│   │   └── CMakeLists.txt
│   └── CMakeLists.txt
├── src/
│   ├── classifier.py           # HistGradientBoosting, 44 features, train/evaluate/save
│   ├── realtime_classifier.py  # Serial reader, jump state machine, live inference
│   ├── signal_processor.py     # Butterworth filter, magnitude, gravity calibration
│   ├── jump_detector.py        # Offline rule-based jump detection
│   ├── data_collector.py       # Serial → CSV recorder (collect / collect_labelled)
│   ├── simulator.py            # Synthetic IMU data generator
│   └── visualizer.py           # Matplotlib plots and reports
├── dino_game/
│   └── game.py             # Pygame dino runner + Analyze screen
├── data/
│   ├── raw/                # Recorded CSVs (still / walk / run / jump)
│   ├── simulated/          # Synthetic data for pipeline testing
│   ├── reports/            # Confusion matrix + feature importance PNGs
│   └── model.joblib        # Trained classifier
├── main.py                 # CLI entry point (guided / collect / train / play / …)
└── requirements.txt
```

---

## CLI Reference

```powershell
# ── Environment (run once per terminal session before idf.py) ─────────────
. C:\Espressif\frameworks\esp-idf-v5.3.1\export.ps1

# ── Guided collection + auto-train (recommended) ──────────────────────────
python main.py guided --port COM6
python main.py guided --port COM6 --duration 10   # 10 s per activity

# ── Manual collection ─────────────────────────────────────────────────────
python main.py collect --list-ports
python main.py collect --port COM6 --duration 5 --output data/raw/still.csv

# ── Train on manually collected data ──────────────────────────────────────
python main.py train --still data/raw/still.csv --walk data/raw/walk.csv \
                     --run  data/raw/run.csv  --jump data/raw/jump.csv

# ── Launch game ───────────────────────────────────────────────────────────
python main.py play --port COM6        # live IMU
python main.py play --demo             # keyboard only

# ── Signal processing & jump detection ───────────────────────────────────
python main.py process --input data/raw/jump.csv --plot

# ── Simulated pipeline test (no hardware needed) ──────────────────────────
python main.py simulate
python main.py classify
```
