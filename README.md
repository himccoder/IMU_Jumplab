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

> Run these commands from the **ESP-IDF PowerShell** (Start → search "ESP-IDF v5.3 PowerShell").

```powershell
cd C:\Users\himni\Documents\IMU_Jumplab-1\firmware

idf.py set-target esp32s3
idf.py build
idf.py -p COM6 flash
```

After `Hard resetting via RTS pin...` the ESP32 is live and streaming.  
You only need to flash once — the firmware persists on the chip.

---

## Data Collection

> Run these from your **normal** Python terminal (not the ESP-IDF one).  
> Replace `COM6` with your port.  Wear the sensor and perform each activity for the full 30 seconds.

```powershell
cd C:\Users\himni\Documents\IMU_Jumplab-1

# Stand completely still
python main.py collect --port COM6 --duration 30 --output data/raw/still.csv

# Walk around the room
python main.py collect --port COM6 --duration 30 --output data/raw/walk.csv

# Run / jog
python main.py collect --port COM6 --duration 30 --output data/raw/run.csv

# Jump repeatedly (~15 jumps, 2-3 s rest between each)
python main.py collect --port COM6 --duration 30 --output data/raw/jump.csv
```

---

## Train the Classifier

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
│   ├── classifier.py       # HistGradientBoosting, 44 features, train/evaluate/save
│   ├── realtime_classifier.py  # Serial reader, jump state machine, live inference
│   ├── signal_processor.py # Butterworth filter, magnitude, gravity calibration
│   ├── jump_detector.py    # Offline rule-based jump detection
│   ├── data_collector.py   # Serial → CSV recorder
│   ├── simulator.py        # Synthetic IMU data generator
│   └── visualizer.py       # Matplotlib plots and reports
├── dino_game/
│   └── game.py             # Pygame dino runner + Analyze screen
├── data/
│   ├── raw/                # Recorded CSVs (still / walk / run / jump)
│   ├── simulated/          # Synthetic data for pipeline testing
│   ├── reports/            # Confusion matrix + feature importance PNGs
│   └── model.joblib        # Trained classifier
├── main.py                 # CLI entry point (collect / train / play / process …)
└── requirements.txt
```

---

## CLI Reference

```powershell
# List serial ports
python main.py collect --list-ports

# Collect 30 s of data
python main.py collect --port COM6 --duration 30 --output data/raw/still.csv

# Train on real data
python main.py train --still data/raw/still.csv --walk data/raw/walk.csv \
                     --run data/raw/run.csv --jump data/raw/jump.csv

# Launch game (IMU)
python main.py play --port COM6

# Launch game (demo / keyboard)
python main.py play --demo

# Process a CSV and detect jumps
python main.py process --input data/raw/jump.csv --plot

# Generate simulated data (no hardware needed)
python main.py simulate

# Train on simulated data
python main.py classify
```
