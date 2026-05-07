# IMU Activity Classifier

Classify human activity — **still, walk, run, jump** — from short 3-second IMU recordings using an ESP32-S3 + MPU6050 sensor and a machine-learning classifier.

---

## Hardware

| MPU6050 Pin | ESP32-S3 Pin |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | GPIO 18 |
| SCL | GPIO 19 |
| AD0 | GND |

Wear the sensor on your **waist / hip**.

---

## Setup

```powershell
pip install -r requirements.txt
```

---

## Firmware (flash once)

```powershell
. C:\Espressif\frameworks\esp-idf-v5.3.1\export.ps1

cd C:\Users\himni\Documents\IMU_Jumplab-1\firmware
idf.py set-target esp32s3
idf.py build
idf.py -p COM6 flash
```

---

## Workflow

### 1. Find your COM port
```powershell
python main.py ports
```

### 2. Record clips (3 seconds each)
```powershell
python main.py record --port COM6 --activity still
python main.py record --port COM6 --activity walk
python main.py record --port COM6 --activity run
python main.py record --port COM6 --activity jump
```
Each command prompts you before each clip and asks if you want another.
Aim for **at least 3 clips per activity**.

### 3. Check clip counts
```powershell
python main.py status
```

### 4. Train the classifier
```powershell
python main.py train
```

### 5. Launch the GUI
```powershell
python app.py
```

- Select your COM port from the dropdown
- Click **Record & Classify**
- A 3-second countdown plays, then it records and classifies automatically
- Results are shown as a bar chart with the predicted activity highlighted

---

## CLI classifier (no GUI)
```powershell
python main.py classify --port COM6
```

---

## Project Structure

```
IMU_Jumplab-1/
├── app.py                  GUI interface (record → classify)
├── main.py                 CLI: ports / status / record / train / classify
├── requirements.txt
├── README.md
├── src/
│   ├── classifier.py       44-feature extractor + HistGradientBoosting
│   ├── signal_processor.py Butterworth filter, magnitude, gravity calibration
│   └── data_collector.py   Serial -> CSV recorder
├── firmware/
│   └── main/main.c         ESP32-S3 ESP-IDF firmware
└── data/
    ├── raw/
    │   ├── still/          still1.csv, still2.csv, ...
    │   ├── walk/           walk1.csv,  walk2.csv,  ...
    │   ├── run/            run1.csv,   run2.csv,   ...
    │   └── jump/           jump1.csv,  jump2.csv,  ...
    ├── reports/            confusion_matrix.png, feature_importances.png
    └── model.joblib        trained classifier
```

---

## CLI Reference

```powershell
python main.py ports                                    # list serial ports
python main.py status                                   # show clip counts
python main.py record --port COM6 --activity still      # record clips
python main.py record --port COM6 --activity walk --once  # record exactly one
python main.py train                                    # train classifier
python main.py classify --port COM6                     # CLI classify loop
python app.py                                           # GUI
```
