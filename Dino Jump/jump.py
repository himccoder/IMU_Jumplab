import time
import serial
import pyautogui

# Change this to your serial port
PORT = "/dev/cu.usbserial-110"   # macOS example
# PORT = "COM3"                  # Windows example
# PORT = "/dev/ttyUSB0"          # Linux example

BAUD = 115200
JUMP_COOLDOWN = 0.25  # seconds, prevents accidental double jumps

pyautogui.FAILSAFE = True

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # let serial settle after opening

    print(f"Listening on {PORT} at {BAUD} baud...")
    print("Focus the Chrome Dino window. Press Ctrl+C to quit.")

    last_jump_time = 0.0

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()

            if not line:
                continue

            print(f"RX: {line}")

            if line == "JUMP":
                now = time.time()
                if now - last_jump_time >= JUMP_COOLDOWN:
                    pyautogui.press("space")
                    print("Jump sent")
                    last_jump_time = now

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        ser.close()

if __name__ == "__main__":
    main()