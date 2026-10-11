import json
import time
import serial

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200

print(f"Connecting to ESP32 on {PORT}...")

try:
    with serial.Serial(PORT, BAUD, timeout=3) as esp:
        print("Connected. Waiting for sensor readings...")
        time.sleep(2)

        for _ in range(10):
            line = esp.readline().decode("utf-8", errors="replace").strip()

            if not line:
                print("No data received in this interval.")
                continue

            try:
                data = json.loads(line)
                print(data)
            except json.JSONDecodeError:
                print("Skipped non-JSON output:", line)

except serial.SerialException as exc:
    print("Serial connection error:", exc)
