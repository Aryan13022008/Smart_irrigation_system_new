import json
import time
import threading
import serial

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200


class ESP32Reader:
    def __init__(self):
        self.latest = None
        self.error = None
        self.lock = threading.Lock()
        self.serial = None
        self.thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
        )
        self.thread.start()

    def _read_loop(self):
        while True:
            try:
                if self.serial is None or not self.serial.is_open:
                    self.serial = serial.Serial(
                        PORT, BAUD, timeout=2
                    )
                    time.sleep(2)
                    self.serial.reset_input_buffer()

                line = self.serial.readline().decode(
                    "utf-8", errors="replace"
                ).strip()

                if not line:
                    continue

                data = json.loads(line)

                with self.lock:
                    self.latest = data
                    self.error = None

            except (serial.SerialException, OSError) as exc:
                with self.lock:
                    self.error = str(exc)
                if self.serial is not None:
                    try:
                        self.serial.close()
                    except Exception:
                        pass
                    self.serial = None
                time.sleep(2)

            except (json.JSONDecodeError, ValueError):
                continue

    def get_data(self):
        with self.lock:
            if self.latest is None:
                return None, self.error
            return dict(self.latest), self.error


@__import__("streamlit").cache_resource
def get_esp32_reader():
    return ESP32Reader()
