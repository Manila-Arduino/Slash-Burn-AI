from dataclasses import dataclass
import serial


@dataclass
class Arduino:
    port: str
    baudrate: int = 9600

    def __post_init__(self):
        self.use_arduino = self.port is not None and self.port != ""
        if self.use_arduino:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)

    def print(self, data: str):
        if self.use_arduino:
            self.ser.write(data.encode())

    def println(self, data: str):
        if self.use_arduino:
            self.ser.write((data + "\n").encode())

    def available(self):
        if not self.use_arduino:
            return False

        return self.ser.in_waiting

    def read(self):
        if not self.use_arduino:
            return ""
        return self.ser.readline().decode().strip()

    def close(self):
        if self.use_arduino:
            self.ser.close()
