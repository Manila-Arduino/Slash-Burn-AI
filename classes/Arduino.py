from dataclasses import dataclass
import serial


@dataclass
class Arduino:
    port: str
    baudrate: int = 9600

    def __post_init__(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)

    def print(self, data: str):
        self.ser.write(data.encode())

    def println(self, data: str):
        self.ser.write((data + "\n").encode())

    def available(self):
        return self.ser.in_waiting

    def read(self):
        return self.ser.readline().decode().strip()

    def close(self):
        self.ser.close()
