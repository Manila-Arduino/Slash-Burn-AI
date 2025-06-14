from subprocess import call
from typing import Callable, Optional
from gpiozero import Button
from time import sleep
from datetime import datetime


def _usage():
    #! SHUTDOWN BUTTON
    shutdown_button = ShutdownButton(16, 2)


class ShutdownButton:

    def __init__(
        self, pin: int, press_interval_s: int = 2, callback: Optional[Callable] = None
    ):
        self.pin = pin
        self.button = Button(self.pin)
        self.callback = callback
        self.button.when_pressed = self.shutdown
        self.last_pressed = datetime.now()
        self.press_interval_s = press_interval_s

        # self.led = LED(self.pinLED)
        # self.led.on()

    def shutdown(self):
        if (datetime.now() - self.last_pressed).total_seconds() < self.press_interval_s:
            return

        print(f"self.callback: {self.callback}")
        if self.callback:
            self.callback()
        print("Shutdown button pressed")
        call("sudo shutdown -h now", shell=True)
