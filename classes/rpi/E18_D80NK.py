from typing import Callable, Optional
from gpiozero import Button
from datetime import datetime


class E18_D80NK:
    def __init__(self, pin: int):
        self.pin = pin
        self.button = Button(self.pin)
        self.button.when_pressed = self.on_clicked
        self.last_pressed = datetime.now()

    def setup(self, activate_s: int = 2, callback: Optional[Callable] = None):
        self.callback = callback
        self.activate_s = activate_s

    def on_clicked(self):
        if (datetime.now() - self.last_pressed).total_seconds() < self.activate_s:
            return

        if self.callback:
            self.callback()

        print("E18_D80NK has detected an object")
        self.last_pressed = datetime.now()
