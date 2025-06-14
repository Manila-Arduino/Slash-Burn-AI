import platform


class RPI:
    def __init__(self, shutdown_button_pin: int = 16) -> None:
        if platform.system() == "Windows":
            return

        from classes.rpi.ShutdownButton import ShutdownButton
        from gpiozero.pins.lgpio import LGPIOFactory
        from gpiozero import Device, PWMLED

        Device.pin_factory = LGPIOFactory(chip=4)
        self.shutdown_button = ShutdownButton(shutdown_button_pin)
