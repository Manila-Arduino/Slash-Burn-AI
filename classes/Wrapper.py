from typing import Callable, List, Optional, Tuple
from pynput.keyboard import Listener, Key

from classes.Rich import Rich


class Wrapper:
    def __init__(
        self,
        loop: Callable,
        keyboardEvents: List[Tuple[str, Callable]] = [],
        onExit: Optional[Callable] = None,
        stop_key="q",
    ) -> None:
        self.loop = loop
        self.keyboardEvents = keyboardEvents
        self.onExit = onExit

        self.stop_key = stop_key

        self.looping = True

        self.listener = Listener(on_press=self.on_press)

        self.listener.start()

        self.start()

    def stop(self) -> None:
        self.looping = False

    def on_press(self, key):
        if hasattr(key, "char") and key.char == self.stop_key:
            self.stop()
            return

        for event in self.keyboardEvents:
            if hasattr(key, "char") and key.char == event[0]:
                event[1]()

    def start(self) -> None:
        while self.looping:
            self.loop()

        self.onExit()
        self.listener.stop()

        Rich.printm("Exiting...")
