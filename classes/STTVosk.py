import json
import signal
import threading
from typing import Callable, Dict
from vosk import Model, KaldiRecognizer
import pyaudio
import os
from datetime import datetime
import sys


def get_model_path():
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, "vosk-model-small-en-us-0.15")
    else:
        return "vosk-model-small-en-us-0.15"


class STTVosk:
    """Add while loop in the parent thread!!

    Example:
    `STTVosk({"hello": hello, "bye": bye})`"""

    _stop_event = threading.Event()
    _thread = None
    audio = None
    last_executed = datetime.now()
    commands: Dict[str, Callable[[str], None]] = {}

    def __init__(self, commands: Dict[str, Callable[[str], None]]):
        model_path = get_model_path()
        self.model = Model(model_path)
        self.commands = commands
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self._stopping = False
        self._start()
        self._setup_signal_handler()

    def _initialize_audio(self):
        if self.audio is not None:
            self.audio.terminate()
        self.audio = pyaudio.PyAudio()

    def _start_thread(self):
        self._initialize_audio()
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            # input_device_index=40,
            frames_per_buffer=4096,
        )
        stream.start_stream()
        while not self._stop_event.is_set():
            data = stream.read(4096, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                result = self.recognizer.Result()
                text = json.loads(result).get("text", "")
                print(f"You said: {text}")
                self._check_text(text)

        stream.stop_stream()
        stream.close()

    def _start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._start_thread)
            self._thread.start()

    def stop(self):
        if not self._stopping:
            self._stopping = True
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join()
            self.audio.terminate()

    def _setup_signal_handler(self):
        def signal_handler(sig, frame):
            print("Ctrl+C detected, stopping...")
            try:
                self.stop()
            except Exception as e:
                print(f"Error stopping: {e}")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

    def _check_text(self, text: str):
        text = text.lower()

        if (datetime.now() - self.last_executed).total_seconds() < 1:
            return

        if self.commands.get(text) is not None:
            self.commands.get(text)(text)
            self.last_executed = datetime.now()
