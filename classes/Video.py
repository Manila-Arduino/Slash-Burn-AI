import os
from dataclasses import dataclass
import cv2
from cv2.typing import MatLike
from typing import Literal, Callable
import uuid
import threading
import queue
import time


@dataclass
class Video:
    cam_index: int = 0
    width: int = 256
    height: int = 256
    save_image_key: str = "d"

    def __post_init__(self):
        # If captures folder does not exist, create it
        if not os.path.exists("captures"):
            os.makedirs("captures")

        self.cap = cv2.VideoCapture(self.cam_index)
        self.frame = None
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")

        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass

            self.q.put(frame)

    def read(self):
        return self.q.get()

    def loop(self, func: Callable):
        while True:
            img = self.capture()

            key = cv2.waitKey(1) & 0xFF  # Read the key press once per loop

            func(img)

            self.display()

            if key == ord("q"):  # Exit key
                break

            if key == ord(self.save_image_key):  # Save image key
                self.save_image()

        self.release()
        print("Done Capturing!")

    def capture(self):
        frame = self.read()

        # Resize
        resized_frame = cv2.resize(frame, (self.width, self.height))

        self.frame = resized_frame
        return resized_frame

        # Overlay yellow circle

        # Display

    def circle(self, color: Literal["yellow", "green"]):
        if color == "yellow":
            color = (63, 201, 248)
        elif color == "green":
            color = (8, 240, 8)

        radius = 12
        x = self.width - radius - 10  # 10 pixels from the right edge
        y = radius + 10  # 10 pixels from the top edge
        cv2.circle(self.frame, (x, y), radius, color, -1)  # Yellow circle

    def display(self):
        cv2.imshow("Capture", self.frame)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def is_pressed(self, key: str) -> bool:
        return cv2.waitKey(1) & 0xFF == ord(key)

    def save_image(self, name: str = ""):
        print("Saving image...")
        if name == "":
            # name = f"{uuid.uuid4().hex}"
            name = str(int(time.time() * 1000))

        if self.frame is not None:
            filename = f"captures/{name}.jpg"
            cv2.imwrite(filename, self.frame)
            print(f"Image saved as {filename}")
        else:
            print("Error: No frame captured.")
