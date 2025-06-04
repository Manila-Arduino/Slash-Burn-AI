import os
from dataclasses import dataclass
import cv2
from typing import Any, Literal, Callable
import uuid
import threading
import queue
import time
import numpy as np

MatLike = np.ndarray


@dataclass
class Video:
    cam_index: int = 0
    width: int = 256
    height: int = 256
    window_name: str = "Capture"
    with_window: bool = True
    full_screen: bool = False
    apiPreference: int = cv2.CAP_DSHOW

    def __post_init__(self):
        # If captures folder does not exist, create it
        if not os.path.exists("captures"):
            os.makedirs("captures")

        self.cap = cv2.VideoCapture(self.cam_index, self.apiPreference)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 280)
        self.cap.set(cv2.CAP_PROP_FPS, 5)

        self.frame = None
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")

        # for custom buttons
        self.buttons: list[tuple[int, int, int, int, Callable[[], None]]] = []
        if self.with_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            print(f"{self.window_name} window created")
            cv2.setMouseCallback(self.window_name, self._on_mouse)

            if self.full_screen:
                cv2.setWindowProperty(
                    self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )

        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def change_cam_index(self, cam_index: int):
        if self.cap.isOpened():
            self.cap.release()
        self.cam_index = cam_index
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")
        self.frame = None

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for x1, y1, x2, y2, callback in self.buttons:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    callback()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                # break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass

            self.q.put(frame)

    def read(self):
        return self.q.get()

    def capture(self, display: bool) -> MatLike:
        img = self._capture()
        cv2.waitKey(1)
        if display:
            self.displayImg(img)
        return img

    # def release(self):
    #     self.release()

    def _capture(self):
        frame = self.read()

        # Resize
        resized_frame = cv2.resize(frame, (self.width, self.height))

        self.frame = resized_frame
        return resized_frame

        # Overlay yellow circle

        # Display

    def circle(self, color: Any):
        if color == "yellow":
            color = (63, 201, 248)
        elif color == "green":
            color = (8, 240, 8)

        radius = 12
        x = self.width - radius - 10  # 10 pixels from the right edge
        y = radius + 10  # 10 pixels from the top edge
        cv2.circle(self.frame, (x, y), radius, color, -1)  # type: ignore

    def displayImg(self, img: MatLike):
        if not self.with_window:
            return
        cv2.imshow(self.window_name, img)  # type: ignore

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

    def text(
        self,
        img: MatLike,
        text: str,
        pos: tuple[int, int] = (0, 0),
        scale: float = 1.0,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> MatLike:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img,
            text,
            pos,
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return img

    def resize(self, img: MatLike, size: tuple[int, int]) -> MatLike:
        return cv2.resize(img, size)

    def padding(
        self,
        img: MatLike,
        top: int = 0,
        bottom: int = 0,
        left: int = 0,
        right: int = 0,
        color: tuple[int, int, int] = (0, 0, 0),
    ) -> MatLike:
        return cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

    def button(
        self,
        img: MatLike,
        text: str,
        pos: tuple[int, int] = (0, 0),
        size: tuple[int, int] = (200, 50),  # new: (width, height)
        scale: float = 1.0,
        thickness: int = 2,
        color: tuple[int, int, int] = (0, 255, 0),
        bg_color: tuple[int, int, int] = (0, 100, 0),
        on_click: Callable[[], None] | None = None,
    ) -> MatLike:
        x, y = pos
        w, h = size
        cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
        cv2.putText(
            img,
            text,
            (x + 10, y + h // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        if on_click:
            # remove old region if same
            self.buttons = [b for b in self.buttons if b[:4] != (x, y, x + w, y + h)]
            self.buttons.append((x, y, x + w, y + h, on_click))

        return img
