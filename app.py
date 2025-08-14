USE_CAMERA = True

from datetime import datetime
import os
import socket
import subprocess
from time import sleep
import numpy as np
from typing import List, Sequence, Union

from pydantic import BaseModel
import rich
from classes.ClassificationObject import ClassificationObject
from classes.BoxedObject import BoxedObject

# from classes.OD_Custom import OD_Custom
from classes.Rich import Rich
from classes.SegmentedObject import SegmentedObject
from classes.Video import Video
from classes.Wrapper import Wrapper
from classes.Yolov11nCls import Yolov11nCls
from classes.firebase_helper import Firebase
from classes.Yolov11nSeg import YoloV11nSeg
from classes.rpi.Yolov8n import YoloV8n
from classes.rpi.rpi import RPI
from decorators.execute_interval import execute_interval

MatLike = np.ndarray


#! START MONA SERVER
# subprocess.Popen('start cmd /k "MonaServer_Win64\\MonaServer.exe"', shell=True)
# sleep(1)

# ? -------------------------------- CONSTANTS
cam_index = 0
img_width = 512
img_height = 512
FOREST_FIRE_CONFIDENCE_THRESHOLD = 0.86
FOREST_FIRE_AREA_THRESHOLD = 0.05
DENSITY_CONFIDENCE_THRESHOLD = 0.10
ILLEGAL_LOGGING_CONFIDENCE_THRESHOLD = 0.10
LOGGING_CONFIDENCE_THRESHOLD = 0.10
LOGGING_AREA_THRESHOLD = 0.05


# ? -------------------------------- CLASSES
rpi = RPI()


def get_ip_default_route() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets actually sent
        return s.getsockname()[0]
    finally:
        s.close()


if USE_CAMERA:
    video = Video(
        cam_index,
        img_width,
        img_height,
    )
else:
    video = Video(
        cam_index,
        img_width,
        img_height,
        rtmp_url=f"rtmp://{get_ip_default_route()}/live/key",
    )

firebase = Firebase(
    "credentials.json",
    use_storage=False,
    use_firestore=True,
)


class Logs(BaseModel):
    id: str
    forest_fire: bool
    forest_density: float
    illegal_logging: bool


yolov11n_seg_fire = YoloV11nSeg(
    "yolov11n_seg_fire.pt",
    [("Fire", (0, 0, 255)), ("Fire", (0, 0, 255))],
    threshold=FOREST_FIRE_CONFIDENCE_THRESHOLD,
    img_width=img_width,
    img_height=img_height,
    max_object_size_percent=1.00,
    allowed=["Fire"],
)

yolov11n_seg_density = YoloV11nSeg(
    "yolov11n_seg_density.pt",
    [
        ("Building", (255, 140, 0)),
        ("Field", (0, 255, 0)),
        ("Forest", (255, 0, 0)),
        ("Lake", (0, 0, 255)),
        ("Road", (128, 128, 128)),
    ],
    threshold=DENSITY_CONFIDENCE_THRESHOLD,
    img_width=img_width,
    img_height=img_height,
    max_object_size_percent=1.00,
    allowed=["Forest"],
)

yolov11n_seg_logging = YoloV11nSeg(
    "yolov11n_seg_logging.pt",
    [
        ("Circle", (255, 140, 0)),
        ("Log", (255, 0, 0)),
    ],
    threshold=LOGGING_CONFIDENCE_THRESHOLD,
    img_width=img_width,
    img_height=img_height,
    max_object_size_percent=0.9,
    allowed=["Log"],
)

# yolov11n_cls_logging = Yolov11nCls(
#     "yolov11n_cls_logging.pt",
#     ["Illegal Logging", "Legal Logging"],
#     threshold=ILLEGAL_LOGGING_CONFIDENCE_THRESHOLD,
#     img_width=img_width,
#     img_height=img_height,
# )


# ? -------------------------------- VARIABLES
forest_fire = False
density = 0.0
illegal_logging = False


# ? -------------------------------- FUNCTIONS
def on_yolov11n_seg_fire_receive(
    max_object: SegmentedObject, results: Sequence[SegmentedObject]
):
    global forest_fire
    total_fire_area = min(sum(obj.area_percent for obj in results), 1.0)
    print(f"Total fire area: {total_fire_area:.2f}")
    forest_fire = total_fire_area >= FOREST_FIRE_AREA_THRESHOLD


def on_yolov11n_seg_density_receive(
    max_object: SegmentedObject, results: Sequence[SegmentedObject]
):
    global density
    density = min(sum(obj.area_percent for obj in results), 1.0)


def on_yolov11n_cls_logging_receive(classification: Union[ClassificationObject, None]):
    global illegal_logging
    illegal_logging = (
        classification is not None and classification.entity == "Illegal Logging"
    )


def on_yolov11n_seg_logging_receive(
    max_object: SegmentedObject, results: Sequence[SegmentedObject]
):
    global illegal_logging
    illegal_logging = (
        min(sum(obj.area_percent for obj in results), 1.0) >= LOGGING_AREA_THRESHOLD
    )


@execute_interval(10)
def firebase_upload():
    global forest_fire, density, illegal_logging

    current_time_iso = datetime.now().astimezone().isoformat()

    logs = Logs(
        id=current_time_iso,
        forest_fire=forest_fire,
        forest_density=round(density, 2),
        illegal_logging=illegal_logging,
    )

    rich.print(logs)

    firebase.write_firestore(
        f"logs/{current_time_iso}",
        logs,
    )


# ? -------------------------------- SETUP
def setup():
    pass


# ? -------------------------------- LOOP
def loop():
    #! VIDEO
    img = video.capture(display=False)

    #! AI 1 - FOREST FIRE [yolov11n-seg]
    yolov11n_seg_fire.detect(img, on_yolov11n_seg_fire_receive)

    #! AI 2 - FOREST DENSITY [yolov11n-seg]
    yolov11n_seg_density.detect(img, on_yolov11n_seg_density_receive)

    #! AI 3 - ILLEGAL LOGGING [yolov11n-cls]
    yolov11n_seg_logging.detect(img, on_yolov11n_seg_logging_receive)
    # yolov11n_cls_logging.detect(img, on_yolov11n_cls_logging_receive)

    #! DISPLAY VIDEO
    img = yolov11n_seg_fire.display(img)
    img = yolov11n_seg_density.display(img)
    img = yolov11n_seg_logging.display(img)
    # img = yolov11n_cls_logging.display(img)
    video.displayImg(img)

    #! UPLOAD TO FIREBASE
    # firebase_upload()


# ? -------------------------------- ETC
setup()


def onExit():
    video.release()


Wrapper(
    loop,
    onExit=onExit,
    keyboardEvents=[
        # ["d", video.save_image],  # type: ignore
    ],
)
