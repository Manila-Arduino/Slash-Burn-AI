import numpy as np
from typing import List, Sequence
from classes.Arduino import Arduino
from classes.BoxedObject import BoxedObject

# from classes.OD_Custom import OD_Custom
from classes.Video import Video
from classes.Wrapper import Wrapper
from classes.rpi.Yolov11nSeg import YoloV11nSeg
from classes.rpi.Yolov8n import YoloV8n
from classes.rpi.rpi import RPI

MatLike = np.ndarray

# ? -------------------------------- CONSTANTS
cam_index = 0
img_width = 512
img_height = 512
input_layer_name = "input_layer_4"
output_layer_name = "output_0"

arduino_port = ""


# ? -------------------------------- CLASSES
rpi = RPI()
video = Video(cam_index, img_width, img_height)

# od_custom = OD_Custom(
#     "detect.tflite",
#     ["crop", "weed"],
#     0.9,
#     img_width=img_width,
#     img_height=img_height,
#     max_object_size_percent=0.80,
# )

yolov11n_seg = YoloV11nSeg(
    "yolov11n_seg_density.pt",
    ["Field", "Forest", "Lake"],
    0.60,
    img_width=img_width,
    img_height=img_height,
    max_object_size_percent=1.00,
)

# ? -------------------------------- VARIABLES


# ? -------------------------------- FUNCTIONS
def on_od_receive(max_object: BoxedObject, results: Sequence[BoxedObject]):
    # TODO 3 ------------------------------------------------
    pass


def on_yolov11n_seg_receive(max_object: BoxedObject, results: Sequence[BoxedObject]):
    # TODO 3 ------------------------------------------------
    pass


# ? -------------------------------- SETUP
def setup():
    pass


# ? -------------------------------- LOOP
def loop():
    #! VIDEO
    img = video.capture(display=False)

    #! AI 1 - FOREST FIRE [yolov11n]
    # TODO
    # img = od_custom.detect(img, on_od_receive=on_od_receive)

    #! AI 2 - FOREST DENSITY [yolov11n-seg]
    img = yolov11n_seg.detect(img, on_yolov11n_seg_receive=on_yolov11n_seg_receive)

    #! AI 3 - ILLEGAL LOGGING [sound]
    # TODO

    #! DISPLAY VIDEO
    video.displayImg(img)


# ? -------------------------------- ETC
setup()


def onExit():
    pass


Wrapper(
    loop,
    onExit=onExit,
    keyboardEvents=[
        # ["d", video.save_image],  # type: ignore
    ],
)
