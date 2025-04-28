from typing import List
from classes.ObjectDetection import MobilenetV2Result, ObjectDetection
from cv2.typing import MatLike
from classes.Arduino import Arduino
from classes.CNNImage import CNNImage
from classes.Video import Video
from classes.Wrapper import Wrapper

# ? -------------------------------- CONSTANTS
cam_index = 0
img_width = 512
img_height = 512
input_layer_name = "input_layer_4"
output_layer_name = "output_0"

arduino_port = ""


# ? -------------------------------- CLASSES
arduino = Arduino(arduino_port)
video = Video(cam_index, img_width, img_height)
cnn = CNNImage(
    ["no_oil", "oil"],
    r"model.tflite",
    img_width,
    img_height,
    input_layer_name=input_layer_name,
    output_layer_name=output_layer_name,
)

od = ObjectDetection(
    "ssd_mobilenet_v2_fpnlite",
    ["crop", "weed"],
    f"detect.tflite",
    print_entities=False,
    threshold=0.99,
    # threshold=0.99999,
)

# ? -------------------------------- VARIABLES


# ? -------------------------------- FUNCTIONS
def on_cnn_predict(predicted_class: str, confidence: float):
    # TODO 2 -------------------------------------------------
    pass


def on_od_receive(results: List[MobilenetV2Result]):
    # TODO 3 ------------------------------------------------
    pass


def on_arduino_receive(s: str):
    # TODO 3 ------------------------------------------------
    pass


# ? -------------------------------- SETUP
def setup():
    pass


# ? -------------------------------- LOOP
def loop():
    #! VIDEO
    img = video.capture(display=False)

    #! CNN
    # predicted_class, confidence = cnn.predict(img, isBatch=False)
    # print(predicted_class, confidence)
    # on_cnn_predict(predicted_class, confidence)

    #! OBJECT DETECTION
    img, results = od.detect_objects(img)
    on_od_receive(results)

    #! DISPLAY VIDEO
    video.displayImg(img)

    #! ARDUINO
    if arduino.available():
        arduino_str = arduino.read()
        print(f"Arduino received: {arduino_str}")
        on_arduino_receive(arduino_str)


# ? -------------------------------- ETC
setup()


def onExit():
    arduino.close()


Wrapper(
    loop,
    onExit=onExit,
    keyboardEvents=[
        ["d", video.save_image],  # type: ignore
    ],
)
