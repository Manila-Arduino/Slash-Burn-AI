from cv2.typing import MatLike
from classes.Arduino import Arduino
from classes.CNNImage import CNNImage
from classes.Video import Video

# ? ------------------------ CONFIG
cam_index = 0
img_width = 512
img_height = 512
arduino_port = "COM3"
save_image_key = "d"
classes = ["no_oil", "oil"]
# ? ------------------------ CONFIG


arduino = Arduino(arduino_port)
cnn = CNNImage(
    classes,
    r"model.tflite",
    img_width,
    img_height,
)


def on_predict(predicted_class: str, confidence: float):
    # TODO 2 -------------------------------------------------
    pass


def on_arduino_receive(s: str):
    # TODO 3 ------------------------------------------------
    pass


#! LOOP
video = Video(cam_index, img_width, img_height)


def loop(img: MatLike):
    if video.is_pressed(save_image_key):
        video.save_image()

    predicted_class, confidence = cnn.predict(img, isBatch=False)
    print(predicted_class, confidence)
    on_predict(predicted_class, confidence)

    if arduino.available():
        arduino_str = arduino.read()
        print(f"Arduino received: {s}")
        on_arduino_receive(arduino_str)


video.loop(loop)
