# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEMPLATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from dataclasses import dataclass
import tensorflow as tf
from PIL import Image
import numpy as np
from typing import List, Tuple
from keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
import cv2


@dataclass
class CNNImage:
    class_names: List[str]
    tflite_filepath: str
    img_height: int
    img_width: int
    input_layer_name: str = "input_1"
    output_layer_name: str = "dense_2"
    threshold_for_binary: float = 0.6

    def __post_init__(self):
        self.is_binary = False
        self.interpreter = tf.lite.Interpreter(model_path=self.tflite_filepath)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.classify_lite = self.interpreter.get_signature_runner("serving_default")

    # Load Image
    def load_img(self, img_path: str):
        img = Image.open(img_path)
        img = img.convert("RGB")

        if img.width != self.img_width or img.height != self.img_height:
            img = img.resize(
                (self.img_width, self.img_height), Image.Resampling.LANCZOS
            )

        img = np.array(img)

        # img = np.array([img])
        return img

    # Predict
    def predict(self, img, isBatch=False) -> Tuple[str, float]:
        # print(f"SHAPE: {img.shape}")
        # print(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).astype(np.float32)
        # img = preprocess_input(img)
        if not isBatch:
            img = np.expand_dims(img, axis=0)

        predictions = self.classify_lite(**{self.input_layer_name: img})[
            self.output_layer_name
        ]  # TODO ---
        # predictions = self.classify_lite(input_1=img)["dense_2"]
        # print(predictions)

        if not isBatch:
            score_lite = tf.nn.softmax(predictions)
            # print(class_names[np.argmax(score_lite)])
            predicted_class = self.class_names[np.argmax(score_lite)]
            confidence = 100 * np.max(score_lite)

        else:
            score_lite = tf.nn.softmax(predictions, axis=1)
            predicted_class = np.argmax(score_lite, axis=1)
            confidence = 100 * np.max(score_lite, axis=1)

        return predicted_class, confidence

    # Predict from image path
    def predict_path(self, img_path) -> Tuple[str, float]:
        img = self.load_img(img_path)
        # print(self.class_names)
        return self.predict(img)
