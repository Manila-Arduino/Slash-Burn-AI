import os
from typing import Callable, List, Sequence, Tuple

from classes.BoxedObject import BoxedObject

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import cv2
import numpy as np

MatLike = np.ndarray


class OD_Custom:

    def __init__(
        self,
        model_path: str,
        objects: List[str],
        threshold: float = 0.5,
        img_width: int = 512,
        img_height: int = 512,
        max_object_size_percent: float = 0.8,
    ) -> None:
        self.threshold = threshold
        self.objects = objects
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]["shape"][1]
        self.width = self.input_details[0]["shape"][2]

        self.float_input = self.input_details[0]["dtype"] == np.float32

        self.input_mean = 127.5
        self.input_std = 127.5

        self.img_width = img_width
        self.img_height = img_height
        self.img_area = self.img_width * self.img_height
        self.max_object_size_percent = max_object_size_percent

    def detect(
        self,
        img: MatLike,
        on_od_receive: Callable[[BoxedObject, Sequence[BoxedObject]], None],
    ) -> MatLike:
        imH, imW, _ = img.shape
        image_resized = cv2.resize(img, (self.width, self.height))

        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.float_input:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[3]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        results: List[BoxedObject] = []

        _ii = 0

        for i in range(len(scores)):
            label = self.objects[int(classes[i])]
            score = scores[i]
            if (score > self.threshold) and (score <= 1.0):
                _ii += 1
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()

                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                area = (xmax - xmin) * (ymax - ymin)
                area_percent = area / self.img_area
                # print(f"Area Percent: {area_percent:.2f} for {label}")

                if area_percent > self.max_object_size_percent:
                    continue

                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                # print((xmin, ymin), (xmax, ymax))

                # Draw label
                fontScale = 0.5
                fontThickness = 2
                object_name = label
                label = "%s: %d%%" % (
                    object_name,
                    int(score * 100),
                )  # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness
                )  # Get font size
                label_ymin = max(
                    ymin, labelSize[1] + 10
                )  # Make sure not to draw label too close to top of window
                cv2.rectangle(
                    img,
                    (xmin, label_ymin - labelSize[1] - 10),
                    (xmin + labelSize[0], label_ymin + baseLine - 10),
                    (255, 255, 255),
                    cv2.FILLED,
                )  # Draw white box to put label text in
                cv2.putText(
                    img,
                    label,
                    (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale,
                    (0, 0, 0),
                    fontThickness,
                )  # Draw label text

                results.append(
                    BoxedObject(object_name, score, (xmin, ymin, xmax, ymax))
                )

        results = sorted(results, key=lambda x: x.score, reverse=True)

        if len(results) > 0:
            max_area_obj = max(
                results,
                key=lambda obj: (obj.boxes[2] - obj.boxes[0])
                * (obj.boxes[3] - obj.boxes[1]),
                default=None,
            )

            if max_area_obj:
                on_od_receive(max_area_obj, results)

        return img
