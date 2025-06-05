import os
from typing import Callable, List, Sequence, Tuple

from classes.BoxedObject import BoxedObject

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import cv2
import numpy as np

MatLike = np.ndarray


class OD_Default:

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.model_path = "object_ssd_mb2.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.objects = [
            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
            "bus",
            "train",
            "truck",
            "boat",
            "trafficlight",
            "firehydrant",
            "streetsign",
            "stopsign",
            "parkingmeter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "hat",
            "backpack",
            "umbrella",
            "shoe",
            "eyeglasses",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sportsball",
            "kite",
            "baseballbat",
            "baseballglove",
            "skateboard",
            "surfboard",
            "tennisracket",
            "bottle",
            "plate",
            "wineglass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hotdog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "mirror",
            "diningtable",
            "window",
            "desk",
            "toilet",
            "door",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cellphone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "blender",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddybear",
            "hairdrier",
            "toothbrush",
            "hairbrush",
        ]

        self.float_input = self.input_details[0]["dtype"] == np.float32

    def detect(
        self,
        img: MatLike,
        on_od_receive: Callable[[BoxedObject, Sequence[BoxedObject]], None],
    ) -> MatLike:
        imH, imW, _ = img.shape
        image_resized = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_resized, (300, 300))
        image_resized = image_resized.reshape(
            1,
            image_resized.shape[0],
            image_resized.shape[1],
            image_resized.shape[2],
        )  # (1, 300, 300, 3)
        image_resized = image_resized.astype(np.uint8)

        # set input tensor
        self.interpreter.set_tensor(self.input_details[0]["index"], image_resized)

        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])

        results: List[BoxedObject] = []

        _ii = 0

        for i in range(boxes.shape[1]):
            label = self.objects[int(classes[0, i])]
            score = scores[0, i]
            if (score > self.threshold) and (score <= 1.0):
                _ii += 1
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                box = boxes[0, i, :]
                xmin = int(box[1] * imW)
                ymin = int(box[0] * imH)
                xmax = int(box[3] * imW)
                ymax = int(box[2] * imH)

                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

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
