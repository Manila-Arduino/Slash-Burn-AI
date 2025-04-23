# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEMPLATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#! ObjectDetection
from math import e
import os

import cv2

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from dataclasses import dataclass
import tensorflow as tf
from PIL import Image
import numpy as np
from typing import Any, Optional, List, Tuple


@dataclass
class MobilenetV2Result:
    entity: str
    score: float
    boxes: Tuple[float, float, float, float]


@dataclass
class ObjectDetection:
    model_type: str
    objects: List[str]
    model_path: Optional[str] = None
    print_entities: bool = False
    threshold: float = 0.95

    def __post_init__(self):
        if self.model_type == "openimages_v4_ssd_mobilenet_v2":
            if self.model_path is None:
                raise Exception(
                    "model_path must not be None for openimages_v4_ssd_mobilenet_v2"
                )

            self.model = tf.saved_model.load(self.model_path)

        elif self.model_type == "ssd_mobilenet_v2_fpnlite":
            if self.model_path is None:
                raise Exception(
                    "model_path must not be None for ssd_mobilenet_v2_fpnlite"
                )

            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.height = self.input_details[0]["shape"][1]
            self.width = self.input_details[0]["shape"][2]

            self.float_input = self.input_details[0]["dtype"] == np.float32

            self.input_mean = 127.5
            self.input_std = 127.5

            # self.input_details = self.interpreter.get_input_details()[0]["index"]
            # self.output_details = self.interpreter.get_output_details()

        elif self.model_type == "custom":
            # TODO
            raise Exception("custom model_type not implemented")

        else:
            raise Exception(f"invalid model_type: {self.model_type}")

    # Load Image
    def load_img(self, img_path: str):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)  # type: ignore
        return img

    # Run Inference
    def _detect_objects_openimages_v4_ssd_mobilenet_v2(
        self, img
    ) -> Tuple[Any, List[MobilenetV2Result]]:
        converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]  # type: ignore
        detections = self.model.signatures["default"](converted_img)  # type: ignore
        results: List[MobilenetV2Result] = []

        found_entities = set()

        for i in range(len(detections["detection_class_entities"])):
            entity = detections["detection_class_entities"][i].numpy().decode("utf-8")

            if self.print_entities:
                if detections["detection_scores"][i] > 0.1:
                    found_entities.add(entity)

            if entity in self.objects:
                score = float(detections["detection_scores"][i].numpy())
                boxes = detections["detection_boxes"][i].numpy().tolist()
                result = MobilenetV2Result(entity, score, boxes)
                results.append(result)

        if self.print_entities:
            print(found_entities)

        return img, results

    # Run Inference
    def _detect_objects_ssd_mobilenet_v2_fpnlite(
        self, img
    ) -> Tuple[Any, List[MobilenetV2Result]]:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imH, imW, _ = img.shape
        image_resized = cv2.resize(img, (self.width, self.height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.float_input:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # img = cv2.resize(img, (320, 320))
        # img = np.array(img).astype(np.float32)
        # img = np.expand_dims(img, axis=0)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[3]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        results: List[MobilenetV2Result] = []

        _ii = 0

        for i in range(len(scores)):
            # print(scores[i])
            if (scores[i] > self.threshold) and (scores[i] <= 1.0):
                _ii += 1

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                # print((xmin, ymin), (xmax, ymax))

                # Draw label
                fontScale = 0.5
                fontThickness = 2
                object_name = self.objects[
                    int(classes[i])
                ]  # Look up object name from "labels" array using class index
                label = "%s: %d%%" % (
                    object_name,
                    int(scores[i] * 100),
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
                    MobilenetV2Result(object_name, scores[i], (xmin, ymin, xmax, ymax))
                )

                # detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
        # print(_ii)

        # for i in range(len(classes[0])):
        #     entity = classes[0][i]
        #     score = scores[0][i]
        #     box = boxes[0][i]

        #     # if entity in objects:
        #     if score > self.threshold:
        #         result = MobilenetV2Result(entity, score, box)
        #         results.append(result)

        # retain only the highest result
        results = sorted(results, key=lambda x: x.score, reverse=True)

        # return results
        # return []
        return img, results

    # def _detect_objects_openimages_v4_ssd_mobilenet_v2_2(
    #     self, img, objects: List[str]
    # ) -> List[MobilenetV2Result]:
    #     converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    #     detections = self.model.signatures["default"](converted_img)
    #     results: List[MobilenetV2Result] = []

    #     found_entities = set()

    #     for i in range(len(detections["detection_class_entities"])):
    #         entity = detections["detection_class_entities"][i].numpy().decode("utf-8")

    #         if self.print_entities:
    #             if detections["detection_scores"][i] > 0.1:
    #                 found_entities.add(entity)

    #         if entity in objects:
    #             score = float(detections["detection_scores"][i].numpy())
    #             boxes = detections["detection_boxes"][i].numpy().tolist()
    #             result = MobilenetV2Result(entity, score, boxes)
    #             results.append(result)

    #     if self.print_entities:
    #         print(found_entities)

    #     return results

    def detect_objects(self, img):
        if self.model_type == "openimages_v4_ssd_mobilenet_v2":
            return self._detect_objects_openimages_v4_ssd_mobilenet_v2(img)

        elif self.model_type == "ssd_mobilenet_v2_fpnlite":
            return self._detect_objects_ssd_mobilenet_v2_fpnlite(img)

        elif self.model_type == "custom":
            # TODO
            raise Exception(
                "detect_objects function on custom model_type not implemented"
            )
        else:
            raise Exception(f"invalid model_type: {self.model_type}")

    def detect_objects_img_path(self, image_path: str):
        img = self.load_img(image_path)
        return self.detect_objects(img)
