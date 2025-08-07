# Yolov8n.py

from typing import Callable, List, Sequence
import cv2
import numpy as np
from ultralytics import YOLO

from classes.BoxedObject import BoxedObject

MatLike = np.ndarray


class YoloV8n:
    def __init__(
        self,
        model_path: str,
        objects: List[str],
        threshold: float = 0.5,
        img_width: int = 512,
        img_height: int = 512,
        max_object_size_percent: float = 0.8,
    ) -> None:
        # load YOLOv8 model
        self.model = YOLO(model_path)
        self.objects = objects
        self.threshold = threshold
        self.img_width = img_width
        self.img_height = img_height
        self.img_area = img_width * img_height
        self.max_object_size_percent = max_object_size_percent

    def detect(
        self,
        img: MatLike,
        on_yolo_receive: Callable[[BoxedObject, Sequence[BoxedObject]], None],
    ) -> MatLike:
        # run inference
        results = self.model.predict(
            source=img,
            conf=self.threshold,
            imgsz=(self.img_width, self.img_height),
            save=False,
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()  # type: ignore
        scores = boxes.conf.cpu().numpy()  # type: ignore
        classes = boxes.cls.cpu().numpy().astype(int)  # type: ignore

        detections: List[BoxedObject] = []
        for (x1, y1, x2, y2), cls, score in zip(xyxy, classes, scores):
            if score < self.threshold:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area / self.img_area > self.max_object_size_percent:
                continue

            print(f"cls {cls} with confidence {score:.2f}")

            label = self.objects[cls]
            # draw box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (10, 255, 0), 2)
            text = f"{label}: {int(score * 100)}%"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                img,
                (int(x1), int(y1) - h - 6),
                (int(x1) + w, int(y1)),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                img,
                text,
                (int(x1), int(y1) - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

            detections.append(
                BoxedObject(label, float(score), (int(x1), int(y1), int(x2), int(y2)))
            )

        # sort by confidence
        detections.sort(key=lambda obj: obj.score, reverse=True)

        if detections:
            on_yolo_receive(detections[0], detections)

        return img
