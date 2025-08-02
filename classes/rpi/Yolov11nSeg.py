# Yolov11nSeg.py

from typing import Callable, List, Sequence
import cv2
import numpy as np
from ultralytics import YOLO

from classes.BoxedObject import BoxedObject

MatLike = np.ndarray


class YoloV11nSeg:
    def __init__(
        self,
        model_path: str,
        objects: List[str],
        threshold: float = 0.5,
        img_width: int = 512,
        img_height: int = 512,
        max_object_size_percent: float = 0.8,
    ) -> None:
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
        on_yolov11n_seg_receive: Callable[[BoxedObject, Sequence[BoxedObject]], None],
    ) -> MatLike:
        results = self.model.predict(
            source=img,
            conf=self.threshold,
            imgsz=(self.img_width, self.img_height),
            save=False,
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        # guard against missing masks
        masks = getattr(r, "masks", None)
        if masks is None or masks.xy is None:
            masks_xy = [[] for _ in range(len(xyxy))]
        else:
            masks_xy = masks.xy
            # print(f"masks_xy: {len(masks_xy)}: {masks_xy}")

        detections: List[BoxedObject] = []
        for (x1, y1, x2, y2), cls, score, polys in zip(xyxy, classes, scores, masks_xy):
            if score < self.threshold:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area / self.img_area > self.max_object_size_percent:
                continue

            label = self.objects[cls]
            color = (255, 10, 10)

            bgr = color[::-1]
            alpha = 0.2

            # draw mask
            # if len(polys) > 0:
            #     polys = polys[0]
            # print("---------------")
            # print(len(polys))
            # print(polys.shape)
            # print(polys)
            # print("---------------")

            # before your polys loop
            overlay = img.copy()

            if isinstance(polys, np.ndarray) and polys.ndim == 2:
                # single polygon of shape (N,2)
                pts = polys.astype(np.int32)
                cv2.fillPoly(overlay, [pts], bgr)

            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # for poly in polys:
            #     if len(poly) < 3:
            #         continue
            #     print(f"poly: {len(poly)}: {poly}")
            #     pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            #     cv2.fillPoly(img, [pts], color)

            # draw bounding box
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
            )
            # draw label background
            text = f"{label}: {int(score * 100)}%"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                img,
                (int(x1), int(y1) - h - 4),
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
                1,
            )

            detections.append(
                BoxedObject(label, float(score), (int(x1), int(y1), int(x2), int(y2)))
            )

        detections.sort(key=lambda obj: obj.score, reverse=True)
        if detections:
            on_yolov11n_seg_receive(detections[0], detections)
        return img
