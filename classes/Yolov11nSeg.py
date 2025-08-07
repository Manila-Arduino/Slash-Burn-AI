# Yolov11nSeg.py

from typing import Callable, Dict, List, Sequence, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from classes.SegmentedObject import SegmentedObject

MatLike = np.ndarray


class YoloV11nSeg:

    def __init__(
        self,
        model_path: str,
        objects: List[Tuple[str, Tuple[int, int, int]]],
        threshold: float = 0.5,
        img_width: int = 512,
        img_height: int = 512,
        max_object_size_percent: float = 0.8,
        allowed: List[str] = [],
    ) -> None:
        self.model = YOLO(model_path)
        self.objects = objects
        self.threshold = threshold
        self.img_width = img_width
        self.img_height = img_height
        self.img_area = img_width * img_height
        self.max_object_size_percent = max_object_size_percent
        self.allowed = allowed

    def detect(
        self,
        img: MatLike,
        on_yolov11n_seg_receive: Callable[
            [SegmentedObject, Sequence[SegmentedObject]], None
        ],
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
        xyxy = boxes.xyxy.cpu().numpy()  # type: ignore
        scores = boxes.conf.cpu().numpy()  # type: ignore
        classes = boxes.cls.cpu().numpy().astype(int)  # type: ignore
        # guard against missing masks
        masks = getattr(r, "masks", None)
        if masks is None or masks.xy is None:
            masks_xy = [[] for _ in range(len(xyxy))]
        else:
            masks_xy = masks.xy
            # print(f"masks_xy: {len(masks_xy)}: {masks_xy}")

        detections: List[SegmentedObject] = []
        for (x1, y1, x2, y2), cls, score, polys in zip(xyxy, classes, scores, masks_xy):
            if score < self.threshold:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area / self.img_area > self.max_object_size_percent:
                continue

            label = self.objects[cls][0]
            color = self.objects[cls][1]

            if self.allowed and label not in self.allowed:
                continue

            bgr = color[::-1]
            alpha = 0.3

            overlay = img.copy()

            if isinstance(polys, np.ndarray) and polys.ndim == 2:
                # single polygon of shape (N,2)
                pts = polys.astype(np.int32)
                cv2.fillPoly(overlay, [pts], bgr)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                cv2.polylines(img, [pts], isClosed=True, color=bgr, thickness=1)
            else:
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # draw label background
            text = f"{label}: {int(score * 100)}%"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if isinstance(polys, np.ndarray) and polys.ndim == 2:
                pts = polys.astype(np.int32)
                # compute centroid via moments
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))

                # adjust origin so text is centered
                x_text = cx - w // 2
                y_text = cy + h // 2

                # draw background rectangle
                cv2.rectangle(
                    img,
                    (x_text, y_text - h - 4),
                    (x_text + w, y_text),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                # put text
                cv2.putText(
                    img,
                    text,
                    (x_text, y_text - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )
            else:
                # fallback to bounding-box placement if no valid polygon
                x1_i, y1_i = int(x1), int(y1)
                cv2.rectangle(
                    img,
                    (x1_i, y1_i - h - 4),
                    (x1_i + w, y1_i),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    img,
                    text,
                    (x1_i, y1_i - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

            if isinstance(polys, np.ndarray) and polys.ndim == 2:
                pts = polys.astype(np.int32)
                poly_area = cv2.contourArea(pts)
                area_percent = poly_area / self.img_area
                point_list = [(int(x), int(y)) for x, y in pts]

                detections.append(
                    SegmentedObject(
                        entity=label,
                        score=float(score),
                        points=point_list,
                        area_percent=area_percent,
                    )
                )

        detections.sort(key=lambda obj: obj.score, reverse=True)
        if detections:
            on_yolov11n_seg_receive(detections[0], detections)
        return img
