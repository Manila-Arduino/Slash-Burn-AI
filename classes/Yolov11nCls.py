from typing import Callable, Dict, List, Sequence, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO

from classes.ClassificationObject import ClassificationObject

MatLike = np.ndarray


class Yolov11nCls:

    def __init__(
        self,
        model_path: str,
        objects: List[str],
        threshold: float = 0.5,
        img_width: int = 512,
        img_height: int = 512,
    ) -> None:
        # self.model = YOLO(model_path) #TODO: UNCOMMENT
        self.objects = objects
        self.threshold = threshold
        self.img_width = img_width
        self.img_height = img_height
        self.img_area = img_width * img_height

    def detect(
        self,
        img: MatLike,
        on_yolov11n_cls_receive: Callable[[Union[ClassificationObject, None]], None],
    ) -> MatLike:
        on_yolov11n_cls_receive(None)  # TODO: UPDATE, just temporary
        return img
