from dataclasses import dataclass
from typing import Tuple


@dataclass
class BoxedObject:
    entity: str
    score: float
    boxes: Tuple[float, float, float, float]
