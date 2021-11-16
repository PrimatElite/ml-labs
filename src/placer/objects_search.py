import numpy as np

from typing import List

from ..common import Object


def find_objects(image: np.ndarray, objects: List[Object]) -> np.ndarray:
    return np.array([obj.convex_hull for obj in objects])
