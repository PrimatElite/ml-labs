import cv2
import numpy as np

from typing import Union

from .image import load_image
from .object_search import find_object


class Object:
    image: np.ndarray
    convex_hull: np.ndarray
    alpha: np.ndarray
    min_length: float

    def __init__(self, image: Union[str, np.ndarray]):
        image: np.ndarray = load_image(image)
        self.image, self.convex_hull, self.alpha = find_object(image)
        rotated_rectangle = cv2.minAreaRect(np.expand_dims(self.convex_hull, 1))
        rotated_box = np.int0(cv2.boxPoints(rotated_rectangle))
        self.min_length = min(np.linalg.norm(rotated_box[1] - rotated_box[0]),
                              np.linalg.norm(rotated_box[1] - rotated_box[2]))
