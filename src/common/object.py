import cv2
import numpy as np

from shapely.geometry import Polygon
from typing import Tuple, Union

from .image import load_image
from .object_search import find_object, PAPER_SIZE


PIXELS_PER_MM = PAPER_SIZE[0] / 297


class Object:
    image: np.ndarray
    convex_hull: np.ndarray
    scaled_convex_hull: np.ndarray
    alpha: np.ndarray
    min_length: float
    bounds: Tuple[float, float, float, float]

    def __init__(self, image: Union[str, np.ndarray]):
        image: np.ndarray = load_image(image)
        self.image, self.convex_hull, self.alpha = find_object(image)
        self.scaled_convex_hull = self.convex_hull.astype(float) / PIXELS_PER_MM
        rotated_rectangle = cv2.minAreaRect(np.expand_dims(self.convex_hull, 1))
        rotated_box = np.int0(cv2.boxPoints(rotated_rectangle))
        self.min_length = min(np.linalg.norm(rotated_box[1] - rotated_box[0]),
                              np.linalg.norm(rotated_box[1] - rotated_box[2])) / PIXELS_PER_MM
        self.bounds = tuple(*Polygon(self.scaled_convex_hull).bounds)
