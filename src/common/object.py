import numpy as np

from typing import Union

from .image import load_image
from .object_search import find_object


class Object:
    image: np.ndarray
    convex_hull: np.ndarray
    alpha: np.ndarray

    def __init__(self, image: Union[str, np.ndarray]):
        image: np.ndarray = load_image(image)
        self.image, self.convex_hull, self.alpha = find_object(image)
