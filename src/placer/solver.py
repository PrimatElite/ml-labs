import numpy as np

from typing import List, Optional, Union

from .objects_search import find_objects
from .polygon_search import find_polygon
from ..common import load_image, Object, pack_objects


class Solver:
    objects: List[Object]

    def __init__(self):
        self.objects = []

    def run(self, image: Union[str, np.ndarray], polygon: Optional[np.ndarray] = None) -> bool:
        image: np.ndarray = load_image(image)
        if polygon is None:
            polygon = find_polygon(image)
        objects = find_objects(image, self.objects)
        return pack_objects(polygon, objects)
