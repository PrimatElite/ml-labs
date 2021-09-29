import cv2
import numpy as np

from typing import Union


def load_image(image_or_path: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_or_path)
    elif isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        raise TypeError(f'Image has incorrect type: {type(image_or_path).__name__}')
    if len(image.shape) != 3 or image.shape[2] != 3 or image.dtype.type != np.uint8:
        raise ValueError('Image has incorrect value')
    return image
