import numpy as np

from typing import Optional, Union

from .placer import placer


def check_image(image: Union[str, np.ndarray], polygon: Optional[np.ndarray] = None) -> bool:
    return placer.run(image, polygon)
