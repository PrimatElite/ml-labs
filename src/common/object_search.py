import numpy as np

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import node, Pipeline
from typing import Tuple


def find_object(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return image, np.array([])


object_search_pipeline = Pipeline([
    node(find_object, 'common.object_search.input_image',
         ['common.object_search.output_image', 'common.object_search.output_polygon'])
])


object_search_catalog = DataCatalog({
    'common.object_search.input_image': MemoryDataSet(),
    'common.object_search.output_image': MemoryDataSet(),
    'common.object_search.output_polygon': MemoryDataSet()
})
