import numpy as np

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import node, Pipeline
from typing import List

from ..common import Object


def find_objects(image: np.ndarray, objects: List[Object]) -> np.ndarray:
    return np.array([obj.polygon for obj in objects])


objects_search_pipeline = Pipeline([
    node(find_objects, ['placer.objects_search.input_image', 'placer.solver.objects'],
         'common.objects_packing.input_objects')
])


objects_search_catalog = DataCatalog({
    'placer.objects_search.input_image': MemoryDataSet()
})
