import numpy as np

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import node, Pipeline


def find_polygon(image: np.ndarray) -> np.ndarray:
    return np.array([])


polygon_search_pipeline = Pipeline([
    node(find_polygon, 'placer.solver.input_image', 'common.objects_packing.input_polygon')
])


polygon_search_catalog = DataCatalog({})
