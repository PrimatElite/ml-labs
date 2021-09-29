import numpy as np

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import node, Pipeline


def pack_objects(polygon: np.ndarray, objects: np.ndarray) -> bool:
    return True


objects_packing_pipeline = Pipeline([
    node(pack_objects, ['common.objects_packing.input_polygon', 'common.objects_packing.input_objects'],
         'common.objects_packing.output_answer')
])


objects_packing_catalog = DataCatalog({
    'common.objects_packing.input_objects': MemoryDataSet(),
    'common.objects_packing.input_polygon': MemoryDataSet(),
    'common.objects_packing.output_answer': MemoryDataSet()
})
