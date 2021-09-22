import numpy as np

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import node, Pipeline
from kedro.runner import SequentialRunner
from typing import List, Optional, Union

from .objects_search import objects_search_catalog, objects_search_pipeline
from .polygon_search import polygon_search_catalog, polygon_search_pipeline
from ..common import load_image, Object, objects_packing_catalog, objects_packing_pipeline


class Solver:
    objects: List[Object]

    def __init__(self):
        self.objects = []

    def run(self, image: Union[str, np.ndarray], polygon: Optional[np.ndarray] = None) -> bool:
        catalog = DataCatalog({
            'placer.solver.input_image': MemoryDataSet(),
            'placer.solver.objects': MemoryDataSet()
        })
        pipeline = Pipeline([
            node(load_image, 'placer.solver.input_image', 'placer.objects_search.input_image')
        ])

        catalog.add_all(objects_search_catalog.datasets.__dict__)
        catalog.add_all(objects_packing_catalog.datasets.__dict__)
        pipeline += objects_search_pipeline + objects_packing_pipeline

        catalog.save('placer.solver.input_image', image)
        catalog.save('placer.solver.objects', self.objects)

        if polygon is not None:
            catalog.save('common.objects_packing.input_polygon', polygon)
        else:
            catalog.add_all(polygon_search_catalog.datasets.__dict__)
            pipeline += polygon_search_pipeline

        runner = SequentialRunner()
        result = runner.run(pipeline, catalog)
        return result['common.objects_packing.output_answer']
