import numpy as np

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import node, Pipeline
from kedro.runner import SequentialRunner
from typing import Union

from .image import load_image
from .object_search import object_search_catalog, object_search_pipeline


class Object:
    image: np.ndarray
    polygon: np.ndarray

    def __init__(self, image: Union[str, np.ndarray]):
        catalog = DataCatalog({
            'common.object.input_image': MemoryDataSet()
        })
        image_pipeline = Pipeline([
            node(load_image, 'common.object.input_image', 'common.object_search.input_image')
        ])

        catalog.add_all(object_search_catalog.datasets.__dict__)
        pipeline = image_pipeline + object_search_pipeline

        catalog.save('common.object.input_image', image)

        runner = SequentialRunner()
        result = runner.run(pipeline, catalog)

        self.image = result['common.object_search.output_image']
        self.polygon = result['common.object_search.output_polygon']
