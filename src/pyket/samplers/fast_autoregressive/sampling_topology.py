from .data_structures import Dependency
from .layer_topology import LayerTopology
from .topology_manager import TopologyManager

import tensorflow
from tensorflow.python.keras.engine.input_layer import InputLayer


class SamplingTopology(LayerTopology):
    def get_spatial_dependency(self, spatial_location):
        return [Dependency(input_index=0, spatial_location=spatial_location)]

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values):
        dependencies_values = dependencies_values[0]
        shape = tensorflow.shape(dependencies_values)[0:1]
        random_batch = tensorflow.random_uniform(shape, dtype=dependencies_values.dtype)
        return tensorflow.expand_dims(2 * tensorflow.cast(tensorflow.exp(dependencies_values[:, 0]) > random_batch,
                                                          self.layer.dtype) - 1, axis=-1)


TopologyManager().register_layer_topology(SamplingTopology, InputLayer)
