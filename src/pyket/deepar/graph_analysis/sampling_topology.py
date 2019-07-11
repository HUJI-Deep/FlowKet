from pyket.deepar.graph_analysis.data_structures import Dependency
from .layer_topology import LayerTopology
from .topology_manager import TopologyManager

import tensorflow
from tensorflow.python.keras.engine.input_layer import InputLayer


class CategorialSamplingTopology(LayerTopology):
    def get_spatial_dependency(self, spatial_location, output_index=0):
        return [Dependency(input_index=0, spatial_location=spatial_location)]

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values, output_index=0):
        dependencies_values = dependencies_values[0]
        return tensorflow.cast(tensorflow.multinomial(dependencies_values, 1, output_dtype=tensorflow.int32),
                               self.layer.dtype)


class PlusMinusOneSamplingTopology(CategorialSamplingTopology):
    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values, output_index=0):

        batch = super().apply_layer_for_single_spatial_location(spatial_location, dependencies_values, output_index)
        return 1 - 2 * batch


TopologyManager().register_layer_topology(InputLayer, CategorialSamplingTopology)
