import numpy

from .data_structures import Dependency
from .layer_topology import LayerTopology
from .topology_manager import TopologyManager

from tensorflow.keras.layers import Reshape


class ReshapeTopology(LayerTopology):
    """docstring for PaddingTopology"""

    def __init__(self, layer):
        super(ReshapeTopology, self).__init__(layer)
        total_size = numpy.prod(self.layer.input_shape[1:])
        self.target_shape = self.layer.target_shape
        if -1 in self.target_shape:
            minus_one_idx = self.target_shape.index(-1)
            if -1 in self.target_shape[minus_one_idx + 1:]:
                raise Exception('Reshape output shape cant contain two unknowns')
            if total_size % numpy.prod(self.target_shape) != 0:
                raise Exception('bad target shape')
            self.target_shape = self.target_shape[:] + (
            -1 * total_size // numpy.prod(self.target_shape),) + self.target_shape[minus_one_idx + 1:]
        if self.target_shape[-1] != self.layer.input_shape[-1]:
            raise Exception("can't move information between spatial and features dimensions")

        self.input_multiplications_factors = self.layer.input_shape[1:-1]
        self.output_multiplications_factors = self.target_shape[:-1]

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values):
        return dependencies_values[0]

    def get_spatial_dependency(self, spatial_location):
        flat_spatial_location = 0
        for dim_location, multiplication_factor in zip(spatial_location, self.output_multiplications_factors):
            flat_spatial_location *= multiplication_factor
            flat_spatial_location += dim_location

        input_spatial_location = []
        for factor in self.input_multiplications_factors[::-1]:
            input_spatial_location.append(flat_spatial_location % factor)
            flat_spatial_location = flat_spatial_location // factor

        return [Dependency(input_index=0, spatial_location=tuple(input_spatial_location[::-1]))]


TopologyManager().register_layer_topology(Reshape, ReshapeTopology)
