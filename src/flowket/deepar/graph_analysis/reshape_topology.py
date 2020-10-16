import numpy

from .data_structures import Dependency
from .layer_topology import LayerTopology
from .topology_manager import TopologyManager

from tensorflow.keras.layers import Reshape


def to_flat_spatial_location(spatial_location, spatial_shape):
    flat_spatial_location = 0
    for dim_location, multiplication_factor in zip(spatial_location, spatial_shape):
        flat_spatial_location *= multiplication_factor
        flat_spatial_location += dim_location
    return flat_spatial_location


def from_flat_index_to_spatial_location(flat_index, spatial_shape):
    input_spatial_location = []
    for factor in spatial_shape[::-1]:
        input_spatial_location.append(flat_index % factor)
        flat_index = flat_index // factor
    return tuple(input_spatial_location[::-1])


class ReshapeTopology(LayerTopology):
    """docstring for PaddingTopology"""

    def __init__(self, layer):
        super(ReshapeTopology, self).__init__(layer)
        total_size = numpy.prod(self.layer.get_input_shape_at(0)[1:])
        self.target_shape = self.layer.target_shape
        if -1 in self.target_shape:
            minus_one_idx = self.target_shape.index(-1)
            if -1 in self.target_shape[minus_one_idx + 1:]:
                raise Exception('Reshape output shape cant contain two unknowns')
            if total_size % numpy.prod(self.target_shape) != 0:
                raise Exception('bad target shape')
            self.target_shape = self.target_shape[:] + (
                -1 * total_size // numpy.prod(self.target_shape),) + self.target_shape[minus_one_idx + 1:]
        if self.target_shape[-1] != self.layer.get_input_shape_at(0)[-1]:
            raise Exception("can't move information between spatial and features dimensions")

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values):
        return dependencies_values[0]

    def get_spatial_dependency(self, spatial_location):
        flat_spatial_location = to_flat_spatial_location(spatial_location, self.target_shape[:-1])
        input_spatial_location = from_flat_index_to_spatial_location(flat_spatial_location, self.layer.get_input_shape_at(0)[1:-1])
        return [Dependency(input_index=0, spatial_location=input_spatial_location)]


TopologyManager().register_layer_topology(Reshape, ReshapeTopology)
