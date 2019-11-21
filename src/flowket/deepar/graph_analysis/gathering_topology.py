from .data_structures import Dependency
from .layer_topology import LayerTopology
from .topology_manager import TopologyManager
from ..layers import GatherLayer


class GatherTopology(LayerTopology):
    """docstring for PaddingTopology"""

    def __init__(self, layer):
        super(GatherTopology, self).__init__(layer)
        self.axis = self.layer.axis
        if self.axis < 0:
            self.axis = len(self.layer.input_shape) - self.axis
        self.features_gathering = self.axis == len(self.layer.input_shape) - 1

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values, output_index=0):
        if self.features_gathering:
            return self.layer(dependencies_values[0])
        return dependencies_values[0]

    def get_spatial_dependency(self, spatial_location, output_index=0):
        if self.features_gathering:
            input_spatial_location = spatial_location
        else:
            input_spatial_location = list(spatial_location)
            input_spatial_location[self.axis - 1] = self.layer.indices[spatial_location[self.axis - 1]]
            input_spatial_location = tuple(input_spatial_location)
        return [Dependency(input_index=0, spatial_location=input_spatial_location)]


TopologyManager().register_layer_topology(GatherLayer, GatherTopology)
