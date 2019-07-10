from .data_structures import Dependency
from .layer_topology import LayerTopology
from .topology_manager import TopologyManager
from ..layers import DownShiftLayer, RightShiftLayer


class DownShiftTopology(LayerTopology):
    """docstring for OneToOneTopology"""

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values):
        if spatial_location[0] == 0:
            batch_size = dependencies_values
            return self.get_zeros(batch_size)
        return dependencies_values[0]

    def get_spatial_dependency(self, spatial_location):
        if spatial_location[0] == 0:
            return []
        shifted_spatial_location = (spatial_location[0] - 1,) + spatial_location[1:]
        return [Dependency(input_index=0, spatial_location=shifted_spatial_location)]


class RightShiftTopology(LayerTopology):
    """docstring for OneToOneTopology"""

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values):
        if spatial_location[1] == 0:
            batch_size = dependencies_values
            return self.get_zeros(batch_size)
        return dependencies_values[0]

    def get_spatial_dependency(self, spatial_location):
        if spatial_location[1] == 0:
            return []
        shifted_spatial_location = spatial_location[:1] + (spatial_location[1] - 1,) + spatial_location[2:]
        return [Dependency(input_index=0, spatial_location=shifted_spatial_location)]


TopologyManager().register_layer_topology(DownShiftLayer, DownShiftTopology)
TopologyManager().register_layer_topology(RightShiftLayer, RightShiftTopology)
