from .data_structures import Dependency
from .layer_topology import LayerTopology
from .topology_manager import TopologyManager

from tensorflow.keras.layers import Concatenate


class ConcatenateTopology(LayerTopology):
    """docstring for ConcatenateTopology"""

    def __init__(self, layer):
        super(ConcatenateTopology, self).__init__(layer)
        self.concat_on_spatial_axis = not (self.layer.axis == -1 or self.layer.axis == 0 or self.layer.axis == len(self.layer.input_shape[0]) - 1)
        self.size_to_concat = [size[self.layer.axis] for size in self.layer.input_shape]

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values):
        if self.concat_on_spatial_axis:
            return dependencies_values[0]
        else:
            return self.layer(dependencies_values)

    def _get_spatial_dependency_when_concat_on_spatial_dim(self, spatial_location):
        output_spatial_location = list(spatial_location)
        for i, size in enumerate(self.size_to_concat):
            if output_spatial_location[self.layer.axis - 1] < size:
                return [Dependency(input_index=i, spatial_location=tuple(output_spatial_location))]
            else:
                output_spatial_location[self.layer.axis - 1] -= size

    def _get_spatial_dependency_when_concat_on_non_spatial_dim(self, spatial_location):
        layer_inputs = self.layer.input
        if not isinstance(layer_inputs, list):
            layer_inputs = [layer_inputs]
        return [Dependency(input_index=i, spatial_location=spatial_location) for i, _ in enumerate(layer_inputs)]

    def get_spatial_dependency(self, spatial_location):
        if self.concat_on_spatial_axis:
            return self._get_spatial_dependency_when_concat_on_spatial_dim(spatial_location)
        else:
            return self._get_spatial_dependency_when_concat_on_non_spatial_dim(spatial_location)


TopologyManager().register_layer_topology(Concatenate, ConcatenateTopology)
