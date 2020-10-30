from tensorflow.keras.layers import Activation, Add, Average, Subtract, Multiply, Maximum, Minimum, \
    LeakyReLU, ELU, ThresholdedReLU, Softmax, Embedding

from .data_structures import Dependency
from .layer_topology import LayerTopology
from .topology_manager import TopologyManager
from ..layers import CastingLayer, ExpandInputDim, LambdaWithOneToOneTopology, ToOneHot, PlusMinusOneToOneHot, \
    NormalizeInLogSpace, LayerNormalization


class OneToOneTopology(LayerTopology):
    """docstring for OneToOneTopology"""
    def __init__(self, layer):
        super(OneToOneTopology, self).__init__(layer)
        inputs_shape = layer.input_shape
        if isinstance(layer, ExpandInputDim) or isinstance(layer, Embedding):
             inputs_shape = inputs_shape + (1,)
        if isinstance(inputs_shape, tuple):
            inputs_shape = [inputs_shape]
        self._spatial_inputs_size = [input_shape[1:-1] for input_shape in inputs_shape]

    def _broadcast_spatial_location(self, spatial_location, input_index):
        broadcasted_spatial_location = []
        for dim_index, dim_location in enumerate(spatial_location):
            if dim_location < self._spatial_inputs_size[input_index][dim_index]:
                broadcasted_spatial_location.append(dim_location)
            elif self._spatial_inputs_size[input_index][dim_index] == 1:
                broadcasted_spatial_location.append(0)
        return tuple(broadcasted_spatial_location)

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values, output_index=0):
        values = dependencies_values
        if len(values) == 1:
            values = values[0]
        return self.layer(values)

    def get_spatial_dependency(self, spatial_location, output_index=0):
        layer_inputs = self.layer.input
        if not isinstance(layer_inputs, list):
            layer_inputs = [layer_inputs]
        return [Dependency(input_index=i, spatial_location=self._broadcast_spatial_location(spatial_location, i))
                for i, _ in enumerate(layer_inputs)]


class OneToOneTopologyWithIdentity(OneToOneTopology):
    """docstring for OneToOneTopology"""

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values, output_index=0):
        values = dependencies_values
        if len(values) == 1:
            values = values[0]
        return values


class OneHotTopologyWithIdentity(OneToOneTopology):
    """docstring for OneToOneTopology"""

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values, output_index=0):
        values = dependencies_values[0]
        return self.layer(values[..., 0])


TopologyManager().register_layer_topology(Activation, OneToOneTopology)
TopologyManager().register_layer_topology(Add, OneToOneTopology)
TopologyManager().register_layer_topology(CastingLayer, OneToOneTopology)
TopologyManager().register_layer_topology(LambdaWithOneToOneTopology, OneToOneTopology)
TopologyManager().register_layer_topology(Subtract, OneToOneTopology)
TopologyManager().register_layer_topology(Multiply, OneToOneTopology)
TopologyManager().register_layer_topology(Average, OneToOneTopology)
TopologyManager().register_layer_topology(Maximum, OneToOneTopology)
TopologyManager().register_layer_topology(Minimum, OneToOneTopology)
TopologyManager().register_layer_topology(LeakyReLU, OneToOneTopology)
TopologyManager().register_layer_topology(ELU, OneToOneTopology)
TopologyManager().register_layer_topology(ThresholdedReLU, OneToOneTopology)
TopologyManager().register_layer_topology(Softmax, OneToOneTopology)
TopologyManager().register_layer_topology(LayerNormalization, OneToOneTopology)
TopologyManager().register_layer_topology(NormalizeInLogSpace, OneToOneTopology)
TopologyManager().register_layer_topology(PlusMinusOneToOneHot, OneHotTopologyWithIdentity)
TopologyManager().register_layer_topology(ToOneHot, OneHotTopologyWithIdentity)
TopologyManager().register_layer_topology(ExpandInputDim, OneToOneTopologyWithIdentity)
TopologyManager().register_layer_topology(Embedding, OneToOneTopology)
