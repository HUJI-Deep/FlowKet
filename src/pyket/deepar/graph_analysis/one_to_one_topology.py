from tensorflow.keras.layers import Activation, Add, Average, Subtract, Multiply, Maximum, Minimum, \
    LeakyReLU, ELU, ThresholdedReLU, Softmax

from .data_structures import Dependency
from .layer_topology import LayerTopology
from .topology_manager import TopologyManager
from ..layers import CastingLayer, ExpandInputDim, LambdaWithOneToOneTopology, ToOneHot, PlusMinusOneToOneHot


class OneToOneTopology(LayerTopology):
    """docstring for OneToOneTopology"""

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values):
        values = dependencies_values
        if len(values) == 1:
            values = values[0]
        return self.layer(values)

    def get_spatial_dependency(self, spatial_location):
        layer_inputs = self.layer.input
        if not isinstance(layer_inputs, list):
            layer_inputs = [layer_inputs]
        return [Dependency(input_index=i, spatial_location=spatial_location) for i, _ in enumerate(layer_inputs)]


class OneToOneTopologyWithIdentity(OneToOneTopology):
    """docstring for OneToOneTopology"""

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values):
        values = dependencies_values
        if len(values) == 1:
            values = values[0]
        return values


class OneHotTopologyWithIdentity(OneToOneTopology):
    """docstring for OneToOneTopology"""

    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values):
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
TopologyManager().register_layer_topology(PlusMinusOneToOneHot, OneHotTopologyWithIdentity)
TopologyManager().register_layer_topology(ToOneHot, OneHotTopologyWithIdentity)
TopologyManager().register_layer_topology(ExpandInputDim, OneToOneTopologyWithIdentity)
