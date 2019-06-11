from tensorflow.keras.layers import Activation, Add, Average, Subtract, Multiply, Maximum, Minimum, \
    LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax

from .data_structures import Dependency
from .layer_topology import LayerTopology
from .topology_manager import TopologyManager
from ...layers import VectorToComplexNumber, CastingLayer, ExpandInputDim, \
    LambdaWithOneToOneTopology


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


TopologyManager().register_layer_topology(OneToOneTopology, Activation)
TopologyManager().register_layer_topology(OneToOneTopology, Add)
TopologyManager().register_layer_topology(OneToOneTopology, CastingLayer)
TopologyManager().register_layer_topology(OneToOneTopology, VectorToComplexNumber)
TopologyManager().register_layer_topology(OneToOneTopology, LambdaWithOneToOneTopology)
TopologyManager().register_layer_topology(OneToOneTopology, Subtract)
TopologyManager().register_layer_topology(OneToOneTopology, Multiply)
TopologyManager().register_layer_topology(OneToOneTopology, Average)
TopologyManager().register_layer_topology(OneToOneTopology, Maximum)
TopologyManager().register_layer_topology(OneToOneTopology, Minimum)
TopologyManager().register_layer_topology(OneToOneTopology, LeakyReLU)
TopologyManager().register_layer_topology(OneToOneTopology, ELU)
TopologyManager().register_layer_topology(OneToOneTopology, ThresholdedReLU)
TopologyManager().register_layer_topology(OneToOneTopology, Softmax)
TopologyManager().register_layer_topology(OneToOneTopologyWithIdentity, ExpandInputDim)
