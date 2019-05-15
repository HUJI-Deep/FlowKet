from ...utils import Singleton
from .layer_topology import LayerTopology

from tensorflow.keras.layers import Wrapper


class TopologyManager(metaclass=Singleton):
    """docstring for TopologyManager"""

    def __init__(self):
        super(TopologyManager, self).__init__()
        self._layer_to_layer_topology = {}

    def register_layer_topology(self, layer_topology, layer):
        self._layer_to_layer_topology[layer] = layer_topology

    def get_layer_topology(self, layer):
        layer_type = type(layer)
        if isinstance(layer, Wrapper):
            return self.get_layer_topology(layer.layer)
        if layer_type in self._layer_to_layer_topology:
            layer_topology = self._layer_to_layer_topology[layer_type](layer)
        elif isinstance(layer, LayerTopology):
            layer_topology = layer
        else:
            raise Exception('unsupported layer type %s' % layer_type)
        return layer_topology
