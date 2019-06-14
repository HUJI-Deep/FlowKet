import abc


class LayerTopology(abc.ABC):
    """docstring for LayerTopology"""

    def __init__(self, layer):
        super(LayerTopology, self).__init__()
        self.layer = layer

    @abc.abstractmethod
    def get_spatial_dependency(self, spatial_location):
        pass

    @abc.abstractmethod
    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values):
        pass
