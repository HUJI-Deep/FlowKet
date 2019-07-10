import abc

import tensorflow


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
        """

        Args:
            dependencies_values: the spatial dependency values return from self.get_spatial_dependency or the batch_size
            if len(self.get_spatial_dependency == 0

        Returns:
            tensor with the layer output at spatial_location
        """
        pass

    def get_zeros(self, batch_size):
        return tensorflow.zeros(shape=(batch_size, self.layer.output_shape[-1],),
                                dtype=tensorflow.as_dtype(self.layer.dtype))
