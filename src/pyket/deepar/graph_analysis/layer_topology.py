import abc

import tensorflow


class LayerTopology(abc.ABC):
    """docstring for LayerTopology"""

    def __init__(self, layer):
        super(LayerTopology, self).__init__()
        self.layer = layer

    @abc.abstractmethod
    def get_spatial_dependency(self, spatial_location, output_index=0):
        pass

    @abc.abstractmethod
    def apply_layer_for_single_spatial_location(self, spatial_location, dependencies_values, output_index=0):
        """

        Args: output_index: used when the layer has more than one output tensors dependencies_values: the spatial
        dependency values return from self.get_spatial_dependency or the batch_size if len(
        self.get_spatial_dependency == 0

        Returns:
            tensor with the layer output at spatial_location
        """
        pass

    def get_zeros(self, batch_size, output_index=0):
        output_shape = self.layer.output_shape
        if isinstance(output_shape, tuple):
            output_shape = [output_shape]
        return tensorflow.zeros(shape=(batch_size, output_shape[output_index][-1],),
                                dtype=tensorflow.as_dtype(self.layer.dtype))
