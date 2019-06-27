from . import SimpleConvNetAutoregressive1D
from ..deepar.layers import ExpandInputDim, GatherLayer
from ..deepar.ordering import to_flat_ordering, to_flat_inverse_ordering

import numpy
from tensorflow.keras.layers import Reshape


class SimpleCustomOrderingAutoregressive(SimpleConvNetAutoregressive1D):
    """docstring for ConvNetAutoregressive1D"""

    def __init__(self, autoregressive_ordering, keras_input_layer, depth, num_of_channels, **kwargs):
        self.autoregressive_ordering = list(autoregressive_ordering)
        self.input_shape = keras_input_layer.input_shape[1:]
        super(SimpleCustomOrderingAutoregressive, self).__init__(self._build_input(keras_input_layer), depth,
                                                                 num_of_channels, should_expand_input_dim=False,
                                                                 **kwargs)
        self.keras_input_layer = keras_input_layer
        self._build_output()

    def _build_input(self, keras_input_layer):
        x = ExpandInputDim()(keras_input_layer)
        x = Reshape((numpy.prod(self.input_shape), 1))(x)
        x = GatherLayer(to_flat_ordering(self.autoregressive_ordering, self.input_shape))(x)
        return x

    def _build_output(self):
        x = self._unnormalized_conditional_log_wave_function
        x = GatherLayer(to_flat_inverse_ordering(self.autoregressive_ordering, self.input_shape))(x)
        x = Reshape(self.input_shape + (2, ))(x)
        self._unnormalized_conditional_log_wave_function = x