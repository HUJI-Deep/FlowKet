from . import SimpleConvNetAutoregressive1D
from ..deepar.layers import ExpandInputDim, GatherLayer
from ..deepar.ordering import to_flat_ordering, to_flat_inverse_ordering
from .abstract_machine import AutoNormalizedAutoregressiveMachine

import numpy

from tensorflow.keras.layers import Reshape
from tensorflow.python.keras import backend as K


class SimpleCustomOrderingAutoregressive(AutoNormalizedAutoregressiveMachine):
    """docstring for ConvNetAutoregressive1D"""

    def __init__(self, autoregressive_ordering, keras_input_layer, depth, num_of_channels, **kwargs):
        self.autoregressive_ordering = list(autoregressive_ordering)
        self.input_shape = K.int_shape(keras_input_layer)[1:]
        one_dim_model = SimpleConvNetAutoregressive1D(self._build_input(keras_input_layer), depth,
                                                      num_of_channels, should_expand_input_dim=False,
                                                      **kwargs)
        self._build_output(one_dim_model.unnormalized_conditional_log_wave_function)
        super(SimpleCustomOrderingAutoregressive, self).__init__(keras_input_layer)

    def _build_input(self, keras_input_layer):
        x = ExpandInputDim()(keras_input_layer)
        x = Reshape((numpy.prod(self.input_shape), 1))(x)
        x = GatherLayer(list(to_flat_ordering(self.autoregressive_ordering, self.input_shape)))(x)
        return x

    def _build_output(self, one_dim_unnormalized_conditional_log_wave_function):
        x = one_dim_unnormalized_conditional_log_wave_function
        x = GatherLayer(list(to_flat_inverse_ordering(self.autoregressive_ordering, self.input_shape)))(x)
        x = Reshape(self.input_shape + (2,))(x)
        self._unnormalized_conditional_log_wave_function = x
    
    @property
    def unnormalized_conditional_log_wave_function(self):
        return self._unnormalized_conditional_log_wave_function
