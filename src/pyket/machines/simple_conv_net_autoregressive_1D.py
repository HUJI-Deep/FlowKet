from .abstract_machine import AutoNormalizedAutoregressiveMachine
from ..layers import ToFloat32, DownShiftLayer, VectorToComplexNumber, WeightNormalization

from tensorflow.keras.layers import Activation, Conv1D, Lambda, Layer, Reshape, ZeroPadding1D
from tensorflow.keras import backend as K


def causal_conv_1d(x, filters, kernel_size, weights_normalization, activation=None):
    padding = kernel_size - 1
    if padding > 0:
        x = ZeroPadding1D(padding=(padding, 0))(x)
    conv_layer = Conv1D(filters=filters, kernel_size=kernel_size, strides=1)
    if weights_normalization:
        conv_layer = WeightNormalization(conv_layer, data_init=False)
    x = conv_layer(x)
    return Activation(activation)(x)


class SimpleConvNetAutoregressive1D(AutoNormalizedAutoregressiveMachine):
    """docstring for ConvNetAutoregressive1D"""

    def __init__(self, keras_input_layer, depth, num_of_channels, kernel_size=3, activation='relu',
                 weights_normalization=True, **kwargs):
        self.depth = depth
        self.num_of_channels = num_of_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.weights_normalization = weights_normalization
        self._build_unnormalized_conditional_log_wave_function(keras_input_layer)
        super(SimpleConvNetAutoregressive1D, self).__init__(keras_input_layer, **kwargs)

    @property
    def unnormalized_conditional_log_wave_function(self):
        return self._unnormalized_conditional_log_wave_function

    def _build_unnormalized_conditional_log_wave_function(self, keras_input_layer):
        x = ToFloat32()(keras_input_layer)
        x = Lambda(lambda y: K.expand_dims(y, axis=-1))(x)
        for i in range(self.depth - 1):
            x = causal_conv_1d(x, filters=self.num_of_channels,
                               kernel_size=self.kernel_size,
                               activation=self.activation,
                               weights_normalization=self.weights_normalization)
        x = DownShiftLayer()(x)
        x = causal_conv_1d(x, filters=4, kernel_size=1,
                           weights_normalization=self.weights_normalization)
        x = Reshape(K.int_shape(keras_input_layer)[1:] + (2, 2))(x)
        self._unnormalized_conditional_log_wave_function = VectorToComplexNumber()(x)
