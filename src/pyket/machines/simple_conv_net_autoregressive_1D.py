from .abstract_machine import AutoNormalizedAutoregressiveMachine
from ..layers import ToFloat32, DownShiftLayer, ExpandInputDim, VectorToComplexNumber, WeightNormalization

from tensorflow.keras.layers import Activation, Conv1D, ZeroPadding1D, Activation


def causal_conv_1d(x, filters, kernel_size, weights_normalization, dilation_rate=1, activation=None):
    padding = kernel_size - 1
    if padding > 0:
        x = ZeroPadding1D(padding=(padding, 0))(x)
    conv_layer = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate)
    if weights_normalization:
        conv_layer = WeightNormalization(conv_layer, data_init=False)
    x = conv_layer(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


class SimpleConvNetAutoregressive1D(AutoNormalizedAutoregressiveMachine):
    """docstring for ConvNetAutoregressive1D"""

    def __init__(self, keras_input_layer, depth, num_of_channels, kernel_size=3,
                 use_dilation=True, max_dilation_rate=None, activation='relu',
                 weights_normalization=True, **kwargs):
        self.depth = depth
        self.num_of_channels = num_of_channels
        self.kernel_size = kernel_size
        self.use_dilation = use_dilation
        self.max_dilation_rate = max_dilation_rate
        self.activation = activation
        self.weights_normalization = weights_normalization
        self._build_unnormalized_conditional_log_wave_function(keras_input_layer)
        super(SimpleConvNetAutoregressive1D, self).__init__(keras_input_layer, **kwargs)

    @property
    def unnormalized_conditional_log_wave_function(self):
        return self._unnormalized_conditional_log_wave_function

    def _build_unnormalized_conditional_log_wave_function(self, keras_input_layer):
        dilation_rate = 1
        x = ExpandInputDim()(keras_input_layer)
        x = ToFloat32()(x)
        for i in range(self.depth - 1):
            x = causal_conv_1d(x, filters=self.num_of_channels,
                               kernel_size=self.kernel_size,
                               activation=self.activation,
                               weights_normalization=self.weights_normalization,
                               dilation_rate=dilation_rate)
            if self.use_dilation:
                if self.max_dilation_rate is not None and dilation_rate < self.max_dilation_rate:
                    dilation_rate *= 2
        x = DownShiftLayer()(x)
        x = causal_conv_1d(x, filters=4, kernel_size=1,
                           weights_normalization=self.weights_normalization)
        self._unnormalized_conditional_log_wave_function = VectorToComplexNumber()(x)
