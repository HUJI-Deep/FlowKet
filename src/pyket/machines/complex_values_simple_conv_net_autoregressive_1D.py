import tensorflow
from tensorflow.keras.layers import Activation, ZeroPadding1D
import tensorflow.keras.backend as K

from .abstract_machine import AutoNormalizedAutoregressiveMachine
from ..layers import ToComplex64, ToComplex128, DownShiftLayer, ExpandInputDim, ComplexConv1D
from ..layers.complex.tensorflow_ops import crelu, lncosh


def causal_conv_1d(x, filters, kernel_size, dilation_rate=1, activation=None, dtype=tensorflow.complex64):
    padding = kernel_size + (kernel_size - 1) * (dilation_rate - 1) - 1
    if padding > 0:
        x = ZeroPadding1D(padding=(padding, 0))(x)
    x = ComplexConv1D(filters=filters, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate, dtype=dtype)(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


class ComplexValuesSimpleConvNetAutoregressive1D(AutoNormalizedAutoregressiveMachine):
    """docstring for ConvNetAutoregressive1D"""

    def __init__(self, keras_input_layer, depth, num_of_channels, kernel_size=3,
                 use_dilation=True, max_dilation_rate=None, activation=lncosh, use_float64_ops=False, **kwargs):
        self.depth = depth
        self.num_of_channels = num_of_channels
        self.kernel_size = kernel_size
        self.use_dilation = use_dilation
        self.max_dilation_rate = max_dilation_rate
        self.activation = activation
        self.use_float64_ops = use_float64_ops
        if use_float64_ops:
            K.set_floatx('float64')
        self.layers_dtype = tensorflow.complex128 if self.use_float64_ops else tensorflow.complex64
        self._build_unnormalized_conditional_log_wave_function(keras_input_layer)
        super(ComplexValuesSimpleConvNetAutoregressive1D, self).__init__(keras_input_layer, **kwargs)

    @property
    def unnormalized_conditional_log_wave_function(self):
        return self._unnormalized_conditional_log_wave_function

    def _build_unnormalized_conditional_log_wave_function(self, keras_input_layer):
        dilation_rate = 1

        x = ExpandInputDim()(keras_input_layer)
        if self.use_float64_ops:
            x = ToComplex128()(x)
        else:   
            x = ToComplex64()(x)
        for i in range(self.depth - 1):
            x = causal_conv_1d(x, filters=self.num_of_channels,
                               kernel_size=self.kernel_size,
                               activation=self.activation, dilation_rate=dilation_rate, dtype=self.layers_dtype)
            if self.use_dilation:
                if self.max_dilation_rate is not None and dilation_rate < self.max_dilation_rate:
                    dilation_rate *= 2
        x = DownShiftLayer()(x)
        self._unnormalized_conditional_log_wave_function = causal_conv_1d(x, filters=2, kernel_size=1, dtype=self.layers_dtype)
