from .abstract_machine import AutoNormalizedAutoregressiveMachine
from ..layers import ToFloat32, DownShiftLayer, VectorToComplexNumber, WeightNormalization

from tensorflow.keras.layers import Activation, Conv1D, Lambda, Layer, Reshape, ZeroPadding1D
from tensorflow.keras import backend as K


class CausalConv1D(Layer):
    """docstring for CausalConv1D"""
    def __init__(self, filters, kernel_size, weights_normalization, activation=None):
        super(CausalConv1D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.weights_normalization = weights_normalization
        self.padding = self.kernel_size - 1

    def build(self, input_shape):
        self.conv_layer = Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=1)
        if self.weights_normalization:
            self.conv_layer = WeightNormalization(self.conv_layer, data_init=False)
        if self.padding > 0:
            self.padding_layer = ZeroPadding1D(padding=(self.padding, 0))
        self.activation_layer = Activation(self.activation)

    def call(self, x, mask=None):
        if self.padding > 0:
            x = self.padding_layer(x)
        x = self.conv_layer(x)
        return self.activation_layer(x)

    def get_config(self):
        config = {'filters': self.filters, 'kernel_size': self.kernel_size, 
                  'activation': self.activation, 'weights_normalization': self.weights_normalization}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

class SimpleConvNetAutoregressive1D(AutoNormalizedAutoregressiveMachine):
    """docstring for ConvNetAutoregressive1D"""
    def __init__(self, keras_input_layer, depth, num_of_channels, kernel_size=3, activation='relu', weights_normalization=True, **kwargs):
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
        x = Lambda(lambda y:K.expand_dims(y, axis=-1))(x)
        for i in range(self.depth - 1):
            x = CausalConv1D(filters=self.num_of_channels, 
                             kernel_size=self.kernel_size,
                             activation=self.activation, 
                             weights_normalization=self.weights_normalization)(x)
        x = DownShiftLayer()(x)
        x = CausalConv1D(filters=4, kernel_size=1, 
                         weights_normalization=self.weights_normalization)(x)
        x = Reshape((K.int_shape(keras_input_layer)[1], 2, 2))(x)
        self._unnormalized_conditional_log_wave_function = VectorToComplexNumber()(x)
