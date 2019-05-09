from .abstract_machine import AutoNormalizedAutoregressiveMachine
from ..layers import ToFloat32, DownShiftLayer, VectorToComplexNumber, WeightNormalization

from tensorflow.keras.layers import Lambda, Conv1D, Reshape, ZeroPadding1D
from tensorflow.keras import backend as K


class SimpleConvNetAutoregressive1D(AutoNormalizedAutoregressiveMachine):
    """docstring for ConvNetAutoregressive1D"""
    def __init__(self, keras_input_layer, depth, num_of_channels, kernel_size=3, strides=1, activation='relu', weights_normalization=True, **kwargs):
        self.depth = depth
        self.num_of_channels = num_of_channels
        self.kernel_size = kernel_size
        self.strides = strides
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
            conv_layer = Conv1D(filters=self.num_of_channels, kernel_size=self.kernel_size, padding='valid',
                        strides=self.strides, activation=self.activation)
            if self.weights_normalization:
                conv_layer = WeightNormalization(conv_layer)
            x = conv_layer(ZeroPadding1D(padding=(self.kernel_size - self.strides, 0))(x))
        x = DownShiftLayer()(x)
        conv_layer = Conv1D(filters=4, kernel_size=1, padding='valid', strides=1)
        if self.weights_normalization:
            conv_layer = WeightNormalization(conv_layer)
        x = conv_layer(x)
        x = Reshape((K.int_shape(keras_input_layer)[1], 2, 2))(x)
        self._unnormalized_conditional_log_wave_function = VectorToComplexNumber()(x)
