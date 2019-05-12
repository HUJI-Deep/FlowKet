from .abstract_machine import AutoNormalizedAutoregressiveMachine
from ..layers import ToFloat32, DownShiftLayer, RightShiftLayer, VectorToComplexNumber, WeightNormalization

from tensorflow.keras.layers import Activation, Add, Concatenate, Conv2D, Lambda, Reshape, ZeroPadding2D
from tensorflow.keras import backend as K


class ConvNetAutoregressive2D(AutoNormalizedAutoregressiveMachine):
    """docstring for ConvNetAutoregressive2D"""

    def __init__(self, keras_input_layer, depth, num_of_channels, kernel_size=3, strides=1, activation='relu',
                 weights_normalization=True, **kwargs):
        self.depth = depth
        self.num_of_channels = num_of_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.strides = strides
        self.activation = activation
        self.weights_normalization = weights_normalization
        self._build_unnormalized_conditional_log_wave_function(keras_input_layer)
        super(ConvNetAutoregressive2D, self).__init__(keras_input_layer, **kwargs)

    def _conv2d(self, filters, kernel_size):
        conv_layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=1)
        if self.weights_normalization:
            conv_layer = WeightNormalization(conv_layer, data_init=False)
        return conv_layer

    def _activation(self):
        return Activation(self.activation)

    def _causal_conv_2d(self, vertical_x, horizontal_x, use_horizontal_mask=False):
        filters = self.num_of_channels
        if self.padding > 0:
            vertical_x = ZeroPadding2D(padding=((self.padding, 0), (self.padding // 2, self.padding // 2)))(vertical_x)
            horizontal_x = ZeroPadding2D(padding=((0, 0), (self.padding, 0)))(horizontal_x)
        vertical_x = self._conv2d(filters=filters, kernel_size=self.kernel_size)(vertical_x)
        x = self._conv2d(filters=filters, kernel_size=(1, self.kernel_size))(horizontal_x)
        x = self._activation()(x)
        if use_horizontal_mask:
            x = RightShiftLayer()(x)
        x = self._conv2d(filters=filters // 2, kernel_size=1)(x)
        y = DownShiftLayer()(self._activation()(vertical_x))
        y = self._conv2d(filters=filters // 2, kernel_size=1)(y)
        x, y = self._activation()(x), self._activation()(y)
        x = Concatenate()([x, y])
        if self.padding > 0:
            x = ZeroPadding2D(padding=((self.padding, 0), (self.padding, 0)))(x)
        horizontal_x = self._conv2d(filters=filters, kernel_size=self.kernel_size)(x)
        return vertical_x, horizontal_x

    def _resiual_block(self, vertical_x, horizontal_x, num_of_layers=2):
        vertical_x_input, horizontal_x_input = vertical_x, horizontal_x
        for _ in range(num_of_layers - 1):
            vertical_x, horizontal_x = self._causal_conv_2d(vertical_x, horizontal_x)
            vertical_x, horizontal_x = self._activation()(vertical_x), self._activation()(horizontal_x)
        vertical_x, horizontal_x = self._causal_conv_2d(vertical_x, horizontal_x)
        vertical_x = Add()([vertical_x_input, vertical_x])
        horizontal_x = Add()([horizontal_x_input, horizontal_x])
        vertical_x, horizontal_x = self._activation()(vertical_x), self._activation()(horizontal_x)
        return vertical_x, horizontal_x

    def _build_unnormalized_conditional_log_wave_function(self, keras_input_layer):
        x = ToFloat32()(keras_input_layer)
        x = Lambda(lambda y: K.expand_dims(y, axis=-1))(x)
        vertical_x, horizontal_x = x, x
        vertical_x, horizontal_x = self._causal_conv_2d(vertical_x, horizontal_x)
        vertical_x, horizontal_x = self._activation()(vertical_x), self._activation()(horizontal_x)
        for i in range(self.depth - 2):
            vertical_x, horizontal_x = self._resiual_block(vertical_x, horizontal_x)
        _, x = self._causal_conv_2d(vertical_x, horizontal_x, use_horizontal_mask=True)
        x = Conv2D(filters=4, kernel_size=1, strides=1)(self._activation()(x))
        x = Reshape(K.int_shape(keras_input_layer)[1:] + (2, 2))(x)
        self._unnormalized_conditional_log_wave_function = VectorToComplexNumber()(x)

    @property
    def unnormalized_conditional_log_wave_function(self):
        return self._unnormalized_conditional_log_wave_function
