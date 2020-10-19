import numpy
import tensorflow
from tensorflow.keras.layers import Activation, Input, Conv1D, Conv2D, Dense, ZeroPadding1D, \
    ZeroPadding2D, Flatten, Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model

from flowket.deepar.layers import ExpandInputDim, PeriodicPadding
from flowket.layers import ComplexConv1D, ComplexConv2D, ComplexDense, ToComplex64, VectorToComplexNumber
from flowket.layers.complex.tensorflow_ops import lncosh
from flowket.machines import Machine


class Linear(Machine):
    """docstring for Linear"""

    def __init__(self, keras_input_layer, **kwargs):
        super(Linear, self).__init__(keras_input_layer, **kwargs)
        x = ToComplex64()(keras_input_layer)
        x = Flatten()(x)
        self.manual_jacobian = x
        self.dense_layer = ComplexDense(1, use_bias=False)
        self._predictions = self.dense_layer(x)

    @property
    def predictions(self):
        return self._predictions


class LinearDepthTwo(Machine):
    """docstring for LinearDepth2"""

    def __init__(self, keras_input_layer, **kwargs):
        super(LinearDepthTwo, self).__init__(keras_input_layer, **kwargs)
        x = ToComplex64()(keras_input_layer)
        x = Flatten()(x)
        flat_input = x
        first_layer = ComplexDense(10, use_bias=False)
        second_layer = ComplexDense(1, use_bias=False)
        x = first_layer(x)
        self._predictions = second_layer(x)
        second_layer_jacobian = x
        num_of_first_layer_params = 10 * x.shape[-1]
        first_layer_jacobian = Lambda(
            lambda y: tensorflow.reshape(
                tensorflow.matmul(tensorflow.reshape(y, (-1, 1)),
                                  second_layer.kernel(),
                                  transpose_b=True),
                (-1, num_of_first_layer_params)))(flat_input)
        self.manual_jacobian = Concatenate()([first_layer_jacobian, second_layer_jacobian])

    @property
    def predictions(self):
        return self._predictions


def complex_values_linear_1d_model():
    input_layer = Input((7,))
    machine = Linear(input_layer)
    return Model(input_layer, machine.predictions)


def real_values_2d_model():
    input_layer = Input((7, 7))
    first_conv_layer = Conv2D(16, kernel_size=3, strides=1)
    second_conv_layer = Conv2D(8, kernel_size=3, strides=1)
    x = ExpandInputDim()(input_layer)
    x = first_conv_layer(ZeroPadding2D(1)(x))
    x = Activation('relu')(x)
    x = second_conv_layer(ZeroPadding2D(1)(x))
    x = Activation('relu')(x)
    x = Flatten()(x)
    first_dense_layer = Dense(7)
    second_dense_layer = Dense(2)
    x = first_dense_layer(x)
    x = Activation('relu')(x)
    x = second_dense_layer(x)
    x = VectorToComplexNumber(axis=-1)(x)
    return Model(input_layer, x)


def complex_values_2d_model():
    input_layer = Input((7, 7))
    first_conv_layer = ComplexConv2D(16, kernel_size=3, strides=1)
    second_conv_layer = ComplexConv2D(8, kernel_size=3, strides=1)
    x = ExpandInputDim()(input_layer)
    x = ToComplex64()(x)
    x = first_conv_layer(ZeroPadding2D(1)(x))
    x = Activation(lncosh)(x)
    x = second_conv_layer(ZeroPadding2D(1)(x))
    x = Activation(lncosh)(x)
    x = Flatten()(x)
    first_dense_layer = ComplexDense(7)
    second_dense_layer = ComplexDense(1)
    x = first_dense_layer(x)
    x = Activation(lncosh)(x)
    x = second_dense_layer(x)
    return Model(input_layer, x)


def complex_values_1d_model():
    padding = ((0, 3),)
    input_layer = Input(shape=(20, 1), dtype='int8')
    x = ToComplex64()(input_layer)
    for i in range(2):
        x = PeriodicPadding(padding)(x)
        x = ComplexConv1D(4, 4, use_bias=False)(x)
        x = Activation(lncosh)(x)
    x = Flatten()(x)
    x = Lambda(lambda y: tensorflow.reduce_sum(y, axis=1, keepdims=True))(x)
    return Model(input_layer, x)


def real_values_1d_model():
    input_layer = Input((7,))
    first_conv_layer = Conv1D(16, kernel_size=3, strides=1)
    second_conv_layer = Conv1D(8, kernel_size=3, strides=1)
    x = ExpandInputDim()(input_layer)
    x = first_conv_layer(ZeroPadding1D(1)(x))
    x = Activation('relu')(x)
    x = second_conv_layer(ZeroPadding1D(1)(x))
    x = Activation('relu')(x)
    x = Conv1D(20
               , kernel_size=1)(x)
    x = Flatten()(x)
    first_dense_layer = Dense(7)
    second_dense_layer = Dense(2)
    x = first_dense_layer(x)
    x = Activation('relu')(x)
    x = second_dense_layer(x)
    x = VectorToComplexNumber(axis=-1)(x)
    return Model(input_layer, x)
