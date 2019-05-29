import numpy as np
import pytest
import tensorflow as tf

from tensorflow.python.ops.parallel_for import gradients
from tensorflow.keras.layers import Activation, Input, Conv1D, Conv2D, Conv2DTranspose, Dense, ZeroPadding1D, \
    ZeroPadding2D, ZeroPadding3D, Flatten, AveragePooling1D, AveragePooling2D, Lambda
from tensorflow.keras.models import Model
import keras.backend as K

from pyket.layers import ExpandInputDim, ComplexConv1D, ComplexConv2D, ComplexDense, ToComplex64, PeriodicPadding
from pyket.layers.complex.tensorflow_ops import lncosh
from pyket.utils.jacobian import gradient_per_example

DEFAULT_TF_GRAPH = tf.get_default_graph()


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
    second_dense_layer = Dense(1)
    x = first_dense_layer(x)
    x = Activation('relu')(x)
    x = second_dense_layer(x)
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
    x = Lambda(lambda y: tf.reduce_sum(y, axis=1, keepdims=True))(x)
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
    x = Conv1D(20, kernel_size=1)(x)
    x = Flatten()(x)
    first_dense_layer = Dense(7)
    second_dense_layer = Dense(1)
    x = first_dense_layer(x)
    x = Activation('relu')(x)
    x = second_dense_layer(x)
    return Model(input_layer, x)


@pytest.mark.parametrize('model_builder, batch_size', [
    (real_values_2d_model, 5),
    (complex_values_2d_model, 5),
    (real_values_1d_model, 5),
    (complex_values_1d_model, 5),
])
def test_equal_to_builtin_jacobian(model_builder, batch_size):
    with DEFAULT_TF_GRAPH.as_default():
        keras_model = model_builder()
        keras_model.summary()
        gradient_per_example_t = gradient_per_example(tf.real(keras_model.output), keras_model)
        tensorflow_jacobian_t = gradients.jacobian(tf.real(keras_model.output),
                                                   keras_model.weights, use_pfor=False)
        print(gradient_per_example_t)
        print(tensorflow_jacobian_t)
        gradient_per_example_func = K.function(inputs=[keras_model.input], outputs=gradient_per_example_t)
        tensorflow_jacobian_func = K.function(inputs=[keras_model.input], outputs=tensorflow_jacobian_t)
        size = (batch_size,) + K.int_shape(keras_model.input)[1:]
        batch = np.random.rand(*size)
        gradient_per_example_vals = gradient_per_example_func([batch])
        tensorflow_jacobian_vals = tensorflow_jacobian_func([batch])
        allclose = [np.allclose(a, b, rtol=1e-4) for a, b in zip(gradient_per_example_vals, tensorflow_jacobian_vals)]
        assert np.all(allclose)
