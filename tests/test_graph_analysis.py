import itertools

from pyket.layers import VectorToComplexNumber, \
    ComplexConv2D, ComplexConv1D, ComplexConv3D, ToComplex64
from pyket.deepar.layers import LambdaWithOneToOneTopology, RightShiftLayer, DownShiftLayer
from pyket.samplers.fast_autoregressive import TopologyManager

import pytest
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Add, Activation, Concatenate, Conv1D, Conv2D, Conv3D, \
    Input, Subtract, Multiply, Average, Maximum, Minimum, LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax, \
    ZeroPadding1D, ZeroPadding2D, ZeroPadding3D

DEFAULT_TF_GRAPH = tf.get_default_graph()


@pytest.mark.parametrize('layer, input_layer, batch_size', [
    (Concatenate(), [Input((6, 6, 5)), Input((6, 6, 5))], 32),
    (Add(), [Input((6, 6, 5)), Input((6, 6, 5))], 32),
    (Subtract(), [Input((6, 6, 5)), Input((6, 6, 5))], 32),
    (Multiply(), [Input((6, 6, 5)), Input((6, 6, 5))], 32),
    (Average(), [Input((6, 6, 5)), Input((6, 6, 5)), Input((6, 6, 5))], 32),
    (Maximum(), [Input((6, 6, 5)), Input((6, 6, 5))], 32),
    (Minimum(), [Input((6, 6, 5)), Input((6, 6, 5))], 32),
    (Activation('relu'), Input((6, 6, 5)), 32),
    (LeakyReLU(), Input((6, 6, 5)), 32),
    (ELU(), Input((6, 6, 5)), 32),
    (ThresholdedReLU(), Input((6, 6, 5)), 32),
    (Softmax(), Input((6, 6, 5)), 32),
    (DownShiftLayer(), Input((6, 6, 5)), 32),
    (DownShiftLayer(), Input((6, 5)), 32),
    (RightShiftLayer(), Input((6, 6, 5)), 32),
    (VectorToComplexNumber(), Input((6, 6, 2)), 32),
    (LambdaWithOneToOneTopology(lambda x: tf.exp(x)), Input((6, 6, 2)), 32),
    (ToComplex64(), Input((6, 6, 2)), 32),
    (Conv2D(8, kernel_size=3, data_format='channels_last', bias_initializer='ones'), Input((6, 6, 5)), 1),
    (Conv2D(8, kernel_size=3, data_format='channels_last', bias_initializer='ones'), Input((6, 6, 1)), 32),
    (Conv2D(8, kernel_size=3, data_format='channels_last', bias_initializer='ones'), Input((6, 6, 5)), 32),
    (Conv2D(8, kernel_size=3, activation='relu', data_format='channels_last', bias_initializer='ones'), Input((6, 6, 5)), 32),
    (Conv2D(8, kernel_size=3, data_format='channels_last', bias_initializer='ones'), Input((7, 7, 5)), 32),
    (Conv2D(8, kernel_size=3, dilation_rate=2, data_format='channels_last', bias_initializer='ones'), Input((10, 10, 5)), 32),
    (Conv2D(8, kernel_size=3, strides=2, data_format='channels_last', bias_initializer='ones'), Input((10, 10, 5)), 32),
    (ZeroPadding2D(((1, 2), (5, 3))), Input((10, 10, 5)), 32),
    (ZeroPadding2D(((1, 0), (5, 0))), Input((10, 10, 5)), 32),
    (ZeroPadding2D(((0, 1), (0, 5))), Input((10, 10, 5)), 32),
    (ZeroPadding3D(((0, 1), (0, 5), (0, 3))), Input((10, 10, 8, 5)), 32),
    (ZeroPadding3D(((1, 0), (5, 0), (3, 0))), Input((10, 10, 8, 5)), 32),
    (ZeroPadding3D(((1, 2), (5, 4), (3, 6))), Input((10, 10, 8, 5)), 32),
    (Conv1D(8, kernel_size=3, data_format='channels_last', bias_initializer='ones'), Input((6, 5)), 32),
    (ZeroPadding1D(padding=(1, 0)), Input((6, 5)), 32),
    (ZeroPadding1D(padding=(1, 1)), Input((6, 5)), 32),
    (ZeroPadding1D(padding=(1, 1)), Input((6, 5)), 32),
    (Conv3D(8, kernel_size=(3, 3, 3), data_format='channels_last', bias_initializer='ones'), Input((6, 6, 6, 5)), 32),
    (ComplexConv2D(8, kernel_size=3), Input((6, 6, 5), dtype=tf.complex64), 32),
    (ComplexConv2D(8, kernel_size=3), Input((7, 7, 5), dtype=tf.complex64), 32),
    (ComplexConv2D(8, kernel_size=3, dilation_rate=2), Input((10, 10, 5), dtype=tf.complex64), 32),
    (ComplexConv2D(8, kernel_size=3, strides=2), Input((10, 10, 5), dtype=tf.complex64), 32),
    (ComplexConv1D(8, kernel_size=3), Input((6, 5), dtype=tf.complex64), 32),
    (ComplexConv3D(8, kernel_size=(3, 1, 1)), Input((6, 1, 1, 5), dtype=tf.complex64), 32),
    (ComplexConv3D(8, kernel_size=3), Input((6, 6, 8, 5), dtype=tf.complex64), 32),
])
def test_apply_layer_for_single_spatial_location(layer, input_layer, batch_size):
    with DEFAULT_TF_GRAPH.as_default():
        normal_output = layer(input_layer)
        has_multiple_inputs = isinstance(input_layer, list)
        layer_output_with_topology_manager = get_layer_output_with_topology_manager(has_multiple_inputs,
                                                                                    input_layer,
                                                                                    layer)
        output_function = K.function(inputs=input_layer,
                                     outputs=[layer_output_with_topology_manager, normal_output])
        if has_multiple_inputs:
            batch = [np.random.rand(*((batch_size,) + layer_input_shape[1:])) for layer_input_shape in
                     layer.input_shape]
        else:
            batch = np.random.choice(100, size=(batch_size,) + layer.input_shape[1:]).astype(np.float32)
        output_values = output_function(batch)
        assert np.allclose(output_values[0], output_values[1], rtol=1e-3, atol=1e-4)


def get_layer_output_with_topology_manager(has_multiple_inputs, input_layer, layer):
    output_values = []
    layer_topology = TopologyManager().get_layer_topology(layer)
    for spatial_location in itertools.product(*[range(dim_shape) for dim_shape in
                                                layer.output_shape[1:-1]]):
        dependencies_values = []
        for dependency in layer_topology.get_spatial_dependency(spatial_location):
            if has_multiple_inputs:
                input_tensor = input_layer[dependency.input_index]
            else:
                input_tensor = input_layer
            dependencies_values.append(tf.slice(input_tensor, (0,) + dependency.spatial_location + (0,),
                                                (-1,) + (1,) * len(dependency.spatial_location) + (-1,)))
        if len(dependencies_values) == 0:
            output_values.append(
                tf.zeros(tf.concat([tf.shape(layer.output)[:1]] + [[1]] * len(spatial_location) + [tf.shape(layer.output)[-1:]], axis=0), dtype=layer.dtype))
        else:
            output_values.append(
                layer_topology.apply_layer_for_single_spatial_location(spatial_location, dependencies_values))
    output_shape = (-1,) + layer.output_shape[1:]
    output_of_apply_layer_for_single_spatial_location = tf.reshape(tf.stack(output_values, axis=1),
                                                                   output_shape)
    return output_of_apply_layer_for_single_spatial_location
