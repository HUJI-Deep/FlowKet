import abc
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Conv1D, Conv2D, Dense

from flowket.deepar.utils.singleton import Singleton

from ..layers import ComplexConv1D, ComplexConv2D, ComplexDense, ComplexLayer
from ..layers.complex.tensorflow_ops import extract_complex_image_patches


class LayerJacobian(abc.ABC):
    """docstring for LayerTopology"""

    def __init__(self, layer):
        super(LayerJacobian, self).__init__()
        self.layer = layer

    @abc.abstractmethod
    def jacobian(self, activations_grad):
        pass


class DenseJacobian(LayerJacobian):
    """docstring for LayerTopology"""

    def __init__(self, layer):
        super(DenseJacobian, self).__init__(layer)
        assert layer.activation is None or layer.activation == tf.keras.activations.linear

    def jacobian(self, activations_grad):
        kernel_jacobian = tf.expand_dims(self.layer.input, axis=-1) @ tf.expand_dims(activations_grad, axis=1)
        bias_jacobian = activations_grad
        if self.layer.use_bias:
            return [kernel_jacobian, bias_jacobian]
        else:
            return [kernel_jacobian]


def to_4d_shape(size):
    if len(size) == 1:
        size = (size[0], 1)
    return (1,) + size + (1,)


class ConvJacobian(LayerJacobian):
    """docstring for LayerTopology"""

    def __init__(self, layer):
        super(ConvJacobian, self).__init__(layer)
        self.extract_image_patches = tf.extract_image_patches
        if isinstance(self.layer, ComplexLayer):
            self.extract_image_patches = extract_complex_image_patches
        assert layer.activation is None or layer.activation == tf.keras.activations.linear
        assert layer.padding == 'valid'
        assert layer.strides == (1,) * len(layer.strides)

    def jacobian(self, activations_grad):
        layer_input = self.layer.input
        is_1d_conv = len(K.int_shape(layer_input)) == 3
        if is_1d_conv:
            layer_input = tf.expand_dims(layer_input, axis=2)
        reshaped_input = tf.reshape(tf.transpose(layer_input, (0, 3, 1, 2)),
                                    (-1,) + K.int_shape(layer_input)[1:-1] + (1,))
        input_spatial_patches = self.extract_image_patches(reshaped_input,
                                                           to_4d_shape(K.int_shape(activations_grad)[1:-1]),
                                                           to_4d_shape(self.layer.strides),
                                                           to_4d_shape(self.layer.dilation_rate),
                                                           self.layer.padding.upper())
        input_spatial_patches = tf.reshape(input_spatial_patches,
                                           (-1, K.int_shape(layer_input)[-1]) + K.int_shape(
                                               input_spatial_patches)[1:])
        reshaped_activations_grads = tf.reshape(activations_grad,
                                                (-1, K.int_shape(input_spatial_patches)[-1], self.layer.filters))
        kernel_jacobian = tf.einsum('bso,bixys->bxyio', reshaped_activations_grads,
                                    input_spatial_patches)
        if is_1d_conv:
            kernel_jacobian = kernel_jacobian[:, :, 0, :, :]
        spatial_axes = tuple(range(1, len(K.int_shape(activations_grad)) - 1))
        bias_jacobian = tf.math.reduce_sum(activations_grad, axis=spatial_axes)
        if self.layer.use_bias:
            return [kernel_jacobian, bias_jacobian]
        else:
            return [kernel_jacobian]


class JacobianManager(metaclass=Singleton):
    """docstring for JacobianManager"""

    def __init__(self):
        super(JacobianManager, self).__init__()
        self._layer_to_layer_jacobian = {}

    def register_layer_jacobian(self, layer_jacobian, layer):
        self._layer_to_layer_jacobian[layer] = layer_jacobian

    def get_layer_jacobian(self, layer):
        layer_type = type(layer)
        if layer_type in self._layer_to_layer_jacobian:
            return self._layer_to_layer_jacobian[layer_type](layer)
        else:
            raise Exception('unsupported layer type %s' % layer_type)


def complex_values_jacobians_to_real_parts(jacobians):
    layer_jacobians_real_weights = []
    for jacobian in jacobians:
        jacobian = tf.conj(jacobian)
        layer_jacobians_real_weights.append(tf.math.real(jacobian))
        layer_jacobians_real_weights.append(tf.math.imag(jacobian))
    return layer_jacobians_real_weights


def gradient_per_example(loss, keras_model):
    results = []
    layers_with_params = [layer for layer in keras_model.layers if layer.count_params() > 0]
    activations_grads = tf.gradients(tf.math.real(loss), [layer.output for layer in layers_with_params])
    for layer, activations_grad in zip(layers_with_params, activations_grads):
        layer_jacobians = JacobianManager().get_layer_jacobian(layer).jacobian(activations_grad)
        if isinstance(layer, ComplexLayer):
            layer_jacobians = complex_values_jacobians_to_real_parts(layer_jacobians)
        results += layer_jacobians
    return [tf.expand_dims(t, axis=1) for t in results]


def predictions_jacobian(keras_model):
    return Lambda(lambda x: gradient_per_example(tf.math.real(x), keras_model))(keras_model.output)


JacobianManager().register_layer_jacobian(DenseJacobian, Dense)
JacobianManager().register_layer_jacobian(DenseJacobian, ComplexDense)
JacobianManager().register_layer_jacobian(ConvJacobian, Conv1D)
JacobianManager().register_layer_jacobian(ConvJacobian, ComplexConv1D)
JacobianManager().register_layer_jacobian(ConvJacobian, Conv2D)
JacobianManager().register_layer_jacobian(ConvJacobian, ComplexConv2D)
