import numpy
import pytest

import tensorflow
from tensorflow.keras.layers import Input, Flatten, Lambda, Concatenate, Multiply
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K

from pyket.layers import ComplexDense, ToComplex64
from pyket.machines import Machine
from pyket.optimizers import ComplexValuesStochasticReconfiguration

ONE_DIM_INPUT = Input(shape=(16,), dtype='int8')
SCALAR_INPUT = Input(shape=(1,), dtype='int8')
TWO_DIM_INPUT = Input(shape=(4, 4), dtype='int8')

graph = tensorflow.get_default_graph()


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
        num_of_first_layer_params = numpy.prod([int(s) for s in first_layer.kernel.shape])
        first_layer_jacobian = Lambda(
            lambda y: tensorflow.reshape(
                tensorflow.matmul(tensorflow.reshape(y, (-1, 1)),
                                  second_layer.kernel,
                                  transpose_b=True),
                (-1, num_of_first_layer_params)))(flat_input)
        self.manual_jacobian = Concatenate()([first_layer_jacobian, second_layer_jacobian])

    @property
    def predictions(self):
        return self._predictions


@pytest.mark.parametrize('input_layer, machine_class, batch_size', [
    (ONE_DIM_INPUT, Linear, 128),
    (ONE_DIM_INPUT, LinearDepthTwo, 128),  # In this case the gradients has non vanished imaginary part
    (TWO_DIM_INPUT, Linear, 128),
    (TWO_DIM_INPUT, LinearDepthTwo, 128),
])
def test_get_wave_function_jacobian(input_layer, machine_class, batch_size):
    with graph.as_default():
        machine = machine_class(input_layer)
        model = Model(inputs=[input_layer], outputs=machine.predictions)
        optimizer = ComplexValuesStochasticReconfiguration(model, machine.predictions_jacobian)
        jacobian_function = K.function(inputs=[input_layer], outputs=[optimizer.get_wave_function_jacobian()])
        manual_jacobian_function = K.function(inputs=[input_layer], outputs=[machine.manual_jacobian])
        sample = numpy.random.choice(2, (batch_size,) + K.int_shape(input_layer)[1:]) * 2 - 1
        jacobian = jacobian_function([sample])[0]
        manual_jacobian = manual_jacobian_function([sample])[0]
        diff_norm = numpy.linalg.norm(jacobian - manual_jacobian, 'fro')
        jacobian_norm = numpy.linalg.norm(manual_jacobian, 'fro')
        assert (diff_norm / jacobian_norm) < 1e-5


@pytest.mark.parametrize('input_layer, batch_size', [
    (SCALAR_INPUT, 1),
    (ONE_DIM_INPUT, 128),
    (TWO_DIM_INPUT, 128),
])
def test_get_complex_value_gradients(input_layer, batch_size):
    with graph.as_default():
        machine = Linear(input_layer)
        model = Model(inputs=[input_layer], outputs=machine.predictions)
        optimizer = ComplexValuesStochasticReconfiguration(model, machine.predictions_jacobian)
        loss = Multiply()([machine.predictions, machine.predictions])
        manual_gradients_layer = Lambda(
            lambda x: tensorflow.reshape(tensorflow.reduce_sum(2.0 * x[0] * x[1], axis=0),
                                         machine.dense_layer.kernel.shape)) \
            ([machine.predictions, machine.manual_jacobian])
        manual_gradients_function = K.function(inputs=[input_layer], outputs=[manual_gradients_layer])
        complex_value_gradients_layer = Lambda(lambda x: optimizer.get_complex_value_gradients(tensorflow.real(x)))(
            loss)
        # complex_value_gradients_layer = optimizer.get_complex_value_gradients(tensorflow.real(loss))
        complex_value_gradients_function = K.function(inputs=[input_layer],
                                                      outputs=[complex_value_gradients_layer])
        sample = numpy.random.choice(2, (batch_size,) + K.int_shape(input_layer)[1:]) * 2 - 1
        complex_value_gradients = complex_value_gradients_function([sample])[0]
        manual_gradients = manual_gradients_function([sample])[0]
        diff_norm = numpy.linalg.norm(complex_value_gradients - manual_gradients)
        gradients_norm = numpy.linalg.norm(manual_gradients)
        assert (diff_norm / gradients_norm) < 1e-5


def pinv(A, b, reltol=1e-6):
    # Compute the SVD of the input matrix A
    s, u, v = tensorflow.svd(A)
    # Invert s, clear entries lower than reltol*s[0].
    atol = tensorflow.reduce_max(s) * reltol
    s = tensorflow.boolean_mask(s, s > atol)
    s_inv = tensorflow.diag(tensorflow.concat([1. / s, tensorflow.zeros([tensorflow.size(b) - tensorflow.size(s)])], 0))
    s_inv = tensorflow.cast(s_inv, A.dtype)
    # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
    return tensorflow.matmul(v, tensorflow.matmul(s_inv, tensorflow.matmul(u, tensorflow.reshape(b, [-1, 1]),
                                                                           transpose_a=True)))


@pytest.mark.parametrize('input_layer, batch_size, diag_shift', [
    (SCALAR_INPUT, 1, 0.01),
    (ONE_DIM_INPUT, 128, 0.01),
    (TWO_DIM_INPUT, 128, 0.01),
])
def test_compute_wave_function_gradient_covariance_inverse_multiplication_directly(input_layer, batch_size, diag_shift):
    with graph.as_default():
        machine = Linear(input_layer)
        model = Model(inputs=[input_layer], outputs=machine.predictions)
        optimizer = ComplexValuesStochasticReconfiguration(model, machine.predictions_jacobian, diag_shift=diag_shift)
        complex_vector_t = K.placeholder(shape=(model.count_params() // 2, 1), dtype=tensorflow.complex64)
        jacobian_minus_mean = machine.manual_jacobian - tensorflow.reduce_mean(machine.manual_jacobian, axis=0,
                                                                               keepdims=True)
        manual_s = tensorflow.eye(model.count_params() // 2, dtype=tensorflow.complex64) * diag_shift
        manual_s += tensorflow.matmul(jacobian_minus_mean, jacobian_minus_mean, adjoint_a=True) / tensorflow.cast(
            batch_size, tensorflow.complex64)
        manual_res_t = pinv(manual_s, complex_vector_t)
        res_t = optimizer.compute_wave_function_gradient_covariance_inverse_multiplication_directly(
            complex_vector_t, jacobian_minus_mean)
        res_function = K.function(inputs=[input_layer, complex_vector_t], outputs=[res_t])
        manual_res_function = K.function(inputs=[input_layer, complex_vector_t], outputs=[manual_res_t])
        sample = numpy.random.choice(2, (batch_size,) + K.int_shape(input_layer)[1:]) * 2 - 1
        real_vector = numpy.random.normal(size=(model.count_params() // 2, 1, 2))
        complex_vector = real_vector[..., 0] + 1j * real_vector[..., 1]
        res = res_function([sample, complex_vector])[0]
        manual_res = manual_res_function([sample, complex_vector])[0]
        diff_norm = numpy.linalg.norm(res - manual_res)
        res_norm = numpy.linalg.norm(manual_res)
        assert (diff_norm / res_norm) < 1e-5


@pytest.mark.parametrize('input_layer, batch_size', [
    (SCALAR_INPUT, 1),
    (ONE_DIM_INPUT, 128),
    (TWO_DIM_INPUT, 128),
])
def test_apply_complex_gradient(input_layer, batch_size):
    with graph.as_default():
        machine = Linear(input_layer)
        model = Model(inputs=[input_layer], outputs=machine.predictions)
        optimizer = ComplexValuesStochasticReconfiguration(model, machine.predictions_jacobian, lr=1.0)
        complex_vector_t = K.placeholder(shape=(model.count_params() // 2, 1), dtype=tensorflow.complex64)
        predictions_function = K.function(inputs=[input_layer], outputs=[machine.predictions])
        sample = numpy.random.choice(2, (batch_size,) + K.int_shape(input_layer)[1:]) * 2 - 1
        predictions_before = predictions_function([sample])[0]
        updates = optimizer.apply_complex_gradient(complex_vector_t)
        apply_gradients_function = K.function(inputs=[input_layer, complex_vector_t],
                                              outputs=[machine.predictions], updates=[updates])
        real_vector = numpy.random.normal(size=(model.count_params() // 2, 1, 2))
        complex_vector = real_vector[..., 0] + 1j * real_vector[..., 1]
        apply_gradients_function([sample, complex_vector])
        predictions_after = predictions_function([sample])[0]
        diff = predictions_after - predictions_before
        manual_diff = sample.reshape((batch_size, -1)) @ complex_vector
        diff_norm = numpy.linalg.norm(diff - manual_diff)
        res_norm = numpy.linalg.norm(manual_diff)
        assert (diff_norm / res_norm) < 1e-5
