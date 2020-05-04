from .simple_models import Linear, LinearDepthTwo

import numpy
import pytest
import tensorflow

from tensorflow.keras.layers import Input, Multiply, Lambda
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K

from flowket.optimizers import ComplexValuesStochasticReconfiguration

DEFAULT_TF_GRAPH = tensorflow.get_default_graph()
ONE_DIM_INPUT = Input(shape=(16,), dtype='int8')
SCALAR_INPUT = Input(shape=(1,), dtype='int8')
TWO_DIM_INPUT = Input(shape=(4, 4), dtype='int8')


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


@pytest.mark.parametrize('input_layer, batch_size, diag_shift, iterative', [
    (SCALAR_INPUT, 1, 0.01, False),
    (ONE_DIM_INPUT, 128, 0.01, False),
    (TWO_DIM_INPUT, 128, 0.01, False),
    (ONE_DIM_INPUT, 128, 0.01, True),
    (TWO_DIM_INPUT, 128, 0.01, True),
])
def test_compute_wave_function_gradient_covariance_inverse_multiplication(input_layer, batch_size, diag_shift,
                                                                          iterative):
    with DEFAULT_TF_GRAPH.as_default():
        machine = Linear(input_layer)
        model = Model(inputs=[input_layer], outputs=machine.predictions)
        if tensorflow.__version__ >= '1.14':
            optimizer = ComplexValuesStochasticReconfiguration(model, machine.predictions_jacobian, diag_shift=diag_shift,
                                                           conjugate_gradient_tol=1e-6,
                                                           iterative_solver=iterative,
                                                           iterative_solver_max_iterations=None, name='optimizer')
        else:
            optimizer = ComplexValuesStochasticReconfiguration(model, machine.predictions_jacobian, diag_shift=diag_shift,
                                                               conjugate_gradient_tol=1e-6,
                                                               iterative_solver=iterative,
                                                               iterative_solver_max_iterations=None)
        complex_vector_t = K.placeholder(shape=(model.count_params() // 2, 1), dtype=tensorflow.complex64)
        jacobian_minus_mean = machine.manual_jacobian - tensorflow.reduce_mean(machine.manual_jacobian, axis=0,
                                                                               keepdims=True)
        manual_s = tensorflow.eye(model.count_params() // 2, dtype=tensorflow.complex64) * diag_shift
        manual_s += tensorflow.matmul(jacobian_minus_mean, jacobian_minus_mean, adjoint_a=True) / tensorflow.cast(
            batch_size, tensorflow.complex64)
        manual_res_t = pinv(manual_s, complex_vector_t)
        res_t = optimizer.compute_wave_function_gradient_covariance_inverse_multiplication(complex_vector_t,
                                                                                           jacobian_minus_mean)
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


@pytest.mark.parametrize('input_layer, batch_size, diag_shift', [
    (SCALAR_INPUT, 1, 0.01),
    (ONE_DIM_INPUT, 128, 0.01),
    (TWO_DIM_INPUT, 128, 0.01),
])
def test_stochastic_reconfiguration_matrix_vector_product_via_jvp(input_layer, batch_size, diag_shift):
    with DEFAULT_TF_GRAPH.as_default():
        machine = Linear(input_layer)
        model = Model(inputs=[input_layer], outputs=machine.predictions)
        if tensorflow.__version__ >= '1.14':
            optimizer = ComplexValuesStochasticReconfiguration(model, machine.predictions_jacobian, diag_shift=diag_shift,
                                                               conjugate_gradient_tol=1e-6,
                                                               iterative_solver_max_iterations=None, name='optimizer')
        else:
            optimizer = ComplexValuesStochasticReconfiguration(model, machine.predictions_jacobian, diag_shift=diag_shift,
                                                               conjugate_gradient_tol=1e-6,
                                                               iterative_solver_max_iterations=None)
        complex_vector_t = K.placeholder(shape=(model.count_params() // 2, 1), dtype=tensorflow.complex64)
        jacobian_minus_mean = machine.manual_jacobian - tensorflow.reduce_mean(machine.manual_jacobian, axis=0,
                                                                               keepdims=True)
        manual_s = tensorflow.eye(model.count_params() // 2, dtype=tensorflow.complex64) * diag_shift
        manual_s += tensorflow.matmul(jacobian_minus_mean, jacobian_minus_mean, adjoint_a=True) / tensorflow.cast(
            batch_size, tensorflow.complex64)
        manual_res_t = tensorflow.matmul(manual_s, complex_vector_t)
        res_t = optimizer.get_stochastic_reconfiguration_matrix_vector_product_via_jvp(complex_vector_t)
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
