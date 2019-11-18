from .simple_models import Linear, LinearDepthTwo

import numpy
import pytest
import tensorflow

from tensorflow.keras.layers import Input, Multiply, Lambda
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K

from flowket.optimizers import ComplexValuesOptimizer

DEFAULT_TF_GRAPH = tensorflow.get_default_graph()
ONE_DIM_INPUT = Input(shape=(16,), dtype='int8')
SCALAR_INPUT = Input(shape=(1,), dtype='int8')
TWO_DIM_INPUT = Input(shape=(4, 4), dtype='int8')


@pytest.mark.parametrize('input_layer, machine_class, batch_size', [
    (ONE_DIM_INPUT, Linear, 128),
    (ONE_DIM_INPUT, LinearDepthTwo, 128),  # In this case the gradients has non vanished imaginary part
    (TWO_DIM_INPUT, Linear, 128),
    (TWO_DIM_INPUT, LinearDepthTwo, 128),
])
def test_get_predictions_jacobian(input_layer, machine_class, batch_size):
    with DEFAULT_TF_GRAPH.as_default():
        machine = machine_class(input_layer)
        model = Model(inputs=[input_layer], outputs=machine.predictions)
        optimizer = ComplexValuesOptimizer(model, machine.predictions_jacobian)
        jacobian_function = K.function(inputs=[input_layer], outputs=[optimizer.get_predictions_jacobian()])
        manual_jacobian_function = K.function(inputs=[input_layer], outputs=[machine.manual_jacobian])
        sample = numpy.random.choice(2, (batch_size,) + K.int_shape(input_layer)[1:]) * 2 - 1
        jacobian = jacobian_function([sample])[0]
        manual_jacobian = manual_jacobian_function([sample])[0]
        diff_norm = numpy.linalg.norm(jacobian - manual_jacobian, 'fro')
        jacobian_norm = numpy.linalg.norm(manual_jacobian, 'fro')
        assert (diff_norm / jacobian_norm) < 1e-5


@pytest.mark.parametrize('input_layer, batch_size, conjugate_gradients', [
    (SCALAR_INPUT, 1, False),
    (ONE_DIM_INPUT, 128, False),
    (TWO_DIM_INPUT, 128, False),
    (SCALAR_INPUT, 1, True),
    (ONE_DIM_INPUT, 128, True),
    (TWO_DIM_INPUT, 128, True),
])
def test_get_complex_value_gradients(input_layer, batch_size, conjugate_gradients):
    with DEFAULT_TF_GRAPH.as_default():
        machine = Linear(input_layer)
        model = Model(inputs=[input_layer], outputs=machine.predictions)
        optimizer = ComplexValuesOptimizer(model, machine.predictions_jacobian)
        loss = Multiply()([machine.predictions, machine.predictions])
        manual_gradients_layer = Lambda(
            lambda x: tensorflow.reshape(tensorflow.reduce_sum(2.0 * x[0] * x[1], axis=0),
                                         machine.dense_layer.kernel.shape)) \
            ([machine.predictions, machine.manual_jacobian])
        if conjugate_gradients:
            manual_gradients_layer = Lambda(lambda x: tensorflow.conj(x))(manual_gradients_layer)
        manual_gradients_function = K.function(inputs=[input_layer], outputs=[manual_gradients_layer])
        complex_value_gradients_layer = Lambda(
            lambda x: optimizer.get_model_parameters_complex_value_gradients(
                tensorflow.real(x), conjugate_gradients=conjugate_gradients))(loss)
        complex_value_gradients_function = K.function(inputs=[input_layer],
                                                      outputs=[complex_value_gradients_layer])
        sample = numpy.random.choice(2, (batch_size,) + K.int_shape(input_layer)[1:]) * 2 - 1
        complex_value_gradients = complex_value_gradients_function([sample])[0]
        manual_gradients = manual_gradients_function([sample])[0]
        diff_norm = numpy.linalg.norm(complex_value_gradients - manual_gradients)
        gradients_norm = numpy.linalg.norm(manual_gradients)
        assert (diff_norm / gradients_norm) < 1e-5


@pytest.mark.parametrize('input_layer, batch_size', [
    (SCALAR_INPUT, 1),
    (ONE_DIM_INPUT, 128),
    (TWO_DIM_INPUT, 128),
])
def test_apply_complex_gradient(input_layer, batch_size):
    with DEFAULT_TF_GRAPH.as_default():
        machine = Linear(input_layer)
        model = Model(inputs=[input_layer], outputs=machine.predictions)
        optimizer = ComplexValuesOptimizer(model, machine.predictions_jacobian, lr=1.0)
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
