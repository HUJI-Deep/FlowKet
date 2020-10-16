import numpy as np
import pytest
import tensorflow
import tensorflow as tf
from tensorflow.python.ops.parallel_for import gradients
import tensorflow.keras.backend as K

from flowket.utils.jacobian import gradient_per_example
from flowket.utils.v2_fake_graph_context import Ctx

from .simple_models import real_values_2d_model, complex_values_2d_model, real_values_1d_model, complex_values_1d_model

if not tensorflow.__version__.startswith('2'):
    DEFAULT_TF_GRAPH = tf.get_default_graph()


@pytest.mark.parametrize('model_builder, batch_size', [
    (real_values_2d_model, 5),
    (complex_values_2d_model, 5),
    (real_values_1d_model, 5),
    (complex_values_1d_model, 5),
])
def test_equal_to_builtin_jacobian(model_builder, batch_size):
    if tensorflow.__version__.startswith('2'):
        ctx = Ctx()
    else:
        ctx = DEFAULT_TF_GRAPH.as_default()
    with ctx:
        keras_model = model_builder()
        keras_model.summary()
        gradient_per_example_t = gradient_per_example(tf.math.real(keras_model.output), keras_model)
        tensorflow_jacobian_t = gradients.jacobian(tf.math.real(keras_model.output),
                                                   keras_model.weights, use_pfor=False)
        print(gradient_per_example_t)
        print(tensorflow_jacobian_t)
        gradient_per_example_func = K.function(inputs=[keras_model.input], outputs=gradient_per_example_t)
        tensorflow_jacobian_func = K.function(inputs=[keras_model.input], outputs=tensorflow_jacobian_t)
        size = (batch_size,) + K.int_shape(keras_model.input)[1:]
        batch = np.random.rand(*size)
        gradient_per_example_vals = gradient_per_example_func([batch])
        tensorflow_jacobian_vals = tensorflow_jacobian_func([batch])
        allclose = [np.allclose(a, b, rtol=1e-3) for a, b in zip(gradient_per_example_vals, tensorflow_jacobian_vals)]
        assert np.all(allclose)
