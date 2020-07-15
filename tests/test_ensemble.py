import itertools

import tensorflow.keras.backend as K
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from flowket.machines.ensemble import make_2d_obc_invariants, make_pbc_invariants, build_ensemble, build_symmetrization_ensemble

from .simple_models import real_values_2d_model, real_values_1d_model

DEFAULT_TF_GRAPH = tf.get_default_graph()


def transform_sample(sample, num_of_rotations, flip):
    if num_of_rotations > 0:
        sample = np.rot90(sample, k=num_of_rotations, axes=(1, 2))
    if flip:
        sample = np.flip(sample, axis=2)
    return sample


def roll_sample(sample, roll_for_axis):
    return np.roll(sample, roll_for_axis, tuple(range(1, len(roll_for_axis))))


@pytest.mark.parametrize('model_builder, batch_size', [
    (real_values_2d_model, 100),
])
def test_make_2d_obc_invariants(model_builder, batch_size):
    with DEFAULT_TF_GRAPH.as_default():
        keras_model = model_builder()
        keras_model.summary()

        shape = K.int_shape(keras_model.input)[1:]
        obc_input = Input(shape=shape, dtype=keras_model.input.dtype)
        invariant_model = make_2d_obc_invariants(obc_input, keras_model)
        invariant_model_func = K.function(inputs=[obc_input], outputs=[invariant_model.output])

        size = (batch_size,) + K.int_shape(keras_model.input)[1:]
        batch = np.random.rand(*size)
        batch_transformations = [batch,
                                 transform_sample(batch, num_of_rotations=1, flip=False),
                                 transform_sample(batch, num_of_rotations=2, flip=False),
                                 transform_sample(batch, num_of_rotations=3, flip=False),
                                 transform_sample(batch, num_of_rotations=0, flip=True),
                                 transform_sample(batch, num_of_rotations=1, flip=True),
                                 transform_sample(batch, num_of_rotations=2, flip=True),
                                 transform_sample(batch, num_of_rotations=3, flip=True)
                                 ]
        vals = [invariant_model_func([transformation])[0] for transformation in batch_transformations]
        allclose = [np.allclose(vals[0], another_val, rtol=1e-3) for another_val in vals[1:]]
        assert np.all(allclose)


@pytest.mark.parametrize('model_builder, batch_size', [
    (real_values_2d_model, 5),
    (real_values_1d_model, 5),
])
def test_make_pbc_invariants(model_builder, batch_size):
    with DEFAULT_TF_GRAPH.as_default():
        keras_model = model_builder()
        keras_model.summary()

        shape = K.int_shape(keras_model.input)[1:]
        pbc_input = Input(shape=shape, dtype=keras_model.input.dtype)
        invariant_model = make_pbc_invariants(pbc_input, keras_model, apply_also_obc_invariants=False)
        invariant_model_func = K.function(inputs=[pbc_input], outputs=[invariant_model.output])

        size = (batch_size,) + K.int_shape(keras_model.input)[1:]
        batch = np.random.rand(*size)
        batch_transformations = [roll_sample(batch, i) for i in
                                 itertools.product(*[range(dim_size) for dim_size in shape])]

        vals = [invariant_model_func([transformation])[0] for transformation in batch_transformations]
        allclose = [np.allclose(vals[0], another_val, rtol=1e-3) for another_val in vals[1:]]
        assert np.all(allclose)


@pytest.mark.parametrize('model_builder, batch_size', [
    (real_values_2d_model, 5),
    (real_values_1d_model, 5),
])
def test_build_symmetrization_ensemble(model_builder, batch_size):
    with DEFAULT_TF_GRAPH.as_default():
        keras_model = model_builder()
        keras_model.summary()

        shape = K.int_shape(keras_model.input)[1:]
        symmetrization_input = Input(shape=shape, dtype=keras_model.input.dtype)
        ensemble_input = Input(shape=shape, dtype=keras_model.input.dtype)

        symmetrization_model = Model(inputs=symmetrization_input, outputs=build_symmetrization_ensemble([symmetrization_input, Lambda(lambda x:x * -1)(symmetrization_input)], keras_model))
        ensemble_model = Model(inputs=ensemble_input, outputs=build_ensemble([keras_model(ensemble_input), keras_model(Lambda(lambda x:x * -1)(ensemble_input))])) 
        
        symmetrization_model_func = K.function(inputs=[symmetrization_input], outputs=[symmetrization_model.output])
        ensemble_model_func = K.function(inputs=[ensemble_input], outputs=[ensemble_model.output])

        size = (batch_size,) + K.int_shape(keras_model.input)[1:]
        batch = np.random.rand(*size)

        symmetrization_model_vals = symmetrization_model_func([batch])[0]
        ensemble_model_vals = ensemble_model_func([batch])[0]
        assert np.allclose(symmetrization_model_vals, ensemble_model_vals, rtol=1e-3)
