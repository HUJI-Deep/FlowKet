import itertools
import functools

from ..layers import Rot90, FlipLeftRight, Roll
from ..layers.complex.tensorflow_ops import angle, complex_log
import numpy
import tensorflow

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Concatenate


def probabilistic_ensemble_op(x, ensemble_size):
    new_real = 0.5 * tensorflow.math.reduce_logsumexp(tensorflow.math.real(x) * 2.0, axis=-1,
                                                 keepdims=True) - 0.5 * numpy.log(ensemble_size)
    new_imag = angle(tensorflow.math.reduce_mean(
        tensorflow.math.exp(tensorflow.complex(tensorflow.zeros_like(new_real), tensorflow.math.imag(x))), axis=-1,
        keepdims=True))
    return tensorflow.complex(new_real, new_imag)


def average_ensemble_op(x):
    return complex_log(tensorflow.math.reduce_mean(tensorflow.math.exp(x), axis=-1, keepdims=True))


def build_ensemble(predictions, probabilistic=True):
    joint_predictions = Concatenate(axis=-1)(predictions)
    ensemble = functools.partial(probabilistic_ensemble_op, 
        ensemble_size=len(predictions)) if probabilistic else average_ensemble_op
    return Lambda(ensemble)(joint_predictions)


def build_symmetrization_ensemble(symmetrization_inputs, predictions_model, probabilistic=True):
    shape = K.int_shape(symmetrization_inputs[0])[1:]
    joint_inputs = Lambda(lambda x:tensorflow.reshape(tensorflow.stack(x, axis=1), (-1, ) + shape))(symmetrization_inputs)
    joint_predictions = predictions_model(joint_inputs)
    reshaped_predictions = Lambda(lambda x:tensorflow.reshape((-1, len(symmetrization_inputs))))(joint_predictions)
    ensemble = functools.partial(probabilistic_ensemble_op, 
        ensemble_size=len(inputs)) if probabilistic else average_ensemble_op
    return Lambda(ensemble)(reshaped_predictions)


def make_2d_obc_invariants(keras_input_layer, predictions_model, probabilistic=True):
    inputs = [keras_input_layer,
              Rot90(num_of_rotations=1)(keras_input_layer),
              Rot90(num_of_rotations=2)(keras_input_layer),
              Rot90(num_of_rotations=3)(keras_input_layer)]
    inputs = inputs + [FlipLeftRight()(x) for x in inputs]
    predictions = [predictions_model(x) for x in inputs]
    ensemble_prediction = build_ensemble(predictions, probabilistic)
    return Model(inputs=keras_input_layer, outputs=ensemble_prediction)


def make_pbc_invariants(keras_input_layer, predictions_model, apply_also_obc_invariants=True, probabilistic=True):
    shape = K.int_shape(keras_input_layer)[1:]
    if apply_also_obc_invariants:
        obc_input = Input(shape=shape, dtype=keras_input_layer.dtype)
        predictions_model = make_2d_obc_invariants(obc_input, predictions_model, probabilistic)
    inputs = [Roll(list(i))(keras_input_layer) for i in itertools.product(*[range(dim_size) for dim_size in shape])]
    ensemble_prediction = build_symmetrization_ensemble(inputs, predictions_model, probabilistic=probabilistic)
    return Model(inputs=keras_input_layer, outputs=ensemble_prediction)


def make_up_down_invariant(keras_input_layer, predictions_model, probabilistic=True):
    shape = K.int_shape(keras_input_layer)[1:]
    inputs = [keras_input_layer, Lambda(lambda x:x * -1)(keras_input_layer)]
    ensemble_prediction = build_symmetrization_ensemble(inputs, predictions_model, probabilistic=probabilistic)
    return Model(inputs=keras_input_layer, outputs=ensemble_prediction)