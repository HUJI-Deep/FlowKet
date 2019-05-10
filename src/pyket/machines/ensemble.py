import itertools

from ..layers import Rot90, FlipLeftRight, Roll
from ..layers.complex.tensorflow_ops import angle
import numpy
import tensorflow

from keras.models import Model
from tensorflow.keras.layers import Lambda, Input


def build_ensemble(predictions):
    joint_predictions = Concatenate(axis=-1)(predictions)
    def inner(x):
        new_real = 0.5 * tensorflow.reduce_logsumexp(tensorflow.real(x) * 2.0, axis=-1, keepdims=True) - 0.5 * numpy.log(len(predictions))
        new_imag = angle(tensorflow.reduce_mean(tensorflow.exp(tensorflow.complex(tensorflow.zeros_like(new_real), tensorflow.imag(x))), axis=-1, keepdims=True))
        return tensorflow.complex(new_real, new_imag)
    return Lambda(inner)(joint_predictions)


def make_obc_invariants(keras_input_layer, predictions_model):
    inputs = [keras_input_layer, 
              Rot90(k=1)(keras_input_layer),
              Rot90(k=2)(keras_input_layer),
              Rot90(k=3)(keras_input_layer)]
    inputs = inputs + [FlipLeftRight(x) for x in inputs]
    predictions = [predictions_model(x) for x in inputs]
    ensemble_prediction = build_ensemble(predictions)
    return Model(inputs=keras_input_layer, outputs=ensemble_prediction)


def make_pbc_invariants(keras_input_layer, predictions_model, apply_also_obc_invariants=True):
    if apply_also_obc_invariants:
        shape = K.int_shape(keras_input_layer)[1:]
        obc_input = Input(shape=shape, dtype=keras_input_layer.dtype)
        predictions_model = make_obc_invariants(obc_input, predictions_model)
    inputs = [Roll(list(i))(keras_input_layer) for i in itertools.product(*[range(dim_size) for dim_size in shape])]
    ensemble_prediction = build_ensemble([predictions_model(x) for x in inputs])
    return Model(inputs=keras_input_layer, outputs=ensemble_prediction)