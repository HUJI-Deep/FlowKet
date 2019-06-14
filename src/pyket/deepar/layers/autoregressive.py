import functools

from .lambda_with_one_to_one_topology import LambdaWithOneToOneTopology

import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Lambda


def normalize_conditional(x, norm_type=1):
    if x.dtype.is_complex:
        x_real, x_imag = tensorflow.real(x), tensorflow.imag(x)
        norm = (1.0 / norm_type) * tensorflow.reduce_logsumexp(x_real * norm_type, axis=-1, keepdims=True)
        x_real = x_real - norm
        return tensorflow.complex(x_real, x_imag)
    else:
        norm = (1.0 / norm_type) * tensorflow.reduce_logsumexp(x * norm_type, axis=-1, keepdims=True)
        return x - norm


def combine_autoregressive_conditionals(x):
    x, x_input = x[0], x[1]
    to_sum_axis = list(range(1, len(x_input.shape)))
    x = tensorflow.reduce_sum(x * x_input, axis=to_sum_axis)
    return tensorflow.reshape(x, (-1, 1))


class NormalizeConditional(LambdaWithOneToOneTopology):
    def __init__(self, norm_type=1, **kwargs):
        function = functools.partial(normalize_conditional, norm_type=norm_type)
        super(NormalizeConditional, self).__init__(function, **kwargs)


class CombineAutoregressiveConditionals(Lambda):
    def __init__(self, **kwargs):
        super(CombineAutoregressiveConditionals, self).__init__(combine_autoregressive_conditionals, **kwargs)


NormalizeConditionalProbabilities = functools.partial(NormalizeConditional, norm_type=1.0)
