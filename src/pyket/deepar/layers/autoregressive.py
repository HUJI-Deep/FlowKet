import functools

import tensorflow
from tensorflow.keras.layers import Lambda


def normalize_in_log_space(x, norm_type=1):
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


class NormalizeInLogSpace(Lambda):
    def __init__(self, norm_type=1.0, **kwargs):
        function = functools.partial(normalize_in_log_space, norm_type=norm_type)
        super(NormalizeInLogSpace, self).__init__(function, **kwargs)


class CombineAutoregressiveConditionals(Lambda):
    def __init__(self, **kwargs):
        super(CombineAutoregressiveConditionals, self).__init__(combine_autoregressive_conditionals, **kwargs)


NormalizeConditionalProbabilities = functools.partial(NormalizeInLogSpace, norm_type=1.0)
