import numpy
import tensorflow

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

from ..deepar.layers import LambdaWithOneToOneTopology


LOG_ZERO = -1e+30


def equal_up_down_spins_function(x):
    x, x_input = x[0], x[1]
    one_hot_input = tensorflow.one_hot((1 - tensorflow.cast(x_input, tensorflow.int32)) // 2, 2, on_value=1.0,
                                       off_value=0.0, axis=-1)
    number_of_spins = numpy.prod(K.int_shape(x_input)[1:])
    input_cumsum = tensorflow.reshape(tensorflow.cumsum(tensorflow.reshape(one_hot_input, (-1, number_of_spins, 2)),
                                                        axis=1, exclusive=True), tensorflow.shape(one_hot_input))
    mask = tensorflow.cast(tensorflow.fill(tensorflow.shape(one_hot_input), LOG_ZERO) *
                           tensorflow.cast(input_cumsum >= number_of_spins / 2.0, tensorflow.float32), x.dtype)
    return x + mask


class EqualUpDownSpins(Lambda):
    def __init__(self, **kwargs):
        super(EqualUpDownSpins, self).__init__(equal_up_down_spins_function, **kwargs)


class FlipSpins(LambdaWithOneToOneTopology):
    def __init__(self, **kwargs):
        super(FlipSpins, self).__init__(lambda x: x * tensorflow.cast(-1, dtype=x.dtype), **kwargs)
