import functools

import tensorflow
from tensorflow.keras.layers import Layer, Lambda


def plus_minus_one_to_one_hot(x):
    return tensorflow.one_hot((1 - tensorflow.cast(x, tensorflow.int32)) // 2, 2, on_value=1.0,
                              off_value=0.0, axis=-1)


def to_one_hot(x, num_of_categories=2):
    return tensorflow.one_hot(tensorflow.cast(x, tensorflow.int32), num_of_categories, on_value=1.0,
                              off_value=0.0, axis=-1)


class ToOneHot(Lambda):
    def __init__(self, num_of_categories=2, **kwargs):
        super(ToOneHot, self).__init__(functools.partial(to_one_hot, num_of_categories=num_of_categories), **kwargs)


class PlusMinusOneToOneHot(Lambda):
    def __init__(self, **kwargs):
        super(PlusMinusOneToOneHot, self).__init__(plus_minus_one_to_one_hot, **kwargs)
