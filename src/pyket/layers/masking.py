import functools

import tensorflow as tf

from tensorflow.keras.layers import Layer


def shift(x, axis):
	begin = [0] * len(x.shape)
	size = [-1] * len(x.shape)
	size[1] = int(x.get_shape()[axis]) - 1 
	upper_slice = tf.slice(x, begin, size)
	size[1] = 1
	lower_slice = tf.slice(x, begin, size)
	return tf.concat([tf.zeros_like(lower_slice, dtype=x.dtype), upper_slice], axis=axis)
	
down_shift = functools.partial(shift, axis=1)
right_shift = functools.partial(shift, axis=2)


class DownShiftLayer(Layer):
    def call(self, x, mask=None):
        return down_shift(x)

class RightShiftLayer(Layer):
    def call(self, x, mask=None):
        return right_shift(x)
