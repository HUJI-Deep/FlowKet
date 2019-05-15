import functools

import tensorflow
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


class VectorToComplexNumber(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(VectorToComplexNumber, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x, mask=None):
        axis = len(K.int_shape(x)) -1 if self.axis == -1 else self.axis
        assert K.int_shape(x)[axis] % 2 == 0
        orig_shape = K.int_shape(x)
        new_shape = list(orig_shape[:axis] + (2, orig_shape[axis] // 2) + orig_shape[axis + 1:])
        new_shape[0] = -1
        x = tensorflow.reshape(x, shape=new_shape)
        real, imag = tensorflow.unstack(x, axis=axis)
        return tensorflow.complex(real, imag)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CastingLayer(Layer):
    def __init__(self, to_type, **kwargs):
        super(CastingLayer, self).__init__(**kwargs)
        self.to_type = to_type

    def call(self, x, mask=None):
        return K.cast(x, dtype=self.to_type)

    def get_config(self):
        config = {'to_type': self.to_type}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


ToFloat32 = functools.partial(CastingLayer, 'float32')
ToFloat64 = functools.partial(CastingLayer, 'float64')
ToComplex64 = functools.partial(CastingLayer, 'complex64')
ToComplex128 = functools.partial(CastingLayer, 'complex128')