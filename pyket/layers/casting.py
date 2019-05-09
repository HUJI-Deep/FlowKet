import functools

import tensorflow

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class VectorToComplexNumber(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(VectorToComplexNumber, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x, mask=None):
        assert K.int_shape(x)[self.axis] == 2
        real, imag = tensorflow.unstack(x, axis=self.axis)
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