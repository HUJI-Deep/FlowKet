import functools

from tensorflow.python.keras.layers import Layer


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
