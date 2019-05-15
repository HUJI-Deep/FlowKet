import tensorflow

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.engine.input_layer import InputLayer


class ExpandInputDim(Layer):
    def call(self, x, mask=None):
        x_layer, _, _ = x._keras_history
        assert isinstance(x_layer, InputLayer)
        return K.expand_dims(x, axis=-1)


class PeriodicPadding(Layer):
    def __init__(self, padding, **kwargs):
        super(PeriodicPadding, self).__init__(**kwargs)
        self.padding = padding

    def call(self, x, mask=None):
        """
        Create a periodic padding (wrap) around the image, to emulate periodic boundary conditions
        """
        for dim, dim_padding in enumerate(self.padding):
            if dim_padding == 0:
                continue
            x_unstacked = tensorflow.unstack(x, axis=dim + 1)
            padded_list = x_unstacked[:-dim_padding] + x_unstacked + x_unstacked[:dim_padding]
            x = tensorflow.stack(padded_list, axis=dim + 1)
        return x

    def get_config(self):
        config = {
            'padding': self.padding,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
