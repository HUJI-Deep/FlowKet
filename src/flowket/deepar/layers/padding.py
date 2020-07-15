import tensorflow

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class ExpandInputDim(Layer):
    def call(self, x, mask=None):
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
            if isinstance(dim_padding, tuple):
                before, after = dim_padding
            else:
                before, after = dim_padding, dim_padding
            x_unstacked = tensorflow.unstack(x, axis=dim + 1)
            padded_list = []
            if before > 0:
                padded_list += x_unstacked[-before:]
            padded_list += x_unstacked
            if after > 0:
                padded_list += x_unstacked[:after]
            x = tensorflow.stack(padded_list, axis=dim + 1)
        return x

    def get_config(self):
        config = {
            'padding': self.padding,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
