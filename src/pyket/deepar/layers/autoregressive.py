import tensorflow


class AutoregressiveLayer(Layer):
    def __init__(self, to_type, **kwargs):
        super(CastingLayer, self).__init__(**kwargs)
        self.to_type = to_type

    def call(self, x, mask=None):
        return K.cast(x, dtype=self.to_type)

    def get_config(self):
        config = {'to_type': self.to_type}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def wave_function(x):
    x, x_input = x[0], x[1]
    one_hot_input = tensorflow.one_hot((1 - tensorflow.cast(x_input, tensorflow.int32)) // 2, 2, on_value=1.0,
                                       off_value=0.0, axis=-1)
    one_hot_input_complex = tensorflow.cast(one_hot_input, dtype=x.dtype)
    to_sum_axis = list(range(1, len(x_input.shape) + 1))
    x = tensorflow.reduce_sum(x * one_hot_input_complex, axis=to_sum_axis, keepdims=True)
    return tensorflow.reshape(x, (-1, 1))