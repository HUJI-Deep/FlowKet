import tensorflow

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class Rot90(Layer):
    def __init__(self, num_of_rotations=1, **kwargs):
        super(Rot90, self).__init__(**kwargs)
        self.num_of_rotations = num_of_rotations

    def call(self, x, mask=None):
        assert len(K.int_shape(x)) == 3
        return tensorflow.image.rot90(tensorflow.expand_dims(x, axis=-1), k=self.num_of_rotations)[..., 0]
        
    def get_config(self):
        config = {'num_of_rotations': self.num_of_rotations}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FlipLeftRight(Layer):
    def __init__(self, **kwargs):
        super(FlipLeftRight, self).__init__(**kwargs)
        
    def call(self, x, mask=None):
        assert len(K.int_shape(x)) == 3
        return tensorflow.image.flip_left_right(tensorflow.expand_dims(x, axis=-1))[..., 0]