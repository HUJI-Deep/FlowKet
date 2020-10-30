import tensorflow
from tensorflow.python.keras.layers import Layer


class GatherLayer(Layer):
    def __init__(self, indices, axis=1, **kwargs):
        super(GatherLayer, self).__init__(**kwargs)
        self.indices = indices
        self.axis = axis
        self.indices_tensor = tensorflow.convert_to_tensor(self.indices)
    
    def call(self, x, mask=None):
        return tensorflow.gather(x, self.indices_tensor, axis=self.axis)

    def get_config(self):
        config = {'indices': self.indices, 'axis': self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
