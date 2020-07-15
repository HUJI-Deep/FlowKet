import tensorflow as tf
from tensorflow.keras.layers import Layer


class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        self.axis = -1
        self.epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', 
                                      shape=(input_shape[-1]),
                                      initializer='ones',
                                      trainable=True)
        self.beta = self.add_weight(name='beta', 
                                      shape=(input_shape[-1]),
                                      initializer='zeros',
                                      trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        u = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=self.axis, keepdims=True)
        x = (x - u) * tf.math.rsqrt(s + self.epsilon)
        x = x * self.gamma + self.beta
        return x

    def compute_output_shape(self, input_shape):
        return input_shape