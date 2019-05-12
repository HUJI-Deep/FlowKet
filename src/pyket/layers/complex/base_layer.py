import tensorflow
import numpy as np

import tensorflow
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Initializer

# we manage both conj and normal trainable collection because normal gradient can be calculated using 
# tf.gradients(loss, tf.get_collection(CONJ_TRAINABLE_VARIABLES)) and the gradient conjugate can be calculated using
# tf.gradients(loss, tf.get_collection(NORMAL_TRAINABLE_VARIABLES))
# The reason for this is that by default if the yours variable is complex that 
# created from 2 real variables the tf.gradient calculate the gradient conjugate
# (and we can't simply use tf.conj(tf.gradients)) because we want jacobians vector products ...
CONJ_TRAINABLE_VARIABLES = tensorflow.GraphKeys.TRAINABLE_VARIABLES + '_conj'
NORMAL_TRAINABLE_VARIABLES = tensorflow.GraphKeys.TRAINABLE_VARIABLES + '_normal'
REAL_TRAINABLE_VARIABLES = tensorflow.GraphKeys.TRAINABLE_VARIABLES + '_real'
IMAG_TRAINABLE_VARIABLES = tensorflow.GraphKeys.TRAINABLE_VARIABLES + '_imag'


class NumpyInitializer(Initializer):
    def __init__(self, np_array):
        self.np_array = np_array

    def __call__(self, shape, dtype=None, partition_info=None):
        return self.np_array


class ComplexLayer(Layer):
    """docstring for ComplexLayer"""

    def __init__(self, dtype=np.complex64, **kwargs):
        super(ComplexLayer, self).__init__(dtype=dtype, **kwargs)

    def add_complex_weight(self, name, shape, complex_initializer, trainable, dtype=tensorflow.float32):
        init_with = complex_initializer(shape)
        if isinstance(init_with, tensorflow.Tensor):
            if init_with.dtype.is_complex:
                real_initializer = tensorflow.real(complex_initializer)
                imag_initializer = -1.0 * tensorflow.imag(complex_initializer)
            else:
                real_initializer = complex_initializer
                imag_initializer = -1.0 * complex_initializer
        else:
            real_initializer = NumpyInitializer(np.real(init_with))
            imag_initializer = NumpyInitializer(-1 * np.imag(init_with))
        real = self.add_weight(name='%s_real' % name,
                               shape=shape, initializer=real_initializer, dtype=dtype, trainable=trainable)
        imag = self.add_weight(name='%s_imag' % name,
                               shape=shape, initializer=imag_initializer, dtype=dtype, trainable=trainable)
        minus_imag = tensorflow.multiply(imag, -1., 'conj_imag')
        if trainable and real not in tensorflow.get_collection(REAL_TRAINABLE_VARIABLES):
            tensorflow.add_to_collection(REAL_TRAINABLE_VARIABLES, real)
            tensorflow.add_to_collection(CONJ_TRAINABLE_VARIABLES, real)
            tensorflow.add_to_collection(NORMAL_TRAINABLE_VARIABLES, real)
            tensorflow.add_to_collection(IMAG_TRAINABLE_VARIABLES, imag)
            tensorflow.add_to_collection(CONJ_TRAINABLE_VARIABLES, imag)
            tensorflow.add_to_collection(NORMAL_TRAINABLE_VARIABLES, minus_imag)
        return tensorflow.complex(real, minus_imag)
