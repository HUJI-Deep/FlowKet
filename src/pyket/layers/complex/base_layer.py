import numpy as np
import tensorflow
from tensorflow.keras.layers import Layer

from .initializers import get as get_initializer
from .initializers import ConjugateDecorator


class ComplexLayer(Layer):
    """docstring for ComplexLayer"""

    def __init__(self, dtype=np.complex64, **kwargs):
        super(ComplexLayer, self).__init__(dtype=dtype, **kwargs)
        self.weights_for_complex_value_params_gradient_conjugate = []
        self.real_weights = []
        self.imag_weights = []
        self.params_dtype = tensorflow.float64 if hasattr(self, 'dtype') and self.dtype == tensorflow.complex128 \
            else tensorflow.float32

    def add_complex_weight(self, name, shape, complex_initializer, trainable=True, dtype=tensorflow.float32):
        complex_initializer = ConjugateDecorator(get_initializer(complex_initializer))
        real = self.add_weight(name='%s_real' % name,
                               shape=shape,
                               initializer=complex_initializer.get_real_part_initializer(),
                               dtype=dtype,
                               trainable=trainable)
        imag = self.add_weight(name='%s_imag' % name,
                               shape=shape,
                               initializer=complex_initializer.get_imag_part_initializer(),
                               dtype=dtype,
                               trainable=trainable)
        minus_imag = tensorflow.multiply(imag, -1., 'conj_imag')
        self.real_weights.append(real)
        self.imag_weights.append(imag)
        self.weights_for_complex_value_params_gradient_conjugate.append(real)
        self.weights_for_complex_value_params_gradient_conjugate.append(minus_imag)
        return tensorflow.complex(real, minus_imag)
