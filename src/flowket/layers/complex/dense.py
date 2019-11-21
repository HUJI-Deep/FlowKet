import itertools

import numpy

import tensorflow
from tensorflow.python.keras import backend as K

from .base_layer import ComplexLayer


class ComplexDense(ComplexLayer):
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='complex_glorot', bias_initializer='zeros',
                 multiply_units_by_input_dim=False, **kwargs):
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.multiply_units_by_input_dim = multiply_units_by_input_dim

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = int(input_shape[-1])
        units = self.units
        if self.multiply_units_by_input_dim:
            units *= input_dim
        self.kernel = self.add_complex_weight(name='kernel',
                                              shape=(input_dim, units),
                                              complex_initializer=self.kernel_initializer,
                                              trainable=True,
                                              dtype=self.params_dtype)
        if self.use_bias:
            self.bias = self.add_complex_weight(name='bias',
                                                shape=(units,),
                                                complex_initializer=self.bias_initializer,
                                                trainable=True,
                                                dtype=self.params_dtype)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class TranslationInvariantComplexDense(ComplexDense):
    def __init__(self, units, **kwargs):
        super(TranslationInvariantComplexDense, self).__init__(units,
                                                               multiply_units_by_input_dim=True, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dims = tuple([int(s) for s in input_shape[1:]])
        self.number_of_visible = numpy.prod(input_dims)
        self.bare_kernel = self.add_complex_weight(name='kernel',
                                                   shape=input_dims + (self.units,),
                                                   complex_initializer=self.kernel_initializer,
                                                   trainable=True,
                                                   dtype=self.params_dtype)

        all_axes = tuple(list(range(len(input_dims))))
        kernel_translations = [tensorflow.manip.roll(self.bare_kernel, i, all_axes) for i in
                               itertools.product(*[range(dim_size) for dim_size in input_dims])]
        self.kernel = tensorflow.reshape(tensorflow.stack(kernel_translations, axis=-1), (self.number_of_visible, -1))

        if self.use_bias:
            self.bare_bias = self.add_complex_weight(name='bias',
                                                     shape=(self.units,),
                                                     complex_initializer=self.bias_initializer,
                                                     trainable=True,
                                                     dtype=self.params_dtype)
            self.bias = tensorflow.reshape(tensorflow.stack([self.bare_bias] * self.number_of_visible, axis=-1), (-1,))
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        return super().call(tensorflow.reshape(inputs, (-1, self.number_of_visible)))
