import tensorflow

from .base_layer import ComplexLayer
from .initializers import complex_independent_filters
from .tensorflow_ops import conv2d_complex


class ComplexConv2D(ComplexLayer):

    def __init__(self, filters, kernel_size, strides=(1,1), use_bias=True, padding='valid',
                 kernel_initializer='complex_independent_filters', bias_initializer='zeros', 
                 activation=None, **kwargs):
        super(ComplexConv2D, self).__init__(**kwargs)
        if initializer == 'complex_independent_filters':
            initializer = complex_independent_filters()
        self.kernel_size = kernel_size
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.params_dtype = tensorflow.float64 if hasattr(self, 'dtype') and self.dtype == tensorflow.complex128 else tensorflow.float32
        self.use_bias = use_bias
        self.strides = [1, strides[0], strides[1], 1]
        self.padding = padding
        self.activation = activation

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_complex_weight('kernel',
            (self.kernel_size[0], self.kernel_size[1], int(input_shape[-1]), self.filters),
            self.kernel_initializer, True, self.params_dtype)
        if self.use_bias:
            self.bias = self.add_complex_weight('bias', (self.filters, ),
                                              self.bias_initializer, True,
                                              self.params_dtype)
        super(ComplexConv2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        padding = self.padding
        actual_kernel = self.kernel
        res = conv2d_complex(x, actual_kernel, self.strides, padding)
        if self.use_bias:
            res = tensorflow.nn.bias_add(res, self.bias)
        if self.activation is not None:
            return self.activation(res)
        return res

