import abc
import functools

import tensorflow

from .base_layer import ComplexLayer
from .tensorflow_ops import keras_conv_to_complex_conv
from ...deepar.graph_analysis.convolutional_topology import TopologyManager, ConvolutionalTopology


def normalize_tuple(value, n, name):
    """Transforms a single int or iterable of ints into an int tuple.
    # Arguments
        value: The value to validate and convert. Could be an int, or any iterable
          of ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. `strides` or
          `kernel_size`. This is only used to format error messages.
    # Returns
        A tuple of n integers.
    # Raises
        ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except ValueError:
                raise ValueError('The `' + name + '` argument must be a tuple of ' +
                                 str(n) + ' integers. Received: ' + str(value) + ' '
                                                                                 'including element ' + str(
                    single_value) + ' of '
                                    'type ' + str(type(single_value)))
    return value_tuple


def normalize_padding(value):
    padding = value.lower()
    allowed = {'valid', 'same'}
    if padding not in allowed:
        raise ValueError('The `padding` argument must be one of "valid", "same" '
                         '(or "causal" for Conv1D). Received: ' + str(padding))
    return padding


class _ComplexConv(ComplexLayer, abc.ABC):

    def __init__(self, rank, filters, kernel_size, strides=1, padding='valid', dilation_rate=1,
                 kernel_initializer='glorot_normal', bias_initializer='zeros',
                 activation=None, use_bias=True, **kwargs):
        super(_ComplexConv, self).__init__(**kwargs)
        self.rank = rank
        self.kernel_size = normalize_tuple(kernel_size, rank,
                                           'kernel_size')
        self.strides = normalize_tuple(strides, rank, 'strides')
        self.padding = normalize_padding(padding)
        self.dilation_rate = normalize_tuple(dilation_rate, rank,
                                             'dilation_rate')
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.activation = activation

    @property
    @abc.abstractmethod
    def real_conv_op(self):
        pass

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_complex_weight('kernel',
                                              self.kernel_size + (int(input_shape[-1]),
                                                                  self.filters),
                                              self.kernel_initializer, True, self.params_dtype)
        if self.use_bias:
            self.bias = self.add_complex_weight('bias', (self.filters,),
                                                self.bias_initializer, True,
                                                self.params_dtype)
        super().build(input_shape)

    def call(self, x):
        dilation_rate = self.dilation_rate
        if self.rank == 1:
            dilation_rate = self.dilation_rate[0]

        res = keras_conv_to_complex_conv(x, self.kernel(),
                                         functools.partial(self.real_conv_op,
                                                           strides=self.strides,
                                                           dilation_rate=dilation_rate,
                                                           padding=self.padding))
        if self.use_bias:
            res = tensorflow.nn.bias_add(res, self.bias())
        if self.activation is not None:
            return self.activation(res)
        return res


class ComplexConv1D(_ComplexConv):

    @property
    def real_conv_op(self):
        return tensorflow.keras.backend.conv1d

    def __init__(self, filters, kernel_size, **kwargs):
        super(ComplexConv1D, self).__init__(1, filters, kernel_size, **kwargs)


class ComplexConv2D(_ComplexConv):

    @property
    def real_conv_op(self):
        return tensorflow.keras.backend.conv2d

    def __init__(self, filters, kernel_size, **kwargs):
        super(ComplexConv2D, self).__init__(2, filters, kernel_size, **kwargs)


class ComplexConv3D(_ComplexConv):

    @property
    def real_conv_op(self):
        return tensorflow.keras.backend.conv3d

    def __init__(self, filters, kernel_size, **kwargs):
        super(ComplexConv3D, self).__init__(3, filters, kernel_size, **kwargs)


class ComplexConvolutionalTopology(ConvolutionalTopology):
    def _prepare_weights(self):
        self.reshaped_weights = tensorflow.reshape(self.layer.kernel(), [-1, self.layer.filters])
        if self.layer.use_bias:
            self.bias = self.layer.bias()


TopologyManager().register_layer_topology(ComplexConv1D, ComplexConvolutionalTopology)
TopologyManager().register_layer_topology(ComplexConv2D, ComplexConvolutionalTopology)
TopologyManager().register_layer_topology(ComplexConv3D, ComplexConvolutionalTopology)
