import numpy

import tensorflow
from tensorflow.keras.layers import Input, Flatten, Lambda, Concatenate

from pyket.layers import ComplexDense, ToComplex64
from pyket.machines import Machine


class Linear(Machine):
    """docstring for Linear"""

    def __init__(self, keras_input_layer, **kwargs):
        super(Linear, self).__init__(keras_input_layer, **kwargs)
        x = ToComplex64()(keras_input_layer)
        x = Flatten()(x)
        self.manual_jacobian = x
        self.dense_layer = ComplexDense(1, use_bias=False)
        self._predictions = self.dense_layer(x)

    @property
    def predictions(self):
        return self._predictions


class LinearDepthTwo(Machine):
    """docstring for LinearDepth2"""

    def __init__(self, keras_input_layer, **kwargs):
        super(LinearDepthTwo, self).__init__(keras_input_layer, **kwargs)
        x = ToComplex64()(keras_input_layer)
        x = Flatten()(x)
        flat_input = x
        first_layer = ComplexDense(10, use_bias=False)
        second_layer = ComplexDense(1, use_bias=False)
        x = first_layer(x)
        self._predictions = second_layer(x)
        second_layer_jacobian = x
        num_of_first_layer_params = numpy.prod([int(s) for s in first_layer.kernel.shape])
        first_layer_jacobian = Lambda(
            lambda y: tensorflow.reshape(
                tensorflow.matmul(tensorflow.reshape(y, (-1, 1)),
                                  second_layer.kernel,
                                  transpose_b=True),
                (-1, num_of_first_layer_params)))(flat_input)
        self.manual_jacobian = Concatenate()([first_layer_jacobian, second_layer_jacobian])

    @property
    def predictions(self):
        return self._predictions
