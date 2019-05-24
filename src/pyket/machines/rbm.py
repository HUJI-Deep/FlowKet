from .abstract_machine import Machine

import tensorflow
from tensorflow.keras.layers import Add, Flatten, Lambda

from ..layers import TranslationInvariantComplexDense, ComplexDense, ToComplex64, ToComplex128
from ..layers.complex.tensorflow_ops import lncosh


class RBM(Machine):
    def __init__(self, keras_input_layer, alpha=1.0, stddev=1.0, use_float64_ops=False, **kwargs):
        super(RBM, self).__init__(keras_input_layer, **kwargs)
        self.use_float64_ops = use_float64_ops
        self.layers_dtype = tensorflow.complex128 if self.use_float64_ops else tensorflow.complex64
        if self.use_float64_ops:
            x = ToComplex128()(self.keras_input_layer)
        else:
            x = ToComplex64()(self.keras_input_layer)
        x = Flatten()(x)
        self._lnthetas = ComplexDense(units=alpha, activation=lncosh,
                                      multiply_units_by_input_dim=True,
                                      dtype=self.layers_dtype,
                                      kernel_initializer=tensorflow.random_normal_initializer(stddev=stddev),
                                      bias_initializer=tensorflow.random_normal_initializer(stddev=stddev))(x)
        x = ComplexDense(units=1, use_bias=False, dtype=self.layers_dtype,
                         kernel_initializer=tensorflow.random_normal_initializer(stddev=stddev))(x)
        y = Lambda(lambda t: tensorflow.reduce_sum(t, axis=-1, keepdims=True))(self._lnthetas)
        self._predictions = Add()([x, y])

    @property
    def predictions(self):
        return self._predictions


class RBMSym(Machine):
    def __init__(self, keras_input_layer, alpha=1, stddev=1.0, use_float64_ops=False, **kwargs):
        super(RBMSym, self).__init__(keras_input_layer, **kwargs)
        self.use_float64_ops = use_float64_ops
        self.layers_dtype = tensorflow.complex128 if self.use_float64_ops else tensorflow.complex64
        if self.use_float64_ops:
            x = ToComplex128()(self.keras_input_layer)
        else:
            x = ToComplex64()(self.keras_input_layer)
        self._lnthetas = TranslationInvariantComplexDense(units=alpha, activation=lncosh, dtype=self.layers_dtype,
                                                          kernel_initializer=tensorflow.random_normal_initializer(
                                                              stddev=stddev),
                                                          bias_initializer=tensorflow.random_normal_initializer(
                                                              stddev=stddev))(x)
        x = Flatten()(x)
        x = Lambda(lambda t: tensorflow.reduce_sum(t, axis=-1, keepdims=True))(x)
        x = ComplexDense(units=1, use_bias=False, dtype=self.layers_dtype)(x)
        y = Lambda(lambda t: tensorflow.reduce_sum(t, axis=-1, keepdims=True))(self._lnthetas)
        self._predictions = Add()([x, y])

    @property
    def predictions(self):
        return self._predictions
