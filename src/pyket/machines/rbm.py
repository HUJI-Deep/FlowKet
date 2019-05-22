from .abstract_machine import Machine

import tensorflow
from tensorflow.keras.layers import Add, Flatten, Lambda

from ..layers import TranslationInvariantComplexDense, ComplexDense, ToComplex64
from ..layers.complex.tensorflow_ops import lncosh


class RBM(Machine):
    def __init__(self, keras_input_layer, alpha=1.0, stddev=1.0, **kwargs):
        super(RBM, self).__init__(keras_input_layer, **kwargs)
        x = ToComplex64()(self.keras_input_layer)
        x = Flatten()(x)
        self._lnthetas = ComplexDense(units=alpha, activation=lncosh,
                                      multiply_units_by_input_dim=True,
                                      kernel_initializer=tensorflow.random_normal_initializer(stddev=stddev),
                                      bias_initializer=tensorflow.random_normal_initializer(stddev=stddev))(x)
        x = ComplexDense(units=1, use_bias=False,
                         kernel_initializer=tensorflow.random_normal_initializer(stddev=stddev))(x)
        y = Lambda(lambda t: tensorflow.reduce_sum(t, axis=-1, keepdims=True))(self._lnthetas)
        self._predictions = Add()([x, y])

    @property
    def predictions(self):
        return self._predictions


class RBMSym(Machine):
    def __init__(self, keras_input_layer, alpha=1, stddev=1.0, **kwargs):
        super(RBMSym, self).__init__(keras_input_layer, **kwargs)
        x = ToComplex64()(self.keras_input_layer)
        self._lnthetas = TranslationInvariantComplexDense(units=alpha, activation=lncosh,
                                                          kernel_initializer=tensorflow.random_normal_initializer(
                                                              stddev=stddev),
                                                          bias_initializer=tensorflow.random_normal_initializer(
                                                              stddev=stddev))(x)
        x = Flatten()(x)
        x = Lambda(lambda t: tensorflow.reduce_sum(t, axis=-1, keepdims=True))(x)
        x = ComplexDense(units=1, use_bias=False)(x)
        y = Lambda(lambda t: tensorflow.reduce_sum(t, axis=-1, keepdims=True))(self._lnthetas)
        self._predictions = Add()([x, y])

    @property
    def predictions(self):
        return self._predictions
