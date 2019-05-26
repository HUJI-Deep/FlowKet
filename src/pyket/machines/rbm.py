import abc

from .abstract_machine import Machine
from ..layers import TranslationInvariantComplexDense, ComplexDense, ToComplex64, ToComplex128
from ..layers.complex.tensorflow_ops import lncosh

import tensorflow
from tensorflow.keras.layers import Add, Flatten, Lambda
import tensorflow.keras.backend as K


class RBMBase(Machine, abc.ABC):
    def __init__(self, keras_input_layer, alpha=1, stddev=1.0, use_float64_ops=False, **kwargs):
        super(RBMBase, self).__init__(keras_input_layer, **kwargs)
        self.use_float64_ops = use_float64_ops
        self.layers_dtype = tensorflow.complex128 if self.use_float64_ops else tensorflow.complex64
        self.alpha = alpha
        if self.use_float64_ops:
            K.set_floatx('float64')
            x = ToComplex128()(self.keras_input_layer)
        else:
            x = ToComplex64()(self.keras_input_layer)
        self.initializer = tensorflow.random_normal_initializer(stddev=stddev, dtype=self.layers_dtype.real_dtype)
        self._predictions = self.build_predictions(x)

    @property
    def predictions(self):
        return self._predictions

    @abc.abstractmethod
    def build_predictions(self, x):
        pass


class RBM(RBMBase):
    def build_predictions(self, x):
        x = Flatten()(x)
        self._lnthetas = ComplexDense(units=self.alpha, activation=lncosh,
                                      multiply_units_by_input_dim=True,
                                      dtype=self.layers_dtype,
                                      kernel_initializer=self.initializer,
                                      bias_initializer=self.initializer)(x)
        x = ComplexDense(units=1,
                         use_bias=False,
                         dtype=self.layers_dtype,
                         kernel_initializer=self.initializer)(x)
        y = Lambda(lambda t: tensorflow.reduce_sum(t, axis=-1, keepdims=True))(self._lnthetas)
        return Add()([x, y])


class RBMSym(RBMBase):
    def build_predictions(self, x):
        self._lnthetas = TranslationInvariantComplexDense(units=self.alpha,
                                                          activation=lncosh,
                                                          dtype=self.layers_dtype,
                                                          kernel_initializer=self.initializer,
                                                          bias_initializer=self.initializer)(x)
        x = Flatten()(x)
        x = Lambda(lambda t: tensorflow.reduce_sum(t, axis=-1, keepdims=True))(x)
        x = ComplexDense(units=1,
                         use_bias=False,
                         dtype=self.layers_dtype,
                         kernel_initializer=self.initializer)(x)
        y = Lambda(lambda t: tensorflow.reduce_sum(t, axis=-1, keepdims=True))(self._lnthetas)
        return Add()([x, y])
