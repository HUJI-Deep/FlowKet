from .abstract_machine import Machine

import tensorflow
from tensorflow.keras.layers import Add, Flatten, Lambda
import tensorflow.keras.backend as K

from ..layers import TranslationInvariantComplexDense, ComplexDense, ToComplex64, ToComplex128
from ..layers.complex.tensorflow_ops import lncosh


class RBM(Machine):
    def __init__(self, keras_input_layer, alpha=1.0, stddev=1.0, use_float64_ops=False, **kwargs):
        super(RBM, self).__init__(keras_input_layer, **kwargs)
        self.use_float64_ops = use_float64_ops
        self.layers_dtype = tensorflow.complex128 if self.use_float64_ops else tensorflow.complex64
        if self.use_float64_ops:
            K.set_floatx('float64')
            x = ToComplex128()(self.keras_input_layer)
        else:
            x = ToComplex64()(self.keras_input_layer)
        x = Flatten()(x)
        initializer = tensorflow.random_normal_initializer(stddev=stddev, dtype=self.layers_dtype.real_dtype)
        self._lnthetas = ComplexDense(units=alpha, activation=lncosh,
                                      multiply_units_by_input_dim=True,
                                      dtype=self.layers_dtype,
                                      kernel_initializer=initializer,
                                      bias_initializer=initializer)(x)
        x = ComplexDense(units=1, use_bias=False, dtype=self.layers_dtype,
                         kernel_initializer=initializer)(x)
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
            K.set_floatx('float64')
            x = ToComplex128()(self.keras_input_layer)
        else:
            x = ToComplex64()(self.keras_input_layer)
        initializer = tensorflow.random_normal_initializer(stddev=stddev, dtype=self.layers_dtype.real_dtype)
        self._lnthetas = TranslationInvariantComplexDense(units=alpha, activation=lncosh, dtype=self.layers_dtype,
                                                          kernel_initializer=initializer ,
                                                          bias_initializer=initializer)(x)
        x = Flatten()(x)
        x = Lambda(lambda t: tensorflow.reduce_sum(t, axis=-1, keepdims=True))(x)
        x = ComplexDense(units=1, use_bias=False, dtype=self.layers_dtype, kernel_initializer=initializer)(x)
        y = Lambda(lambda t: tensorflow.reduce_sum(t, axis=-1, keepdims=True))(self._lnthetas)
        self._predictions = Add()([x, y])

    @property
    def predictions(self):
        return self._predictions
