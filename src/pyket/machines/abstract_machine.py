import abc

import tensorflow
from tensorflow.keras.layers import Lambda
from tensorflow.python.ops.parallel_for import gradients

from pyket.layers import LambdaWithOneToOneTopology


class Machine(abc.ABC):
    def __init__(self, keras_input_layer, use_pfor=False):
        super(Machine, self).__init__()
        self.keras_input_layer = keras_input_layer
        self.use_pfor = use_pfor

    @property
    @abc.abstractmethod
    def predictions(self):
        pass

    def predictions_jacobian(self, params):
        def jacobian(x):
            return gradients.jacobian(tensorflow.real(x), params, use_pfor=self.use_pfor)
        return Lambda(jacobian)(self.predictions)


class AutoregressiveMachine(Machine):
    def __init__(self, keras_input_layer, **kwargs):
        super(AutoregressiveMachine, self).__init__(keras_input_layer, **kwargs)

        def wave_function(x):
            x, x_input = x[0], x[1]
            one_hot_input = tensorflow.one_hot((1 - tensorflow.cast(x_input, tensorflow.int32)) // 2, 2, on_value=1.0,
                                               off_value=0.0, axis=-1)
            one_hot_input_complex = tensorflow.cast(one_hot_input, dtype=x.dtype)
            to_sum_axis = list(range(1, len(x_input.shape) + 1))
            x = tensorflow.reduce_sum(x * one_hot_input_complex, axis=to_sum_axis, keepdims=True)
            return tensorflow.reshape(x, (-1, 1))

        self._predictions = Lambda(wave_function, name='wave_function')(
            [self.conditional_log_wave_function, self.keras_input_layer])

    @property
    def predictions(self):
        return self._predictions

    @property
    @abc.abstractmethod
    def conditional_log_wave_function(self):
        pass


class AutoNormalizedAutoregressiveMachine(AutoregressiveMachine):
    def __init__(self, keras_input_layer, **kwargs):
        def normalize_prob_func(x):
            if x.dtype.is_complex:
                x_real, x_imag = tensorflow.real(x), tensorflow.imag(x)
                norm = 0.5 * tensorflow.reduce_logsumexp(x_real * 2, axis=-1, keepdims=True)
                x_real = x_real - norm
                return tensorflow.complex(x_real, x_imag)
            else:
                norm = 0.5 * tensorflow.reduce_logsumexp(x * 2, axis=-1, keepdims=True)
                return x - norm

        self._conditional_log_wave_function = LambdaWithOneToOneTopology(
            normalize_prob_func, name='conditional_log_wave_function')(
            self.unnormalized_conditional_log_wave_function)
        self._conditional_log_probs = LambdaWithOneToOneTopology(
            lambda x: tensorflow.real(x) * 2.0)(self._conditional_log_wave_function)
        super(AutoNormalizedAutoregressiveMachine, self).__init__(keras_input_layer, **kwargs)

    @property
    def conditional_log_wave_function(self):
        return self._conditional_log_wave_function

    @property
    def conditional_log_probs(self):
        return self._conditional_log_probs

    @property
    @abc.abstractmethod
    def unnormalized_conditional_log_wave_function(self):
        pass


class DirectSamplingMachine(abc.ABC):
    @property
    @abc.abstractmethod
    def samples(self):
        pass
