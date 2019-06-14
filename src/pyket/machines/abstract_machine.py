import abc

import tensorflow
from tensorflow.keras.layers import Lambda
from tensorflow.python.ops.parallel_for import gradients

from pyket.layers import ToComplex64
from ..deepar.layers import LambdaWithOneToOneTopology, CombineAutoregressiveConditionals, \
    NormalizeInLogSpace, PlusMinusOneToOneHot
from ..deepar.graph_analysis.dependency_graph import assert_valid_probabilistic_model


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
        one_hot_input = PlusMinusOneToOneHot()(keras_input_layer)
        one_hot_input = ToComplex64()(one_hot_input)
        self._predictions = CombineAutoregressiveConditionals(name='wave_function')(
            [self.conditional_log_wave_function, one_hot_input])

    @property
    def predictions(self):
        return self._predictions

    @property
    @abc.abstractmethod
    def conditional_log_wave_function(self):
        pass


class AutoNormalizedAutoregressiveMachine(AutoregressiveMachine):
    def __init__(self, keras_input_layer, **kwargs):
        self._conditional_log_wave_function = NormalizeInLogSpace(norm_type=2, name='conditional_log_wave_function')\
            (self.unnormalized_conditional_log_wave_function)
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


class AutoregressiveWrapper(AutoNormalizedAutoregressiveMachine):
    def __init__(self, conditional_wave_functions_model, **kwargs):
        super(AutoNormalizedAutoregressiveMachine, self).__init__(conditional_wave_functions_model.input, **kwargs)
        self.conditional_wave_functions_model = conditional_wave_functions_model

    @property
    def unnormalized_conditional_log_wave_function(self):
        return self.conditional_wave_functions_model.output


def keras_conditional_wave_functions_to_wave_function(conditional_wave_functions_model):
    assert_valid_probabilistic_model(conditional_wave_functions_model)
    return AutoregressiveWrapper(conditional_wave_functions_model)
