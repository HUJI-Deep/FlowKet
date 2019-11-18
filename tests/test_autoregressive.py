from flowket.machines import SimpleConvNetAutoregressive1D, ConvNetAutoregressive2D
from flowket.exact.utils import to_log_wave_function_vector

import numpy
import scipy
import pytest
import tensorflow
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

ONE_DIM_INPUT = Input(shape=(16,), dtype='int8')
TWO_DIM_INPUT = Input(shape=(4, 4), dtype='int8')
SMALL_TWO_DIM_INPUT = Input(shape=(4, 3), dtype='int8')

graph = tensorflow.get_default_graph()


@pytest.mark.parametrize('machine_input, machine_class, machine_args', [
    (ONE_DIM_INPUT, SimpleConvNetAutoregressive1D, {'depth': 7, 'num_of_channels': 16, 'weights_normalization': False}),
    (TWO_DIM_INPUT, ConvNetAutoregressive2D, {'depth': 4, 'num_of_channels': 16, 'weights_normalization': False})
])
def test_autoregressive_have_normalize_distribution(machine_input, machine_class, machine_args):
    with graph.as_default():
        machine = machine_class(machine_input, **machine_args)
        model = Model(inputs=machine_input, outputs=machine.predictions)
        log_wave_function = to_log_wave_function_vector(model)
        log_distribution = numpy.real(2 * log_wave_function)
        log_distribution_sum = scipy.special.logsumexp(log_distribution)
        distribution_sum = numpy.exp(log_distribution_sum)
        assert distribution_sum == pytest.approx(1.0, 0.00001)
