from pyket.machines import SimpleConvNetAutoregressive1D
from pyket.exact.utils import to_log_wave_function_vector

import numpy
import scipy
import pytest
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

ONE_DIM_INPUT = Input(shape=(16,), dtype='int8')


@pytest.mark.parametrize('machine_input, machine', [
    (ONE_DIM_INPUT,
     SimpleConvNetAutoregressive1D(ONE_DIM_INPUT, depth=7, num_of_channels=16, weights_normalization=False))
])
def test_autoregressive_have_normalize_distribution(machine_input, machine):
    model = Model(inputs=machine_input, outputs=machine.predictions)
    log_wave_function = to_log_wave_function_vector(model)
    log_distribution = numpy.real(2 * log_wave_function)
    log_distribution_sum = scipy.special.logsumexp(log_distribution)
    distribution_sum = numpy.exp(log_distribution_sum)
    assert distribution_sum == pytest.approx(1.0, 0.00001)
