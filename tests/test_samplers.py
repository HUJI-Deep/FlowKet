from pyket.exact.utils import binary_array_to_decimal_array
from pyket.machines import SimpleConvNetAutoregressive1D, ConvNetAutoregressive2D
from pyket.samplers import AutoregressiveSampler, ExactSampler, FastAutoregressiveSampler, \
    MetropolisHastingsLocal, MetropolisHastingsSampler
from pyket.operators import Operator
from pyket.optimization import ExactVariational

import numpy
import pytest
import tensorflow
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.python.keras import backend as K


ONE_DIM_INPUT = Input(shape=(10,), dtype='int8')
TWO_DIM_INPUT = Input(shape=(4, 3), dtype='int8')
GRAPH = tensorflow.get_default_graph()
BATCH_SIZE = 2 ** 10


class IdentityOperator(Operator):
    def __init__(self, hilbert_state_shape):
        super(IdentityOperator, self).__init__(hilbert_state_shape)

    def find_conn(self, sample):
        all_conn = numpy.expand_dims(sample, axis=0)
        mel = numpy.full((all_conn.shape[0], sample.shape[0]), 1.0)
        return all_conn, mel, numpy.ones_like(mel, dtype=numpy.bool)


def sampler_factory(sampler_class, machine, machine_input, num_of_samples):
    if issubclass(sampler_class, MetropolisHastingsSampler):
        input_size = numpy.product(list(K.int_shape(machine_input)[1:]))
        model = Model(inputs=machine_input, outputs=machine.predictions)
        return sampler_class(model, num_of_samples, mini_batch_size=BATCH_SIZE,
                             num_of_chains=num_of_samples // (2 ** 7),
                             unused_sampels=input_size)
    else:
        model = Model(inputs=machine_input, outputs=machine.conditional_log_probs)
        return sampler_class(model, num_of_samples, mini_batch_size=BATCH_SIZE)


@pytest.mark.parametrize('sampler_class, machine_input, machine_class, machine_args', [
    (AutoregressiveSampler, ONE_DIM_INPUT, SimpleConvNetAutoregressive1D,
     {'depth': 5, 'num_of_channels': 16, 'weights_normalization': False}),
    (AutoregressiveSampler, TWO_DIM_INPUT, ConvNetAutoregressive2D,
     {'depth': 2, 'num_of_channels': 16, 'weights_normalization': False}),
    (FastAutoregressiveSampler, ONE_DIM_INPUT, SimpleConvNetAutoregressive1D,
     {'depth': 5, 'num_of_channels': 16, 'weights_normalization': False}),
    (FastAutoregressiveSampler, TWO_DIM_INPUT, ConvNetAutoregressive2D,
     {'depth': 2, 'num_of_channels': 16, 'weights_normalization': False}),
    (MetropolisHastingsLocal, ONE_DIM_INPUT, SimpleConvNetAutoregressive1D,
     {'depth': 5, 'num_of_channels': 16, 'weights_normalization': False}),
    (MetropolisHastingsLocal, TWO_DIM_INPUT, ConvNetAutoregressive2D,
     {'depth': 2, 'num_of_channels': 16, 'weights_normalization': False}),
])
def test_sampler_by_l1(sampler_class, machine_input, machine_class, machine_args):
    # this test based on https://arxiv.org/pdf/1308.3946.pdf
    with GRAPH.as_default():
        machine = machine_class(machine_input, **machine_args)
        model = Model(inputs=machine_input, outputs=machine.predictions)
        operator = IdentityOperator(list(K.int_shape(machine_input)[1:]))
        exact_variational = ExactVariational(model, operator, BATCH_SIZE)
        exact_variational._update_wave_function_arrays()
        exact_sampler = ExactSampler(exact_variational, BATCH_SIZE)
        num_of_samples = max(2 ** 16, 16 * exact_variational.num_of_states)
        exact_sampler._set_batch_size(num_of_samples, mini_batch_size=BATCH_SIZE)
        batch_from_exact_sampler= next(exact_sampler)
        sampler = sampler_factory(sampler_class, machine, machine_input, num_of_samples)
        batch_from_sampler = next(sampler)
        sampler_chosen_idx = binary_array_to_decimal_array(batch_from_sampler.reshape((num_of_samples, -1)))
        exact_sampler_chosen_idx = binary_array_to_decimal_array(batch_from_exact_sampler.reshape((num_of_samples, -1)))
        x = numpy.bincount(sampler_chosen_idx.astype(numpy.int), minlength=exact_variational.num_of_states)
        y = numpy.bincount(exact_sampler_chosen_idx.astype(numpy.int), minlength=exact_variational.num_of_states)
        z = ((numpy.square(x - y) - x - y) / (x + y + 1e-20)).sum()
        assert z <= numpy.sqrt(num_of_samples)
