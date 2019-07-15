import numpy

from ..deepar.samplers.base_sampler import Sampler
from ..exact.utils import decimal_array_to_binary_array


class ExactSampler(Sampler):
    """docstring for Sampler"""

    def __init__(self, exact_variational, batch_size, **kwargs):
        super(ExactSampler, self).__init__(exact_variational.input_size, batch_size, **kwargs)
        self.exact_variational = exact_variational

    def __next__(self):
        decimal_batch = numpy.random.choice(self.exact_variational.num_of_states,
                                            size=self.batch_size,
                                            p=self.exact_variational.probs)
        binary_batch = decimal_array_to_binary_array(decimal_batch,
                                                     num_of_bits=self.exact_variational.number_of_spins)
        return binary_batch.reshape((self.batch_size,) + self.input_size)


class WaveFunctionSampler(Sampler):
    """docstring for WaveFunctionSampler"""

    def __init__(self, wave_function_vector, input_size, batch_size,  **kwargs):
        super(WaveFunctionSampler, self).__init__(input_size, batch_size, **kwargs)
        self.wave_function_vector = wave_function_vector
        self.log_probs = numpy.real(self.wave_function_vector) * 2.0
        self.probs = numpy.exp(self.log_probs)

    def __next__(self):
        decimal_batch = numpy.random.choice(self.probs.shape[0],
                                            size=self.batch_size,
                                            p=self.probs)
        binary_batch = decimal_array_to_binary_array(decimal_batch,
                                                     num_of_bits=int(numpy.log2(self.probs.shape[0])))
        return binary_batch.reshape((self.batch_size,) + self.input_size)
