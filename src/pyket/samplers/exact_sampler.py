import numpy

from .base_sampler import Sampler
from ..exact.utils import decimal_array_to_binary_array


class ExactSampler(Sampler):
    """docstring for Sampler"""
    def __init__(self, exact_variational, batch_size, **kwargs):
        super(ExactSampler, self).__init__(exact_variational.input_size, batch_size, **kwargs)
        self.exact_variational = exact_variational
    
    def next_batch(self):
        self.batch = decimal_array_to_binary_array(
            numpy.random.choice(self.exact_variational.num_of_states,
                                size=self.batch_size, p=self.exact_variational.probs),
            self.exact_variational.number_of_spins).reshape((self.batch_size, )+self.input_size)