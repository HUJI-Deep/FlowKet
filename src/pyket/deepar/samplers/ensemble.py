from .base_sampler import Sampler

import numpy


class Ensemble(Sampler):
    """docstring for Ensemble"""

    def __init__(self, sampler_list):
        super(Ensemble, self).__init__(sampler_list[0].input_size, len(sampler_list) * sampler_list[0].batch_size)
        self.sampler_list = sampler_list
        assert numpy.all([sampler.batch_size == sampler_list[0].batch_size for sampler in sampler_list])

    def __next__(self):
        return numpy.concatenate([next(sampler) for sampler in self.sampler_list], axis=0)
