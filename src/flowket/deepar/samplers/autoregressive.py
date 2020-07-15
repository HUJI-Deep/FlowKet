import copy

from .base_sampler import Sampler
from ..ordering import raster as raster_ordering

from tqdm import tqdm
import numpy


class AutoregressiveSampler(Sampler):
    """docstring for AutoregressiveSampler"""

    def __init__(self, conditional_log_probs_machine, batch_size,
                 use_progress_bar=False, autoregressive_ordering=None, zero_base=True, **kwargs):
        super(AutoregressiveSampler, self).__init__(input_size=conditional_log_probs_machine.input_shape[1:],
                                                    batch_size=batch_size, **kwargs)
        self.conditional_log_probs_machine = conditional_log_probs_machine
        self.use_progress_bar = use_progress_bar
        if autoregressive_ordering is None:
            autoregressive_ordering = raster_ordering
        self.autoregressive_ordering = autoregressive_ordering
        self.zero_base = zero_base

    def copy_with_new_batch_size(self, batch_size, mini_batch_size=None):
        new_sampler = copy.copy(self)
        new_sampler._set_batch_size(batch_size, mini_batch_size)
        return new_sampler

    def __next__(self):
        batch = numpy.zeros((self.batch_size,) + self.input_size)
        random_batch = numpy.random.rand(*((self.batch_size,) + self.input_size))
        progress = tqdm if self.use_progress_bar else lambda x: x
        for i in progress(list(self.autoregressive_ordering(self.input_size))):
            log_probs = self.conditional_log_probs_machine.predict(batch, batch_size=self.mini_batch_size)
            if len(i) == 1:
                h = i[0]
                batch[:, h] = numpy.exp(log_probs[:, h, 0]) > random_batch[:, h]
                if not self.zero_base:
                    batch[:, h] = 2 * batch[:, h] - 1
            elif len(i) == 2:
                h, w = i
                batch[:, h, w] = numpy.exp(log_probs[:, h, w, 0]) > random_batch[:, h, w]
                if not self.zero_base:
                    batch[:, h, w] = 2 * batch[:, h, w] - 1
            else:
                # todo support general autoregressive models with more than 2 dims
                raise Exception('AutoregressiveSampler support dims <= 2 ')
        return batch
