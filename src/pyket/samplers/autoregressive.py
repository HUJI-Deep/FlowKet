import itertools

from .base_sampler import Sampler

from tqdm import tqdm
import numpy


def raster_autoregressive_order(input_size):
    return itertools.product(*[range(dim_size) for dim_size in input_size])


class AutoregressiveSampler(Sampler):
    """docstring for AutoregressiveSampler"""

    def __init__(self, conditional_log_probs_machine, batch_size,
                 use_progress_bar=False, autoregressive_ordering=None, **kwargs):
        super(AutoregressiveSampler, self).__init__(input_size=conditional_log_probs_machine.input_shape[1:],
                                                    batch_size=batch_size, **kwargs)
        self.conditional_log_probs_machine = conditional_log_probs_machine
        self.use_progress_bar = use_progress_bar
        if autoregressive_ordering is None:
            autoregressive_ordering = raster_autoregressive_order
        self.autoregressive_ordering = autoregressive_ordering

    def next_batch(self, random_batch=None):
        if random_batch is None:
            random_batch = numpy.random.rand(*((self.batch_size,) + self.input_size))
        progress = tqdm if self.use_progress_bar else lambda x: x
        for i in progress(list(self.autoregressive_ordering(self.input_size))):
            log_probs = self.conditional_log_probs_machine.predict(self.batch, batch_size=self.mini_batch_size)
            if len(i) == 1:
                h = i[0]
                self.batch[:, h] = 2 * (numpy.exp(log_probs[:, h, 0]) > random_batch[:, h]) - 1
            elif len(i) == 2:
                h, w = i
                self.batch[:, h, w] = 2 * (numpy.exp(log_probs[:, h, w, 0]) > random_batch[:, h, w]) - 1
            else:
                # todo support generalautoregressive models with more than 2 dims
                raise Exception('AutoregressiveSampler support dims <= 2 ')
