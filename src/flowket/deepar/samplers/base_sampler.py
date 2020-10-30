import abc
import copy


class Sampler(abc.ABC):
    """docstring for Sampler"""

    def __init__(self, input_size, batch_size, mini_batch_size=None):
        super(Sampler, self).__init__()
        self.input_size = input_size
        self._set_batch_size(batch_size, mini_batch_size=mini_batch_size)

    def _set_batch_size(self, batch_size, mini_batch_size=None):
        if mini_batch_size is None:
            mini_batch_size = batch_size
        if batch_size < mini_batch_size:
            mini_batch_size = batch_size
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

    def copy_with_new_batch_size(self, batch_size, mini_batch_size=None):
        new_sampler = copy.copy(self)
        new_sampler._set_batch_size(batch_size, mini_batch_size)
        return new_sampler

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        pass
