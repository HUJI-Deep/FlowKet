import abc
import math


class MiniBatchGenerator(object):
    """docstring for MiniBatchGenerator"""
    def __init__(self, batch_size, mini_batch_size):
        super(MiniBatchGenerator, self).__init__()
        self.set_batch_size(batch_size, mini_batch_size)

    @abc.abstractmethod
    def next_batch(self):
        # should return batch & loss coefficients
        pass

    def set_batch_size(self, batch_size, mini_batch_size=None):
        if mini_batch_size is None:
            mini_batch_size = batch_size
        if mini_batch_size > batch_size:
            mini_batch_size = batch_size
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self._idx = self.batch_size
        self.update_params_frequency = math.ceil(self.batch_size / float(self.mini_batch_size))
        return self.update_params_frequency

    def next_mini_batch_size(self):
        if self._idx + self.mini_batch_size > self.batch_size:
            self._x, self._y = self.next_batch()
            self._idx = 0
        self._idx += self.mini_batch_size
        return self._x[self._idx - self.mini_batch_size:self._idx, ...], self._y[self._idx - self.mini_batch_size:self._idx]

    def next(self):
        return self.next_mini_batch_size()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def to_generator(self):
        while True:
            yield next(self)
