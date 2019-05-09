import abc

import numpy


class Sampler(abc.ABC):
    """docstring for Sampler"""
    def __init__(self, input_size, batch_size, mini_batch_size=None):
        super(Sampler, self).__init__()
        self.input_size = input_size
        self.set_batch_size(batch_size, mini_batch_size=mini_batch_size)
    
    def set_batch_size(self, batch_size, mini_batch_size=None):
        if mini_batch_size is None:
            mini_batch_size = batch_size
        if batch_size < mini_batch_size:
            mini_batch_size = batch_size
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.batch = numpy.zeros((self.batch_size,) + self.input_size)
        
    @abc.abstractmethod
    def next_batch(self):
        pass