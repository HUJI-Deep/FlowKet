import functools
import time

import numpy
import tensorflow
from tensorflow.python.keras import backend as K

from ..observables.monte_carlo import Observable


class VariationalMonteCarlo(object):
    """docstring for VariationalMonteCarlo"""

    def __init__(self, model, operator, sampler, mini_batch_size=None):
        super(VariationalMonteCarlo, self).__init__()
        self.model = model
        self.operator = operator
        self.set_sampler(sampler, mini_batch_size)
        self._graph = tensorflow.get_default_graph()
        self._session = K.get_session()
        self.current_batch = None
        self.wave_function = functools.partial(self.model.predict, batch_size=self._mini_batch_size)
        self.energy_observable = Observable(operator)

    def set_sampler(self, sampler, mini_batch_size=None):
        self.sampler = sampler
        self._batch_size = sampler.batch_size
        if mini_batch_size is None:
            mini_batch_size = sampler.batch_size
        self._mini_batch_size = mini_batch_size

    def _update_batch_local_energy(self):
        self.current_energy, self.current_local_energy_variance, self.current_local_energy = \
            self.energy_observable.estimate(self.wave_function, self.current_batch)

    def loss_coefficients(self):
        local_energy_minus_mean = self.current_local_energy - self.current_energy
        return numpy.conj(local_energy_minus_mean)

    def next(self):
        K.set_session(self._session)
        with self._graph.as_default():
            self.start_time = time.time()
            self.current_batch = next(self.sampler)
            self.sampling_end_time = time.time()
            self._update_batch_local_energy()
            self.local_energy_end_time = time.time()
            return self.current_batch, self.loss_coefficients() / self._batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def to_generator(self):
        while True:
            yield next(self)
