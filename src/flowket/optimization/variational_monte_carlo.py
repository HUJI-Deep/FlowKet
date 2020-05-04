import functools
import time

import numpy
import tensorflow
from tensorflow.python.keras import backend as K

from .mini_batch_generator import MiniBatchGenerator
from ..observables.monte_carlo import Observable, BaseObservable


class VariationalMonteCarlo(MiniBatchGenerator):
    """docstring for VariationalMonteCarlo"""

    def __init__(self, model, operator, sampler, mini_batch_size=None, wave_function_evaluation_batch_size=None, **kwargs):
        super(VariationalMonteCarlo, self).__init__(sampler.batch_size, mini_batch_size, **kwargs)
        if wave_function_evaluation_batch_size is None:
            wave_function_evaluation_batch_size = self.mini_batch_size
        self.model = model
        self.operator = operator
        self.sampler = sampler
        self._graph = tensorflow.get_default_graph()
        self._session = K.get_session()
        self.current_batch = None
        self.wave_function = functools.partial(self.model.predict, batch_size=wave_function_evaluation_batch_size)
        self.energy_observable = operator
        if not isinstance(self.energy_observable, BaseObservable):
            self.energy_observable = Observable(operator)

    def set_sampler(self, sampler, mini_batch_size=None):
        self.sampler = sampler
        return self.set_batch_size(sampler.batch_size, mini_batch_size)

    def _update_batch_local_energy(self):
        self.current_energy, self.current_local_energy_variance, self.current_local_energy = \
            self.energy_observable.estimate(self.wave_function, self.current_batch)

    def loss_coefficients(self):
        local_energy_minus_mean = self.current_local_energy - self.current_energy
        return numpy.conj(local_energy_minus_mean)

    def next_batch(self):
        K.set_session(self._session)
        with self._graph.as_default():
            self.start_time = time.time()
            self.current_batch = next(self.sampler)
            self.sampling_end_time = time.time()
            self._update_batch_local_energy()
            self.local_energy_end_time = time.time()
            return self.current_batch, self.loss_coefficients() / self.batch_size
