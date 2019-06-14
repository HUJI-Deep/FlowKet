import functools
import time

import tensorflow
from tensorflow.python.keras import backend as K
import numpy


class Observable(object):
    """docstring for ExactVariational"""

    def __init__(self, operator, wave_function):
        super(Observable, self).__init__()
        self.operator = operator
        self.wave_function = wave_function

    def _get_flat_local_connections_log_values(self, local_connections, all_use_conn):
        local_connections_reshape = numpy.moveaxis(local_connections, 1, 0).reshape((-1,)
                                                                                    + local_connections.shape[2:])
        flat_conn = local_connections_reshape[all_use_conn.astype(numpy.bool).T.flatten(), ...]
        return self.wave_function(flat_conn)[:, 0]

    def estimate_optimized_for_unbalanced_local_connections(self, local_connections, hamiltonian_values,
                                                            all_use_conn):
        batch_size = all_use_conn.shape[1]
        local_values = numpy.zeros((batch_size,), dtype=numpy.complex128)
        flat_log_values = self._get_flat_local_connections_log_values(local_connections, all_use_conn)
        conn_per_sample = all_use_conn.sum(axis=0).astype(numpy.int32)
        idx = 0
        for i in range(batch_size):
            sample_log_values = flat_log_values[idx:idx + conn_per_sample[i]]
            idx += conn_per_sample[i]
            sample_val_division = numpy.exp(sample_log_values - sample_log_values[0])
            local_values[i] = numpy.multiply(hamiltonian_values[all_use_conn.astype(numpy.bool)
                                                                [:, i], i], sample_val_division).sum()
        return local_values

    def estimate_optimized_for_balanced_local_connections(self, local_connections, hamiltonian_values):
        flat_conn = local_connections.reshape((-1,) + self.operator.hilbert_state_shape)
        flat_log_values = self.wave_function(flat_conn)[:, 0]
        log_values = flat_log_values.reshape(local_connections.shape[0:2])
        log_val_diff = log_values - log_values[0, :]
        connections_division = numpy.exp(log_val_diff)
        return numpy.multiply(hamiltonian_values, connections_division).sum(axis=0)

    def estimate(self, sample):
        local_connections, hamiltonian_values, all_use_conn = self.operator.find_conn(sample)
        if all_use_conn.mean() < 0.95:
            local_values = self.estimate_optimized_for_unbalanced_local_connections(local_connections,
                                                                                    hamiltonian_values,
                                                                                    all_use_conn)
        else:
            local_values = self.estimate_optimized_for_balanced_local_connections(local_connections, hamiltonian_values)
        mean_value = numpy.mean(local_values)
        variance = numpy.var(numpy.real(local_values))
        return mean_value, variance, local_values


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
        self.energy_observable = Observable(operator, self.wave_function)

    def set_sampler(self, sampler, mini_batch_size=None):
        self.sampler = sampler
        self._batch_size = sampler.batch_size
        if mini_batch_size is None:
            mini_batch_size = sampler.batch_size
        self._mini_batch_size = mini_batch_size

    def _update_batch_local_energy(self):
        self.current_energy, self.current_local_energy_variance, self.current_local_energy = \
            self.energy_observable.estimate(self.current_batch)

    def next(self):
        K.set_session(self._session)
        with self._graph.as_default():
            self.start_time = time.time()
            self.current_batch = next(self.sampler)
            self.sampling_end_time = time.time()
            self._update_batch_local_energy()
            self.local_energy_end_time = time.time()
            local_energy_minus_mean = self.current_local_energy - self.current_energy
            return self.current_batch, numpy.conj(local_energy_minus_mean) / self._batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def to_generator(self):
        while True:
            yield next(self)
