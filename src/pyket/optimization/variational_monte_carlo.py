import time

import tensorflow
import numpy


class VariationalMonteCarlo(object):
    """docstring for VariationalMonteCarlo"""
    def __init__(self, model, operator, sampler, mini_batch_size=None):
        super(VariationalMonteCarlo, self).__init__()
        self.model = model
        self.operator = operator
        self.sampler = sampler
        self._batch_size = sampler.batch_size
        self._graph = tensorflow.get_default_graph()
        self.current_local_energy = numpy.zeros((self._batch_size), dtype=numpy.complex128)
        self.current_energy = None
        self.current_local_energy_variance = None
        self.current_batch = None
        if mini_batch_size is None:
            mini_batch_size = sampler.batch_size
        self._mini_batch_size = mini_batch_size

    def _get_flat_local_connections_log_values(self, local_connections, all_use_conn):
        local_connections_reshape = numpy.moveaxis(local_connections, 1, 0).reshape((-1,) + local_connections.shape[2:])
        flat_conn = local_connections_reshape[all_use_conn.astype(numpy.bool).T.flatten(), ...]
        return self.model.predict(flat_conn, batch_size=self._mini_batch_size)[:, 0]

    def _update_batch_local_energy_for_unbalanced_local_connections(self, local_connections, hamiltonian_values, all_use_conn):
        flat_log_values = self._get_flat_local_connections_log_values(local_connections, all_use_conn)
        conn_per_sample = all_use_conn.sum(axis=0).astype(numpy.int32)
        idx = 0
        for i in range(self._batch_size):
            sample_log_values = flat_log_values[idx:idx + conn_per_sample[i]]
            idx += conn_per_sample[i]
            sample_val_division = numpy.exp(sample_log_values - sample_log_values[0])
            self.current_local_energy[i] = numpy.multiply(hamiltonian_values[all_use_conn.astype(numpy.bool)[:, i],i], sample_val_division).sum()
        self.current_energy, self.current_local_energy_variance = numpy.mean(self.current_local_energy), numpy.var(numpy.real(self.current_local_energy))
    
    def _update_batch_local_energy_for_balanced_local_connections(self, local_connections, hamiltonian_values, all_use_conn):
        flat_conn = local_connections.reshape((-1, ) + self.model.input_shape[1:])
        flat_log_values = self.model.predict(flat_conn, batch_size=self._mini_batch_size)[:, 0]
        log_values = flat_log_values.reshape(local_connections.shape[0:2])
        log_val_diff = log_values - log_values[0, :]
        connections_division = numpy.exp(log_val_diff)
        self.current_local_energy = numpy.multiply(hamiltonian_values, connections_division).sum(axis=0)
        self.current_energy, self.current_local_energy_variance = numpy.mean(self.current_local_energy), numpy.var(numpy.real(self.current_local_energy))

    def _update_batch_local_energy(self):
        local_connections, hamiltonian_values, all_use_conn = self.operator.find_conn(self.current_batch)
        if all_use_conn.mean() < 0.95:
            self._update_batch_local_energy_for_unbalanced_local_connections(local_connections, hamiltonian_values, all_use_conn)
        else:
            self._update_batch_local_energy_for_balanced_local_connections(local_connections, hamiltonian_values, all_use_conn)

    def _next_batch(self):
        with self._graph.as_default():
            self.start_time = time.time()
            self.sampler.next_batch()
            self.current_batch = self.sampler.batch
            self.sampling_end_time = time.time()
            self._update_batch_local_energy()
            self.local_energy_end_time = time.time()
            return self.current_batch, numpy.conj(self.current_local_energy - numpy.mean(self.current_local_energy)) / self._batch_size

    def __call__(self):
        while True:
            yield self._next_batch()