import  numpy

from .observable import BaseObservable


def get_flat_local_connections_log_values(wave_function, local_connections, all_use_conn):
    local_connections_reshape = numpy.moveaxis(local_connections, 1, 0).reshape((-1,)
                                                                                + local_connections.shape[2:])
    flat_conn = local_connections_reshape[all_use_conn.astype(numpy.bool).T.flatten(), ...]
    return wave_function(flat_conn)[:, 0]


class Observable(BaseObservable):
    """docstring for ExactVariational"""

    def __init__(self, operator):
        super(Observable, self).__init__()
        self.operator = operator

    def local_values_optimized_for_unbalanced_local_connections(self, wave_function, local_connections,
                                                                hamiltonian_values,
                                                                all_use_conn):
        batch_size = all_use_conn.shape[1]
        local_values = numpy.zeros((batch_size,), dtype=numpy.complex128)
        flat_log_values = get_flat_local_connections_log_values(wave_function, local_connections, all_use_conn)
        conn_per_sample = all_use_conn.sum(axis=0).astype(numpy.int32)
        idx = 0
        for i in range(batch_size):
            sample_log_values = flat_log_values[idx:idx + conn_per_sample[i]]
            idx += conn_per_sample[i]
            sample_val_division = numpy.exp(sample_log_values - sample_log_values[0])
            local_values[i] = numpy.multiply(hamiltonian_values[all_use_conn.astype(numpy.bool)
                                                                [:, i], i], sample_val_division).sum()
        return local_values

    def local_values_optimized_for_balanced_local_connections(self, wave_function, local_connections, hamiltonian_values):
        flat_conn = local_connections.reshape((-1,) + self.operator.hilbert_state_shape)
        flat_log_values = wave_function(flat_conn)[:, 0]
        log_values = flat_log_values.reshape(local_connections.shape[0:2])
        log_val_diff = log_values - log_values[0, :]
        connections_division = numpy.exp(log_val_diff)
        return numpy.multiply(hamiltonian_values, connections_division).sum(axis=0)

    def local_values(self, wave_function, configurations):
        local_connections, hamiltonian_values, all_use_conn = self.operator.find_conn(configurations)
        if all_use_conn.mean() < 0.95:
            return self.local_values_optimized_for_unbalanced_local_connections(wave_function,
                                                                                local_connections,
                                                                                hamiltonian_values,
                                                                                all_use_conn)
        else:
            return self.local_values_optimized_for_balanced_local_connections(wave_function,
                                                                              local_connections,
                                                                              hamiltonian_values)
