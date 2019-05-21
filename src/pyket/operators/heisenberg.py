from .operator import OperatorOnGrid

import numpy


class Heisenberg(OperatorOnGrid):
    """docstring for Heisenberg"""

    def __init__(self, total_sz=0.0, unitary_rotation=True, **kwargs):
        super(Heisenberg, self).__init__(**kwargs)
        if unitary_rotation:
            self.off_diag = -2.0
        else:
            self.off_diag = 2.0
        self.total_sz = total_sz
        self.total_size = numpy.prod(self.hilbert_state_shape)
        self.find_conn_calculators = {}
        self.max_number_of_local_connections = self.total_size * len(self.hilbert_state_shape) + 1

    def random_states(self, num_of_states):
        if self.total_sz is not None:
            size = self.total_size // 2
            states = numpy.zeros((num_of_states, self.total_size))
            for i in range(num_of_states):
                states[i, ...] = numpy.concatenate(
                    [numpy.ones((size + int(self.total_sz),)), numpy.full((size - int(self.total_sz),), -1)], axis=0)
                numpy.random.shuffle(states[i, ...])
            states = states.reshape((num_of_states,) + self.hilbert_state_shape)
            assert numpy.all(states.sum(axis=tuple(range(1, len(sample.shape)))) == self.total_sz)
            return states
        return super(Heisenberg, self).random_states(num_of_states)

    def use_state(self, state):
        if self.total_sz is None:
            return True
        return state.sum() == self.total_sz

    def find_conn(self, sample):
        batch_size = sample.shape[0]
        if batch_size not in self.find_conn_calculators:
            calculator = HeisenbergFindConn(self, batch_size, pbc=self.pbc)
            self.find_conn_calculators[batch_size] = calculator
        return self.find_conn_calculators[batch_size].find_conn(sample)


class HeisenbergFindConn(object):
    """docstring for HeisenbergFindConn"""

    def __init__(self, ham, batch_size, pbc=True):
        super(HeisenbergFindConn, self).__init__()
        self.ham = ham
        self.pbc = pbc
        self.batch_size = batch_size
        num_of_conn = self.ham.max_number_of_local_connections
        self.all_conn = numpy.zeros((num_of_conn, batch_size) + self.ham.hilbert_state_shape)
        self.all_mel = numpy.zeros((num_of_conn, batch_size))
        self.all_use_conn = numpy.zeros((num_of_conn, batch_size))
        # todo support general hilbert_state_shape dimention
        dim = len(ham.hilbert_state_shape)
        assert (dim == 2)
        shape = self.ham.hilbert_state_shape[0], self.ham.hilbert_state_shape[1], dim, batch_size, \
                self.ham.hilbert_state_shape[0], \
                self.ham.hilbert_state_shape[1]
        self.sample_conn = self.all_conn[1:, ...].view().reshape(shape)
        shape = self.ham.hilbert_state_shape[0], self.ham.hilbert_state_shape[1], dim, batch_size
        self.use_conn = self.all_use_conn[1:, ...].view().reshape(shape)
        self.all_use_conn[0, :] = True
        self.mel = self.all_mel[1:, ...].view().reshape(shape)
        self.self_mel = numpy.zeros((batch_size,) + self.ham.hilbert_state_shape)
        self.tmp_arr = numpy.zeros((batch_size,))

    def find_conn(self, sample):
        assert sample.shape[0] == self.batch_size
        self.all_conn[:] = sample[numpy.newaxis, ...]
        self.calc_conn_and_mel(sample, self.self_mel)
        numpy.sum(self.self_mel, axis=(1, 2), out= self.all_mel[0, :])
        return self.all_conn.view(), self.all_mel.view(), self.all_use_conn.view()

    def calc_conn_and_mel(self, sample, self_mel):
        for i in range(self.ham.hilbert_state_shape[0]):
            for j in range(self.ham.hilbert_state_shape[1]):
                dim_idx = 0
                if self.ham.hilbert_state_shape[0] > 1:
                    if i + 1 < self.ham.hilbert_state_shape[0]:
                        numpy.multiply(sample[:, i, j], sample[:, i + 1, j], out=self.tmp_arr)
                        self_mel[:, i, j] += self.tmp_arr
                        self.use_conn[i, j, dim_idx, :] = (sample[:, i, j] != sample[:, i + 1, j])
                        self.sample_conn[i, j, dim_idx, :, i, j] = sample[:, i + 1, j]
                        self.sample_conn[i, j, dim_idx, :, i + 1, j] = sample[:, i, j]
                    elif self.pbc:
                        numpy.multiply(sample[:, i, j], sample[:, 0, j], out=self.tmp_arr)
                        self_mel[:, i, j] += self.tmp_arr
                        self.use_conn[i, j, dim_idx, :] = (sample[:, i, j] != sample[:, 0, j])
                        self.sample_conn[i, j, dim_idx, :, i, j] = sample[:, 0, j]
                        self.sample_conn[i, j, dim_idx, :, 0, j] = sample[:, i, j]
                    else:
                        self.use_conn[i, j, dim_idx, :] = False
                    self.mel[i, j, dim_idx, :] = numpy.where(self.use_conn[i, j, dim_idx, :], self.ham.off_diag, 0)
                    dim_idx += 1
                if self.ham.hilbert_state_shape[1] > 1:
                    if j + 1 < self.ham.hilbert_state_shape[1]:
                        numpy.multiply(sample[:, i, j], sample[:, i, j + 1], out=self.tmp_arr)
                        self_mel[:, i, j] += self.tmp_arr
                        self.use_conn[i, j, dim_idx, :] = (sample[:, i, j] != sample[:, i, j + 1])
                        self.sample_conn[i, j, dim_idx, :, i, j] = sample[:, i, j + 1]
                        self.sample_conn[i, j, dim_idx, :, i, j + 1] = sample[:, i, j]
                    elif self.pbc:
                        numpy.multiply(sample[:, i, j], sample[:, i, 0], out=self.tmp_arr)
                        self_mel[:, i, j] += self.tmp_arr
                        self.use_conn[i, j, dim_idx, :] = (sample[:, i, j] != sample[:, i, 0])
                        self.sample_conn[i, j, dim_idx, :, i, j] = sample[:, i, 0]
                        self.sample_conn[i, j, dim_idx, :, i, 0] = sample[:, i, j]
                    else:
                        self.use_conn[i, j, dim_idx, :] = False
                    self.mel[i, j, dim_idx, :] = numpy.where(self.use_conn[i, j, dim_idx, :], self.ham.off_diag, 0)
