from .operator import OperatorOnGrid

import itertools

import numpy


class Ising(OperatorOnGrid):
    """docstring for Ising"""

    def __init__(self, h=1.0, j=1.0, **kwargs):
        super(Ising, self).__init__(**kwargs)
        self.h = h
        self.j = j       
        self.max_number_of_local_connections = numpy.prod(self.hilbert_state_shape) + 1
    
    def find_conn(self, sample):
        sample_conn = numpy.tile(sample, self.hilbert_state_shape + (1, ) * (len(self.hilbert_state_shape) + 1))
        padding_size = ((0, 0), ) + ((0, 1), ) * len(self.hilbert_state_shape)
        if self.pbc:
            padded_sample = numpy.pad(sample, padding_size, mode='wrap')
        else:
            padded_sample = numpy.pad(sample, padding_size, mode='constant', constant_values=0) 
        num_of_samples = sample.shape[0]
        mel = numpy.zeros_like(sample, dtype=numpy.float32)
        # todo support general hilbert_state_shape dimention
        assert(len(self.hilbert_state_shape) <= 2) 
        # for i in itertools.product(*[range(dim_size) for dim_size in self.hilbert_state_shape ]):
        if len(self.hilbert_state_shape) == 1:
            for i in range(self.hilbert_state_shape[0]):
                sample_conn[i, :, i] *= -1
                mel[:, i] -= self.j * sample[:, i] * padded_sample[:, i + 1]
        else:    
            for i in range(self.hilbert_state_shape[0]):
                for j in range(self.hilbert_state_shape[1]):
                    sample_conn[i, j, :, i, j] *= -1
                    if self.hilbert_state_shape[0] > 1:
                        mel[:, i, j] -= self.j * sample[:, i, j] * padded_sample[:, i + 1, j]
                    if self.hilbert_state_shape[1] > 1:
                        mel[:, i, j] -= self.j * sample[:, i, j] * padded_sample[:, i, j + 1]
        self_mel = mel.sum(axis=tuple(range(1, len(sample.shape))))
        conn = sample_conn.reshape((-1, ) + sample.shape)
        all_conn = numpy.concatenate((numpy.expand_dims(sample, axis=0), conn), axis=0)
        mel = numpy.full((all_conn.shape[0], num_of_samples), -self.h)
        mel[0, :] = self_mel
        return all_conn, mel, numpy.ones_like(mel, dtype=numpy.bool)
