import numpy

from .operator import Operator


class NetketOperatorWrapper(Operator):
    def __init__(self, netket_operator, hilbert_state_shape,
                 max_number_of_local_connections=None, should_calc_unused=None):
        assert numpy.prod(hilbert_state_shape) == netket_operator.hilbert.size
        super(NetketOperatorWrapper, self).__init__(hilbert_state_shape)
        self.netket_operator = netket_operator
        self.should_calc_unused = should_calc_unused
        self.max_number_of_local_connections = max_number_of_local_connections
        self.estimated_number_of_local_connections = max_number_of_local_connections
        self._check_if_old_netket()
        if max_number_of_local_connections is None:
            self.estimated_number_of_local_connections = self._calculate_num_of_local_connectios_from_netket_operator()
            if self.should_calc_unused is None:
                self.should_calc_unused = False
        else:
            if self.should_calc_unused is None:
                self.should_calc_unused = True

    def random_states(self, num_of_states):
        import netket
        random_engine = netket.utils.RandomEngine()
        hilbert_space = self.netket_operator.hilbert
        results = numpy.zeros((num_of_states, hilbert_space.size))
        for i in range(num_of_states):
            hilbert_space.random_vals(results[i, :], random_engine)
        return numpy.reshape(results, (num_of_states, ) + self.hilbert_state_shape)

    def _check_if_old_netket(self):
        res = self.netket_operator.get_conn(numpy.array([1]*self.netket_operator.hilbert.size))
        self.old_netket = len(res) == 3

    def _calculate_num_of_local_connectios_from_netket_operator(self):
        random_state = self.random_states(1)
        res = self.netket_operator.get_conn(random_state.flatten())
        if self.old_netket:
            mel, _, _ = res
        else:
            _, mel = res
        return len(mel)

    def new_netket_find_conn(self, sample):
        conn_list, mel_list = [], []
        batch_size = sample.shape[0]
        for i in range(batch_size):
            flat_input = sample[i, ...].flatten()
            sample_conn, sample_mel = self.netket_operator.get_conn(flat_input)
            conn_list.append(sample_conn)
            mel_list.append(sample_mel)
            assert numpy.all(sample_conn[0,:] == flat_input)
        self.estimated_number_of_local_connections = max(numpy.max([len(x) for x in mel_list]), self.estimated_number_of_local_connections)
        all_conn = numpy.zeros((self.estimated_number_of_local_connections,) + sample.shape)
        batch_mel = numpy.zeros((self.estimated_number_of_local_connections, sample.shape[0]), dtype=numpy.complex128)
        for i in range(batch_size):
            all_conn[:len(conn_list[i]), i, ...] = conn_list[i].reshape((-1,) + sample.shape[1:])
            batch_mel[:len(mel_list[i]), i] = mel_list[i]
        if self.should_calc_unused:
            all_conn_use = batch_mel != 0.0
        else:
            all_conn_use = numpy.ones((self.estimated_number_of_local_connections, batch_size), dtype=numpy.bool)
        all_conn_use[0, :] = True
        return all_conn, batch_mel, all_conn_use

    def old_netket_find_conn(self, sample):
        all_conn = numpy.zeros((self.estimated_number_of_local_connections,) + sample.shape)
        batch_mel = numpy.zeros((self.estimated_number_of_local_connections, sample.shape[0]), dtype=numpy.complex128)
        batch_size = sample.shape[0]
        for i in range(batch_size):
            flat_input = sample[i, ...].flatten()
            sample_mel, to_change_idx_list, to_change_vals_list = self.netket_operator.get_conn(flat_input)
            if len(sample_mel) > self.estimated_number_of_local_connections:
                print('wrong max_number_of_local_connections fixing and continue recursively')
                self.estimated_number_of_local_connections = len(sample_mel)
                return self.find_conn(sample)
            sample_mel = sample_mel + [0.0] * (self.estimated_number_of_local_connections - len(sample_mel))
            batch_mel[:, i] = numpy.array(sample_mel)
            self_conn_idx = -1
            for j, (to_change_idx, to_change_vals) in enumerate(zip(to_change_idx_list, to_change_vals_list)):
                conn = flat_input.copy()
                if len(to_change_idx) == 0:
                    self_conn_idx = j
                for to_change_id, to_change_val in zip(to_change_idx, to_change_vals):
                    conn[to_change_id] = to_change_val
                all_conn[j, i, ...] = conn.reshape(sample.shape[1:])
            assert self_conn_idx >= 0
            if self_conn_idx != 0:
                temp_conn = all_conn[0, ...]
                all_conn[0, ...] = all_conn[self_conn_idx, ...]
                all_conn[self_conn_idx, ...] = temp_conn
                tmp_mel = batch_mel[0, i]
                batch_mel[0, i] = batch_mel[self_conn_idx, i]
                batch_mel[self_conn_idx, i] = tmp_mel
        if self.should_calc_unused:
            all_conn_use = batch_mel != 0.0
        else:
            all_conn_use = numpy.ones((self.estimated_number_of_local_connections, batch_size), dtype=numpy.bool)
        all_conn_use[0, :] = True
        return all_conn, batch_mel, all_conn_use

    def find_conn(self, sample):
        if self.old_netket:
            return self.old_netket_find_conn(sample)
        return self.new_netket_find_conn(sample)
