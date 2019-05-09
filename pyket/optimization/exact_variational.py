import tensorflow
import numpy as np

from ..exact.utils import binary_array_to_decimal_array, decimal_array_to_binary_array, fsum, complex_norm_log_fsum_exp


class ExactVariational(object):
    """docstring for ExactVariational"""
    def __init__(self, model, operator, batch_size):
        super(ExactVariational, self).__init__()
        self.model = model
        self.operator = operator
        self._build_wave_function_arrays(model.input_shape[1:])
        self._build_batch_arrays(batch_size)
        self._build_local_connections()
        self._build_batch_local_connections_arrays()
        self._graph = tensorflow.get_default_graph()
        self.current_energy = None
        self.current_local_energy_variance = None

    def _build_wave_function_arrays(self, input_size):
        self.input_size = input_size
        self.number_of_spins = np.prod(self.input_size)
        self.num_of_states = 2 ** self.number_of_spins
        self.wave_function = np.zeros(shape=(self.num_of_states, ), dtype=np.complex128)
        self.psi_squared = np.zeros_like(self.wave_function, dtype=np.complex128)
        self.probs = np.zeros_like(self.wave_function, dtype=np.float64)
        self.log_probs = np.zeros_like(self.wave_function, dtype=np.float64)
        self.energies = np.zeros_like(self.wave_function, dtype=np.complex128)
        # self.local_energies_variance = np.zeros_like(self.wave_function, dtype=np.float64)
        self.probs_mult_energy_mean = np.zeros_like(self.wave_function, dtype=np.complex128)
        self.energy_grad_coefficients = np.zeros_like(self.wave_function, dtype=np.complex128)
        self.energies_squared = np.zeros_like(self.wave_function, dtype=np.complex128)
        self.states = decimal_array_to_binary_array(np.arange(self.num_of_states), self.number_of_spins, False).reshape((self.num_of_states, ) + self.input_size)
        self.wave_function_norm_squared = None

    def _build_batch_arrays(self, batch_size):
        if batch_size > self.num_of_states:
            batch_size = self.num_of_states
        if self.num_of_states % batch_size != 0:
            raise Exception('In exact the batch size must divide the total number of states in the system')
        self.batch_size = batch_size
        self.num_of_batch_until_full_cycle = self.num_of_states / self.batch_size
        self.batch_complex_local_energies = np.zeros((self.batch_size, ), dtype=np.complex128)
        self.batch_complex_local_energies_squared = np.zeros((self.batch_size, ), dtype=np.complex128)
    
    def _build_batch_local_connections_arrays(self):
        self.log_values = np.zeros((self.states_hamiltonian_values.shape[0], self.batch_size), dtype=np.complex128)
        self.log_val_add = np.zeros_like(self.log_values, dtype=np.complex128)
        self.val_mult = np.zeros_like(self.log_values, dtype=np.complex128)
        self.batch_complex_local_energies_before_sum = np.zeros_like(self.log_values, dtype=np.complex128)
        self.batch_complex_local_energies_squared_before_sum = np.zeros_like(self.log_values, dtype=np.complex128)
        self.batch_real_energies = np.zeros_like(self.log_values, dtype=np.float64)
    
    def _calculate_max_number_of_local_connections(self):
        max_number_of_local_connections = 0
        for i in range(0, self.num_of_states, self.batch_size):
            local_connections, _, _ = self.operator.find_conn(self.states[i:i+self.batch_size, ...])
            number_of_local_connections = len(local_connections)
            if number_of_local_connections > max_number_of_local_connections:
                max_number_of_local_connections = number_of_local_connections
        return max_number_of_local_connections

    def _build_local_connections(self):
        max_number_of_local_connections = self.operator.max_number_of_local_connections 
        if max_number_of_local_connections is None:
            max_number_of_local_connections = self._calculate_max_number_of_local_connections()
        self.states_local_connections = np.zeros(shape=(max_number_of_local_connections, self.num_of_states) + self.input_size, dtype=np.int32)
        self.states_idx_local_connections = np.zeros(shape=(max_number_of_local_connections, self.num_of_states), 
            dtype=np.int32)
        self.states_hamiltonian_values = np.zeros(shape=(max_number_of_local_connections, self.num_of_states), dtype=np.complex128)
        for i in range(0, self.num_of_states, self.batch_size):
            local_connections, hamiltonian_values, all_use_conn = self.operator.find_conn(self.states[i:i+self.batch_size, ...])
            self.states_hamiltonian_values[:local_connections.shape[0], i:i+self.batch_size] = hamiltonian_values
            self.states_local_connections[:local_connections.shape[0], i:i+self.batch_size, ...] = (local_connections + 1) // 2
            self.states_idx_local_connections[:, i:i+self.batch_size] = binary_array_to_decimal_array(self.states_local_connections[:, i:i+self.batch_size, ...].reshape((-1, self.number_of_spins))).reshape((-1, self.batch_size))

    def _update_wave_function_arrays(self):
        for i in range(0, self.num_of_states, self.batch_size):
            self.wave_function[i:i+self.batch_size] = self.model.predict(self.states[i:i+self.batch_size, ...])[:, 0]
        np.multiply(self.wave_function, 2.0, out=self.psi_squared)
        wave_function_log_norm_squared = complex_norm_log_fsum_exp(self.psi_squared)   
        self.wave_function_norm_squared = np.exp(wave_function_log_norm_squared)
        np.subtract(np.real(self.psi_squared), wave_function_log_norm_squared, out=self.log_probs)
        np.exp(self.log_probs, out=self.probs)
    
    def _update_local_energy(self):
        for i in range(0, self.num_of_states, self.batch_size):
            for j in range(self.states_local_connections.shape[0]):
                self.log_values[j, :] = self.wave_function[self.states_idx_local_connections[j, i:i+self.batch_size]]
            np.add(np.conj(self.log_values), self.log_values[0, :], out=self.log_val_add)
            np.exp(self.log_val_add, out=self.val_mult)
            np.sum(np.multiply(np.conj(self.states_hamiltonian_values[:, i:i+self.batch_size]), 
                self.val_mult, out=self.batch_complex_local_energies_before_sum), axis=0, out=self.batch_complex_local_energies)
            np.sum(np.multiply(np.conj(self.states_hamiltonian_values[:, i:i+self.batch_size]) ** 2, 
                self.val_mult, out=self.batch_complex_local_energies_squared_before_sum), axis=0, out=self.batch_complex_local_energies_squared)
            self.energies[i:i+self.batch_size] = self.batch_complex_local_energies / self.wave_function_norm_squared
            self.energies_squared[i:i+self.batch_size] = self.batch_complex_local_energies_squared
        self.current_energy = fsum(self.energies)
        self.current_local_energy_variance = fsum(self.energies_squared) / self.wave_function_norm_squared - self.current_energy*self.current_energy
        np.multiply(self.probs.astype(np.complex128), self.current_energy, out=self.probs_mult_energy_mean)
        np.subtract(self.energies, self.probs_mult_energy_mean, out=self.energy_grad_coefficients)

    def machine_updated(self):
        with self._graph.as_default():
            self._update_wave_function_arrays()
            self._update_local_energy()

    def __call__(self):
        while True:
            self.machine_updated()
            for i in range(0, self.num_of_states, self.batch_size):
                yield self.states[i:i+self.batch_size], self.energy_grad_coefficients[i:i+self.batch_size]