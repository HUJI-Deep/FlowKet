import time

from ..exact.utils import binary_array_to_decimal_array, decimal_array_to_binary_array, fsum, complex_norm_log_fsum_exp

import tensorflow
import numpy as np


class ExactObservable(object):
    """docstring for ExactVariational"""

    def __init__(self, exact_variational, operator, calculate_variance_of_the_local_operator=False):
        super(ExactObservable, self).__init__()
        self.exact_variational = exact_variational
        self.operator = operator
        self.calculate_variance_of_the_local_operator = calculate_variance_of_the_local_operator
        self._build_wave_function_arrays()
        self._build_local_connections()
        self._build_batch_local_connections_arrays()

    def _build_wave_function_arrays(self):
        self.energies = np.zeros_like(shape=(self.exact_variational.num_of_states,), dtype=np.complex128)
        if self.calculate_variance_of_the_local_operator:
            self.naive_energies = np.zeros_like(self.energies, dtype=np.complex128)
            self.naive_local_energy_minus_energy = np.zeros_like(self.energies, dtype=np.complex128)
            self.naive_local_energy_minus_energy_squared = np.zeros_like(self.energies, dtype=np.float64)
            self.probs_mult_local_energy_variance = np.zeros_like(self.energies, dtype=np.float64)

    def _build_batch_local_connections_arrays(self):
        self.batch_complex_local_energies = np.zeros((self.exact_variational.batch_size, ), dtype=np.complex128)
        self.batch_naive_complex_local_energies = np.zeros((self.exact_variational.batch_size, ), dtype=np.complex128)
        self.log_values = np.zeros((self.states_hamiltonian_values.shape[0],
                                    self.exact_variational.batch_size), dtype=np.complex128)
        self.log_val_add = np.zeros_like(self.log_values, dtype=np.complex128)
        self.val_mult = np.zeros_like(self.log_values, dtype=np.complex128)
        self.batch_complex_local_energies_before_sum = np.zeros_like(self.log_values, dtype=np.complex128)
        self.batch_real_energies = np.zeros_like(self.log_values, dtype=np.float64)
        if self.calculate_variance_of_the_local_operator:
            self.log_val_sub = np.zeros_like(self.log_values, dtype=np.complex128)
            self.val_div = np.zeros_like(self.log_values, dtype=np.complex128)
            self.batch_naive_complex_local_energies_before_sum = np.zeros_like(self.log_values, dtype=np.complex128)

    def _build_local_connections(self):
        max_number_of_local_connections = self.operator.max_number_of_local_connections
        if max_number_of_local_connections is None:
            max_number_of_local_connections = self.calculate_max_number_of_local_connections()
        self.states_local_connections = np.zeros(
            shape=(max_number_of_local_connections, self.exact_variational.num_of_states) +
                  self.exact_variational.input_size, dtype=np.int32)
        self.states_idx_local_connections = np.zeros(shape=(max_number_of_local_connections, self.exact_variational.num_of_states),
                                                     dtype=np.int32)
        self.states_hamiltonian_values = np.zeros(shape=(max_number_of_local_connections, self.exact_variational.num_of_states),
                                                  dtype=np.complex128)
        for i in range(0, self.exact_variational.num_of_states, self.exact_variational.batch_size):
            local_connections, hamiltonian_values, all_use_conn = self.operator.find_conn(
                self.exact_variational.states[i:i + self.exact_variational.batch_size, ...])
            self.states_hamiltonian_values[:local_connections.shape[0], i:i + self.exact_variational.batch_size] = hamiltonian_values
            self.states_local_connections[:local_connections.shape[0], i:i + self.exact_variational.batch_size, ...] = (
                                                                                                                               local_connections + 1) // 2
            self.states_idx_local_connections[:, i:i + self.exact_variational.batch_size] = binary_array_to_decimal_array(
                self.states_local_connections[:, i:i + self.exact_variational.batch_size, ...].reshape(
                    (-1, self.exact_variational.number_of_spins))).reshape((-1, self.exact_variational.batch_size))

    def calculate_max_number_of_local_connections(self):
        max_number_of_local_connections = 0
        for i in range(0, self.exact_variational.num_of_states, self.exact_variational.batch_size):
            local_connections, _, _ = self.operator.find_conn(
                self.exact_variational.states[i:i + self.exact_variational.batch_size, ...])
            number_of_local_connections = len(local_connections)
            if number_of_local_connections > max_number_of_local_connections:
                max_number_of_local_connections = number_of_local_connections
        return max_number_of_local_connections

    def update_local_energy(self):
        for i in range(0, self.exact_variational.num_of_states, self.exact_variational.batch_size):
            for j in range(self.states_local_connections.shape[0]):
                self.log_values[j, :] = self.exact_variational.wave_function[self.states_idx_local_connections[j, i:i+self.exact_variational.batch_size]]
            np.add(np.conj(self.log_values), self.log_values[0, :], out=self.log_val_add)
            np.exp(self.log_val_add, out=self.val_mult)
            np.sum(np.multiply(np.conj(self.states_hamiltonian_values[:, i:i+self.exact_variational.batch_size]),
                self.val_mult, out=self.batch_complex_local_energies_before_sum), axis=0, out=self.batch_complex_local_energies)
            self.energies[i:i+self.exact_variational.batch_size] = self.batch_complex_local_energies / self.exact_variational.wave_function_norm_squared
            if self.calculate_variance_of_the_local_operator:
                np.subtract(self.log_values, self.log_values[0, :], out=self.log_val_sub)
                np.exp(self.log_val_sub, out=self.val_div)
                np.sum(np.multiply(self.states_hamiltonian_values[:, i:i+self.exact_variational.batch_size],
                    self.val_div, out=self.batch_naive_complex_local_energies_before_sum), axis=0, out=self.batch_naive_complex_local_energies)
                self.naive_energies[i:i+self.exact_variational.batch_size] = self.batch_naive_complex_local_energies
        self.current_energy = fsum(self.energies)
        if self.calculate_variance_of_the_local_operator:
            np.subtract(self.naive_energies, self.current_energy, out=self.naive_local_energy_minus_energy)
            np.multiply(np.real(self.naive_local_energy_minus_energy), np.real(self.naive_local_energy_minus_energy), out=self.naive_local_energy_minus_energy_squared)
            np.multiply(self.naive_local_energy_minus_energy_squared, self.exact_variational.probs, out=self.probs_mult_local_energy_variance)
            self.current_local_energy_variance = fsum(self.probs_mult_local_energy_variance).item()


class ExactVariational(object):
    """docstring for ExactVariational"""
    def __init__(self, model, operator, batch_size):
        super(ExactVariational, self).__init__()
        self.model = model
        self.operator = operator
        self._set_batch_size(batch_size)
        self._build_wave_function_arrays(model.input_shape[1:])
        self._graph = tensorflow.get_default_graph()
        self.energy_observable = ExactObservable(self, operator, calculate_variance_of_the_local_operator=True)

    def _build_wave_function_arrays(self, input_size):
        self.input_size = input_size
        self.number_of_spins = np.prod(self.input_size)
        self.num_of_states = 2 ** self.number_of_spins
        self.wave_function = np.zeros(shape=(self.num_of_states, ), dtype=np.complex128)
        self.psi_squared = np.zeros_like(self.wave_function, dtype=np.complex128)
        self.probs = np.zeros_like(self.wave_function, dtype=np.float64)
        self.log_probs = np.zeros_like(self.wave_function, dtype=np.float64)

        self.probs_mult_energy_mean = np.zeros_like(self.wave_function, dtype=np.complex128)
        self.energy_grad_coefficients = np.zeros_like(self.wave_function, dtype=np.complex128)

        self.states = decimal_array_to_binary_array(np.arange(self.num_of_states), self.number_of_spins, False).reshape((self.num_of_states, ) + self.input_size)
        self.wave_function_norm_squared = None

    def _set_batch_size(self, batch_size):
        if batch_size > self.num_of_states:
            batch_size = self.num_of_states
        if self.num_of_states % batch_size != 0:
            raise Exception('In exact the batch size must divide the total number of states in the system')
        self.batch_size = batch_size
        self.num_of_batch_until_full_cycle = self.num_of_states // self.batch_size

    def _update_wave_function_arrays(self):
        for i in range(0, self.num_of_states, self.batch_size):
            self.wave_function[i:i+self.batch_size] = self.model.predict(self.states[i:i+self.batch_size, ...])[:, 0]
        np.multiply(self.wave_function, 2.0, out=self.psi_squared)
        wave_function_log_norm_squared = complex_norm_log_fsum_exp(self.psi_squared)   
        self.wave_function_norm_squared = np.exp(wave_function_log_norm_squared)
        np.subtract(np.real(self.psi_squared), wave_function_log_norm_squared, out=self.log_probs)
        np.exp(self.log_probs, out=self.probs)
    
    def _update_local_energy(self):
        self.energy_observable.update_local_energy()
        np.multiply(self.probs.astype(np.complex128), self.energy_observable.current_energy, out=self.probs_mult_energy_mean)
        np.subtract(self.energy_observable.energies, self.probs_mult_energy_mean, out=self.energy_grad_coefficients)

    def machine_updated(self):
        with self._graph.as_default():
            self.machine_updated_start_time = time.time()
            self._update_wave_function_arrays()
            self.wave_function_update_end_time = time.time()
            self._update_local_energy()
            self.local_energy_update_end_time = time.time()

    def to_generator(self):
        while True:
            self.machine_updated()
            for i in range(0, self.num_of_states, self.batch_size):
                yield self.states[i:i+self.batch_size], self.energy_grad_coefficients[i:i+self.batch_size]

    def __iter__(self):
        return self.to_generator()
