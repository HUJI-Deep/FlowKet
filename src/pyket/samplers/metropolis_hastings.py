import abc

from pyket.deepar.samplers.base_sampler import Sampler

import numpy


def sum_correlations(correlations):
    for i in range(1, (correlations.shape[0]) // 2):
        if correlations[2 * i - 1] + correlations[2 * i] < 0:
            return correlations[:2 * i].sum()
    return correlations.sum()


class MetropolisHastingsSampler(Sampler):
    """docstring for MetropoliceSampler"""

    def __init__(self, machine, batch_size, num_of_chains=1, unused_sampels=0, discard_ratio=10, **kwargs):
        super(MetropolisHastingsSampler, self).__init__(input_size=machine.input_shape[1:], batch_size=batch_size,
                                                        **kwargs)
        self.machine = machine
        print('using %s parallel samplers' % num_of_chains)
        self.unused_sampels = unused_sampels
        print('unused sampels in each sweep : %s' % unused_sampels)
        self.num_of_chains = num_of_chains
        if self.batch_size % self.num_of_chains != 0:
            raise Exception('Num of samplers must divide the batch size')
        self.sample = numpy.random.choice([-1, 1], size=(self.num_of_chains,) + self.input_size)
        self.candidates = numpy.copy(self.sample)
        self.first_temp_out = numpy.zeros((num_of_chains,), dtype=numpy.complex128)
        self.second_temp_out = numpy.zeros((num_of_chains,), dtype=numpy.complex128)
        self.first_temp_out_real, self.second_temp_out_real = numpy.zeros_like(self.first_temp_out), numpy.zeros_like(
            self.first_temp_out)
        self.accepts = numpy.zeros_like(self.first_temp_out, dtype=numpy.bool)
        self.acceptance_ratio = 1.0
        self.discard_ratio = discard_ratio
        print('discard ratio : %s' % discard_ratio)

    def machine_updated(self):
        self.sample_machine_values = self.machine.predict(self.sample, batch_size=self.mini_batch_size)[:, 0]

    @abc.abstractmethod
    def _sweep(self):
        pass

    def warn_up(self, num_of_iterations):
        for i in range(num_of_iterations):
            for j in range(self.unused_sampels):
                self._sweep()

    def __next__(self):
        batch = numpy.empty((self.batch_size,) + self.input_size)
        batch_divided_to_chains = batch.view().reshape(
            (self.num_of_chains, self.batch_size // self.num_of_chains) + self.input_size)
        self.machine_updated()
        accepts_counter = 0
        sampels_per_chain = self.batch_size // self.num_of_chains
        if self.discard_ratio > 0:
            self.warn_up(sampels_per_chain // self.discard_ratio)
        for i in range(sampels_per_chain):
            for j in range(self.unused_sampels + 1):
                accepts_counter += self._sweep()
            batch_divided_to_chains[:, i, ...] = self.sample
        self.acceptance_ratio = accepts_counter / (self.batch_size * (self.unused_sampels + 1))
        return batch

    def calc_r_hat_value(self, estimated_values):
        # according to page 285 in Bayesian Data Analysis Third Edition
        sampels_per_chain = self.batch_size // self.num_of_chains
        estimated_values_per_chain = estimated_values.view().reshape((self.num_of_chains, sampels_per_chain))
        chains_mean = estimated_values_per_chain.mean(axis=1)
        mean = chains_mean.mean()
        between_diff = chains_mean - mean
        within_diff = estimated_values_per_chain - chains_mean[:, numpy.newaxis]
        between_variance = (sampels_per_chain / (self.num_of_chains - 1.0)) * (between_diff * between_diff).sum()
        if sampels_per_chain == 1:
            print("Can't estimate r_hat and correlations_sum")
            return 1.0, between_variance, 0.0, self.batch_size
        within_variance = (within_diff * within_diff).sum(axis=1).mean() / (sampels_per_chain - 1)
        variance = ((sampels_per_chain - 1) * within_variance + between_variance) / sampels_per_chain
        r_hat = numpy.sqrt(variance / within_variance)
        v = numpy.zeros(sampels_per_chain - 1)
        for t in range(sampels_per_chain - 1):
            t_diff = estimated_values_per_chain[:, t + 1:] - estimated_values_per_chain[:, :sampels_per_chain - t - 1]
            v[t] = (t_diff * t_diff).mean(axis=1).mean()
        correlations = 1 - (v / (2 * variance))
        correlations_sum = sum_correlations(correlations)
        effective_sample_size = self.batch_size / (1 + 2 * correlations_sum)
        return r_hat, variance, correlations_sum, effective_sample_size


class MetropolisHastingsHastingSymmetricProposal(MetropolisHastingsSampler):
    """docstring for MetropoliceHastingSymmetricProposal"""

    @abc.abstractmethod
    def _next_candidates(self):
        pass

    def _sweep(self):
        num_of_chains = self.sample.shape[0]
        self._next_candidates()
        candidates_machine_values = self.machine.predict(self.candidates, batch_size=self.mini_batch_size)[:, 0]
        if not numpy.all(numpy.isfinite(candidates_machine_values)):
            print(candidates_machine_values)
            raise Exception('candidates_machine_values has not finite element')
        log_ratio = numpy.multiply(numpy.subtract(numpy.real(candidates_machine_values),
                                                  numpy.real(self.sample_machine_values),
                                                  out=self.first_temp_out_real), 2.0, out=self.second_temp_out_real)
        numpy.greater(numpy.exp(log_ratio), numpy.random.uniform(size=num_of_chains), out=self.accepts)
        self.sample[self.accepts, ...] = self.candidates[self.accepts, ...]
        self.sample_machine_values[self.accepts] = candidates_machine_values[self.accepts]
        return self.accepts.sum()

    def __init__(self, machine, batch_size, **kwargs):
        super(MetropolisHastingsHastingSymmetricProposal, self).__init__(machine, batch_size, **kwargs)


class MetropolisHastingsLocal(MetropolisHastingsHastingSymmetricProposal):
    """docstring for MetropoliceLocal"""

    def _next_candidates(self):
        i = list(range(self.num_of_chains))
        idx_for_dim = tuple([i] + [numpy.random.randint(low=dim_size, size=self.num_of_chains).tolist() for dim_size in
                                   self.sample.shape[1:]])
        numpy.copyto(self.candidates, self.sample)
        self.candidates[idx_for_dim] *= -1

    def __init__(self, machine, batch_size, **kwargs):
        super(MetropolisHastingsLocal, self).__init__(machine, batch_size, **kwargs)


class MetropolisHastingsGlobal(MetropolisHastingsHastingSymmetricProposal):
    """docstring for MetropolisHastingsGlobal"""

    def _next_candidates(self):
        self.candidates = next(self.global_sampler)

    def __init__(self, machine, batch_size, global_sampler, **kwargs):
        super(MetropolisHastingsGlobal, self).__init__(machine, batch_size,
                                                       num_of_chains=global_sampler.batch_size, **kwargs)
        self.global_sampler = global_sampler


class MetropolisHastingsUniform(MetropolisHastingsHastingSymmetricProposal):
    """docstring for MetropoliceUniform"""

    def _next_candidates(self):
        self.candidates = numpy.random.choice([-1, 1], size=self.sample.shape)

    def __init__(self, machine, batch_size, **kwargs):
        super(MetropolisHastingsUniform, self).__init__(machine, batch_size, **kwargs)


class MetropolisHastingsHamiltonian(MetropolisHastingsSampler):
    """docstring for MetropoliceHamiltonian"""

    def _sweep(self):
        all_conn, mel, use_conn = self.hamiltonian.find_conn(self.sample)
        # todo implement without loops
        num_of_conn = numpy.count_nonzero(use_conn, axis=0)
        for i in range(self.num_of_chains):
            poosible_candidates = all_conn[use_conn[:, i], i, ...]
            self.candidates[i, ...] = poosible_candidates[numpy.random.choice(int(num_of_conn[i])), ...]
        all_conn, mel, use_conn = self.hamiltonian.find_conn(self.candidates)
        candidates_num_of_conn = numpy.count_nonzero(use_conn, axis=0)
        candidates_machine_values = self.machine.predict(self.candidates, batch_size=self.mini_batch_size)[:, 0]
        log_ratio = numpy.multiply(numpy.subtract(numpy.real(candidates_machine_values),
                                                  numpy.real(self.sample_machine_values),
                                                  out=self.first_temp_out_real), 2.0, out=self.second_temp_out_real)
        log_ratio += numpy.log(num_of_conn / candidates_num_of_conn)
        numpy.greater(numpy.exp(log_ratio), numpy.random.uniform(size=self.num_of_chains), out=self.accepts)
        self.sample[self.accepts, ...] = self.candidates[self.accepts, ...]
        self.sample_machine_values[self.accepts] = candidates_machine_values[self.accepts]
        return self.accepts.sum()

    def __init__(self, machine, batch_size, hamiltonian, **kwargs):
        super(MetropolisHastingsHamiltonian, self).__init__(machine, batch_size, **kwargs)
        self.hamiltonian = hamiltonian
        self.sample = self.hamiltonian.random_states(self.num_of_chains)
