import numpy
from tensorflow.keras.callbacks import Callback


class MCMCStats(Callback):
    def __init__(self, generator, log_in_batch_or_epoch=True, **kwargs):
        super(MCMCStats, self).__init__(**kwargs)
        self.generator = generator
        self.log_in_batch_or_epoch = log_in_batch_or_epoch

    def add_mcmc_logs(self, logs):
        r_hat, _, correlations_sum, effective_sample_size = self.generator.sampler.calc_r_hat_value(
            numpy.real(self.generator.current_local_energy))
        logs['mcmc/acceptance_ratio'] = self.generator.sampler.acceptance_ratio
        logs['mcmc/energy_r_hat'] = r_hat
        logs['mcmc/energy_effective_sample_size'] = effective_sample_size
        logs['mcmc/energy_correlations_sum'] = correlations_sum

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if self.log_in_batch_or_epoch:
            self.add_mcmc_logs(logs)

    def on_epoch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if not self.log_in_batch_or_epoch:
            self.add_mcmc_logs(logs)
