import numpy
from tensorflow.keras.callbacks import Callback

from ...exact.utils import fdot


class ExactSigmaZ(Callback):
    def __init__(self, generator, log_every_batch=True, **kwargs):
        super(ExactSigmaZ, self).__init__(**kwargs)
        self.generator = generator
        self.log_every_batch = log_every_batch
        self.batch_iter = 0
        self._prepare_sigma_z_vals()

    def _prepare_sigma_z_vals(self):
        total_spins_per_sample = numpy.prod(self.generator.states.shape[1:])
        axis_to_sum = tuple((range(1, len(self.generator.states.shape))))
        self._abs_sigma_z_vals = numpy.absolute(self.generator.states.sum(axis=axis_to_sum)) / total_spins_per_sample
        self._sigma_z_vals = self.generator.states.sum(axis=axis_to_sum) / total_spins_per_sample

    def add_sigma_z_logs(self, logs, generator, prefix=""):
        logs['observables/abs_sigma_z' ] = fdot(self._abs_sigma_z_vals, self.generator.probs)
        logs['observables/sigma_z'] = fdot(self._sigma_z_vals, self.generator.probs)
        
    def on_batch_end(self, batch, logs={}):
        if self.log_every_batch and self.batch_iter % self.generator.num_of_batch_until_full_cycle == 0:
            self.add_sigma_z_logs(logs, self.generator)
        
    def on_epoch_end(self, batch, logs={}):
        self.add_sigma_z_logs(logs, self.generator)
        