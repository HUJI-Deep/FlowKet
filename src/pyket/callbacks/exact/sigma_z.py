import numpy
from tensorflow.keras.callbacks import Callback

from ...exact.utils import fdot


class ExactSigmaZ(Callback):
    def __init__(self, generator, log_in_batch_or_epoch=True, **kwargs):
        super(ExactSigmaZ, self).__init__(**kwargs)
        self.generator = generator
        self.log_in_batch_or_epoch = log_in_batch_or_epoch
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
        if self.log_in_batch_or_epoch and ((batch % self.generator.num_of_batch_until_full_cycle) == 0):
            self.add_sigma_z_logs(logs, self.generator)
        
    def on_epoch_end(self, batch, logs={}):
        if not self.log_in_batch_or_epoch:
            self.add_sigma_z_logs(logs, self.generator)
