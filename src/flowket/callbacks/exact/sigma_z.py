import numpy
from tensorflow.keras.callbacks import Callback

from ...exact.utils import fdot


class ExactSigmaZ(Callback):
    def __init__(self, exact_variational, log_in_batch_or_epoch=True, **kwargs):
        super(ExactSigmaZ, self).__init__(**kwargs)
        self.exact_variational = exact_variational
        self.log_in_batch_or_epoch = log_in_batch_or_epoch
        self._prepare_sigma_z_vals()

    def _prepare_sigma_z_vals(self):
        total_spins_per_sample = numpy.prod(self.exact_variational.states.shape[1:])
        axis_to_sum = tuple((range(1, len(self.exact_variational.states.shape))))
        self._abs_sigma_z_vals = numpy.absolute(self.exact_variational.states.sum(axis=axis_to_sum)) / total_spins_per_sample
        self._sigma_z_vals = self.exact_variational.states.sum(axis=axis_to_sum) / total_spins_per_sample

    def add_sigma_z_logs(self, logs):
        logs['observables/abs_sigma_z' ] = fdot(self._abs_sigma_z_vals, self.exact_variational.probs)
        logs['observables/sigma_z'] = fdot(self._sigma_z_vals, self.exact_variational.probs)
        
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if self.log_in_batch_or_epoch and ((batch % self.exact_variational.num_of_batch_until_full_cycle) == 0):
            self.add_sigma_z_logs(logs)
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if not self.log_in_batch_or_epoch:
            self.add_sigma_z_logs(logs)
