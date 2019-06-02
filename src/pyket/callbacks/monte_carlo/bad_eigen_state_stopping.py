import warnings

import numpy
from tensorflow.keras.callbacks import Callback


class BadEigenStateStopping(Callback):
    def __init__(self, ground_state_energy_upper_bound, variance_tol=1e-2,
                 relative_error_to_stop=0.1, min_epoch=10, **kwargs):
        super(BadEigenStateStopping, self).__init__(**kwargs)
        self.ground_state_energy_upper_bound = ground_state_energy_upper_bound
        self.variance_tol = variance_tol
        self.relative_error_to_stop = relative_error_to_stop
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if 'val_energy/energy' in logs:
            energy = logs['val_energy/energy']
            variance = logs['val_energy/local_energy_variance']
        elif 'energy/energy' in logs:
            energy = logs['energy/energy']
            variance = logs['energy/local_energy_variance']
        else:
            warnings.warn("Can't find local energy stats, skipping bad eigen state early stopping",
                          RuntimeWarning)
            return
        if epoch < self.min_epoch:
            return
        relative_error = (energy - self.ground_state_energy_upper_bound) / numpy.abs(
            self.ground_state_energy_upper_bound)
        if relative_error > self.relative_error_to_stop and variance < self.variance_tol:
            self.model.stop_training = True
