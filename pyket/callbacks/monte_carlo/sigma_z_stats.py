import numpy
from tensorflow.keras.callbacks import Callback


class SigmaZStats(Callback):
    def __init__(self, generator, validation_generator=None, log_every_batch=True, **kwargs):
        super(SigmaZStats, self).__init__(**kwargs)
        self.generator = generator
        self.validation_generator = validation_generator
        self.log_every_batch = log_every_batch

    def add_sigma_z_logs(self, logs, generator, prefix=""):
        total_spins_per_sample = numpy.prod(generator.current_batch.shape[1:])
        axis_to_sum = tuple((range(1, len(generator.current_batch.shape))))
        logs['%sobservables/abs_sigma_z'  % prefix] = numpy.absolute(generator.current_batch.sum(axis=axis_to_sum)).mean() / total_spins_per_sample
        logs['%sobservables/sigma_z' % prefix] = generator.current_batch.sum(axis=axis_to_sum).mean() / total_spins_per_sample
        
    def on_batch_end(self, batch, logs={}):
        if self.log_every_batch:
            self.add_sigma_z_logs(logs, self.generator)
        
    def on_epoch_end(self, batch, logs={}):
        self.add_sigma_z_logs(logs, self.generator)
        if self.validation_generator is not None:
            self.add_sigma_z_logs(logs, self.validation_generator, 'val_')