import time
from tensorflow.keras.callbacks import Callback


class RuntimeStats(Callback):
    def __init__(self, generator, log_in_batch_or_epoch=True, **kwargs):
        super(RuntimeStats, self).__init__(**kwargs)
        self.generator = generator
        self.log_in_batch_or_epoch = log_in_batch_or_epoch
        
    def add_runtime_stats_to_logs(self, logs):
        gradient_update_end_time = time.time()
        logs['times/sampling'] = self.generator.sampling_end_time - self.generator.start_time
        logs['times/local_energy'] = self.generator.local_energy_end_time - self.generator.sampling_end_time
        logs['times/gradients'] = gradient_update_end_time - self.generator.local_energy_end_time
        logs['times/total'] = gradient_update_end_time - self.generator.start_time
        
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if self.log_in_batch_or_epoch:
            self.add_runtime_stats_to_logs(logs)

    def on_epoch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if not self.log_in_batch_or_epoch:
            self.add_runtime_stats_to_logs(logs)
