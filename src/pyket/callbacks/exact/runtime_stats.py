import time
from tensorflow.keras.callbacks import Callback


class RuntimeStats(Callback):
    def __init__(self, generator, log_in_batch_or_epoch=True, **kwargs):
        super(RuntimeStats, self).__init__(**kwargs)
        self.generator = generator
        self.log_in_batch_or_epoch = log_in_batch_or_epoch
        
    def add_runtime_stats_to_logs(self, logs):
        gradient_update_end_time = time.time()
        logs['times/wave_function_update'] = self.generator.wave_function_update_end_time - \
                                             self.generator.machine_updated_start_time
        logs['times/local_energy'] = self.generator.local_energy_update_end_time - \
                                     self.generator.wave_function_update_end_time
        logs['times/gradients'] = gradient_update_end_time - self.generator.local_energy_update_end_time
        logs['times/total'] = gradient_update_end_time - self.generator.machine_updated_start_time
        
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if self.log_in_batch_or_epoch and ((batch + 1) % self.generator.num_of_batch_until_full_cycle) == 0:
            self.add_runtime_stats_to_logs(logs)

    def on_epoch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if not self.log_in_batch_or_epoch:
            self.add_runtime_stats_to_logs(logs)
