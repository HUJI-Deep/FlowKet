from tensorflow.keras.callbacks import Callback


class Runtimes(Callback):
    def __init__(self, generator, **kwargs):
        super(Runtimes, self).__init__(**kwargs)
        self.generator = generator
        
    def add_energy_stats_to_logs(self, logs, generator):
        logs['times/wave_function_update'] = self.generator.wave_function_update_end_time - self.generator.machine_updated_start_time
        logs['times/local_energy'] = self.generator.local_energy_update_end_time - self.generator.wave_function_update_end_time
        logs['times/gradients'] = self.generator.gradient_update_end_time - self.generator.local_energy_update_end_time
        logs['times/total'] = self.generator.gradient_update_end_time - self.generator.machine_updated_start_time
        
    def on_batch_end(self, batch, logs={}):
        if self.log_in_batch_or_epoch and ((batch % self.generator.num_of_batch_until_full_cycle) == 0):
            self.add_energy_stats_to_logs(logs, self.generator)
        
    def on_epoch_end(self, batch, logs={}):
        if not self.log_in_batch_or_epoch:
            self.add_energy_stats_to_logs(logs, self.generator)
