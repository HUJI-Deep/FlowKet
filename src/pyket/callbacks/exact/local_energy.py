import numpy
from tensorflow.keras.callbacks import Callback


class ExactLocalEnergy(Callback):
    def __init__(self, generator, true_ground_state_energy=None, log_every_batch=True, **kwargs):
        super(ExactLocalEnergy, self).__init__(**kwargs)
        self.generator = generator
        self.true_ground_state_energy = true_ground_state_energy
        self.log_every_batch = log_every_batch
        self.batch_iter = 0

    def add_energy_stats_to_logs(self, logs, generator):
        logs['energy/energy'] = numpy.real(generator.current_energy)
        logs['energy/local_energy_variance'] = numpy.real(generator.current_local_energy_variance)
        if self.true_ground_state_energy is not None:
            logs['energy/relative_error'] = (self.true_ground_state_energy - numpy.real(generator.current_energy)) / self.true_ground_state_energy
        
    def on_batch_end(self, batch, logs={}):
        if self.log_every_batch and self.batch_iter % self.generator.num_of_batch_until_full_cycle == 0:
            self.add_energy_stats_to_logs(logs, self.generator)
        self.batch_iter += 1
        
    def on_epoch_end(self, batch, logs={}):
        self.add_energy_stats_to_logs(logs, self.generator)
