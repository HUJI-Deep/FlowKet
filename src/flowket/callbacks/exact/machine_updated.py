from tensorflow.keras.callbacks import Callback


class MachineUpdated(Callback):
    def __init__(self, exact_variational, update_in_batch_or_epoch=True, update_local_energy=True, **kwargs):
        super(MachineUpdated, self).__init__(**kwargs)
        self.exact_variational = exact_variational
        self.update_in_batch_or_epoch = update_in_batch_or_epoch
        self.update_local_energy = update_local_energy

    def on_batch_end(self, batch, logs=None):
        if self.update_in_batch_or_epoch:
            if self.update_local_energy:
                self.exact_variational.machine_updated()
            else:
                self.exact_variational._update_wave_function_arrays()

    def on_epoch_end(self, epoch, logs=None):
        if not self.update_in_batch_or_epoch:
            if self.update_local_energy:
                self.exact_variational.machine_updated()
            else:
                self.exact_variational._update_wave_function_arrays()
