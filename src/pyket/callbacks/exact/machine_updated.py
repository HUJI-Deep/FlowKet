from tensorflow.keras.callbacks import Callback


class MachineUpdated(Callback):
    def __init__(self, exact_variational, update_in_batch_or_epoch=True, **kwargs):
        super(MachineUpdated, self).__init__(**kwargs)
        self.exact_variational = exact_variational
        self.update_in_batch_or_epoch = update_in_batch_or_epoch

    def on_batch_end(self, batch, logs=None):
        if self.update_in_batch_or_epoch:
            self.exact_variational.machine_updated()

    def on_epoch_end(self, epoch, logs=None):
        if not self.update_in_batch_or_epoch:
            self.exact_variational.machine_updated()
